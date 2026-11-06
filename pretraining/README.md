# scOntoFM pretraining

This folder reproduces the released checkpoint (`resources/weights/scOntoFM`). Pretraining has four stages:

```
CELLxGENE h5ad ──► preprocess + tokenize ──► tokenized corpus (HF dataset, 29.16M cells)
                                                   │
cl-basic.obo ──► ontology/build_lca_distance.py ──►│ LCA distance matrix (all CL terms)
                                                   ▼
                         triplets/build_celltype_index_map.py ──► CL_id_map.pkl
                                                   ▼
                         triplets/generate_triplets.py ──► triplet_ind_hard_top50.pkl (27,525,118 triplets)
                                                   ▼
                         train_scOntoFM.py (DeepSpeed) ──► final_model_centroid_spearman_dynamicMargin_top50
```

The outputs of stages 1–3 for the released model ship with the repository: `assets/` holds the
large files, which `python scripts/download_assets.py` fetches. Only the tokenized corpus has to be
rebuilt.

| File | Description |
|---|---|
| `train_scOntoFM.py` | Pretraining entry point: model, DeepSpeed config, training loop |
| `losses.py` | `compute_ontology_alignment_loss`, `dynamic_triplet_loss`, `get_cls_embedding` |
| `data.py` | `TripletDataset`, `TripletCollator` (padding + MLM masking) |
| `ontology_targets.py` | Loads the LCA matrix and computes the ontology depth of every CL term |
| `run_pretrain.sh` | Single-node DeepSpeed launcher |
| `ontology/build_lca_distance.py` | Pairwise LCA distances between all Cell Ontology terms |
| `triplets/build_celltype_index_map.py` | `{cell type: {dataset: [row indices]}}` index of the corpus |
| `triplets/generate_triplets.py` | Ontology-guided offline triplet mining |

All commands below run from the repository root.

---

## 1. Pretraining corpus

* **Source.** 234 *Homo sapiens* scRNA-seq datasets from the CZI CELLxGENE Census (2025-01-30
  release), 10x platforms only. The training cells cover 210 tissues, 566 cell types and 58 diseases.
* **Cell filtering.** Primary data only (`is_primary_data == True`). Cells with fewer than 200
  detected genes are removed. The resulting corpus has 29,160,112 cells. Malignant, unannotated
  and obsolete-term cells stay in the corpus but are dropped when the triplet index is built
  (step 3), so training uses 27,525,118 cells.
* **Gene filtering.** Genes detected in fewer than 5 cells are removed.
* **Normalization and tokenization.** Library size is normalized to 10,000 counts. Each cell is
  then tokenized by median-normalized rank: the top 2,000 genes, framed as `[<cls>] … [<eos>]`.
  The vocabulary (`resources/vocab/token_dictionary.pkl`) and the gene medians
  (`resources/vocab/gene_medians.pkl`) were both computed on this corpus.

```bash
# per dataset: filter + normalize, then tokenize with obs metadata
python scripts/preprocess.py --input raw_h5ad/ --output_dir processed_h5ad/ \
    --min_genes_per_cell 200 --min_cells_per_gene 5
python scripts/tokenize_h5ad.py --input processed_h5ad/ --output_dir corpus_arrow/ --format arrow
```

`--format arrow` writes one Arrow file per dataset. Each file holds `input_ids` and the CELLxGENE
obs metadata, including `cell_type_ontology_term_id` and `dataset_id`. Merge these files into a
single HuggingFace dataset saved with `save_to_disk`, for example:

```python
from pathlib import Path
from datasets import Dataset, concatenate_datasets
parts = [Dataset.from_file(str(p)) for p in sorted(Path("corpus_arrow").glob("*.arrow"))]
concatenate_datasets(parts).save_to_disk("scLLM_onto_dataset")
```

Training needs only the `input_ids` and `cell_type_ontology_term_id` columns. The index map also
needs `dataset_id`.

> **Row order matters.** `CL_id_map.pkl` and `triplet_ind_hard_top50.pkl` store *row indices*
> into the corpus. The shipped triplets are valid only for the original `scLLM_onto_dataset`
> (same datasets in the same order). If you rebuild the corpus, rerun steps 3–4.

## 2. Ontology LCA distances

```bash
python pretraining/ontology/build_lca_distance.py \
    --obo resources/ontology/cl-basic.obo \
    --out assets/celltype_similarity_LCA_deepest_allnodes.pkl
```

The script reads every term in `cl-basic.obo` (CL release 2025-07-30, 3,129 non-obsolete terms)
as a DAG, using `is_a` edges plus `relationship` edges. For each pair of terms (u, v), it walks up
the graph and collects their common ancestors. It then picks the **deepest** common ancestor:
greatest depth from the root, with ties broken by shorter total path. The distance is

  Δ(u, v) = d(u, LCA) + d(v, LCA)

Pairs without a common ancestor are stored as NaN. The output is a pickle
`{"Distance": (N, N) array, "nodes": [CL ids]}`. The computation visits all N² pairs (single process; expect hours).

## 3. Cell type → dataset → cell index map

```bash
python pretraining/triplets/build_celltype_index_map.py \
    --dataset_path /path/to/scLLM_onto_dataset --out assets/CL_id_map.pkl
```

The script builds `{CL_id: {dataset_id: [row indices]}}` from the corpus metadata. It removes
`CL:0001064` (malignant cell), `CL:0000651`, `CL:0010003`, `unknown` and `CL:0000402` (`--exclude`).

**Corpus size versus training size.** The released corpus has **29,160,112** cells. Only the
**27,525,118** cells left after this step are used in training: every one of them is the anchor of
exactly one triplet, and positives and negatives are also drawn from this set. The other
1,634,994 cells stay in the corpus but never reach the model:

| `cell_type_ontology_term_id` | Cells | Why excluded |
|---|---:|---|
| `unknown` | 1,039,113 | no cell-type annotation |
| `CL:0001064` malignant cell | 563,344 | not a normal cell type in the ontology hierarchy |
| `CL:0010003` epithelial cell of alveolus of lung | 23,557 | obsolete term in CL 2025-07-30 (not in the LCA matrix) |
| `CL:0000651` mucous neck cell | 8,980 | obsolete term in CL 2025-07-30 (not in the LCA matrix) |

`CL:0000402` (obsolete) is also on the list but does not occur in this corpus. This leaves 566 of
the 570 cell-type labels.

## 4. Offline ontology-guided triplet mining

```bash
python pretraining/triplets/generate_triplets.py \
    --cl_id_map assets/CL_id_map.pkl \
    --lca_path assets/celltype_similarity_LCA_deepest_allnodes.pkl \
    --obo resources/ontology/cl-basic.obo \
    --top_k 50 --seed 42 \
    --out assets/triplet_ind_hard_top50.pkl
```

The script produces **one triplet per cell** (anchor *a* of type *c_a*):

* **Hard-negative cell types.** These are fixed once per cell type. The candidates are all corpus
  cell types except *c_a* itself, its direct parents and children (Δ ≤ 1), and types with no common
  ancestor. They are sorted by Δ(c_a, ·), and the **M = 50** closest are kept (`--top_k`). When
  several types tie at the 50th distance, all tied types are pooled and 50 are sampled at random
  (seed 42). This yields a set of hard to semi-hard negatives that are ontologically close.
* **Positive.** A random cell of type *c_a* from a **different source dataset**, which encourages
  batch-invariant features. A same-dataset cell (never the anchor itself) is used only when *c_a*
  occurs in a single dataset.
* **Negative.** A uniformly sampled cell from the pooled hard-negative cell types.

Cell types are processed in order of ontology depth (shortest path from `CL:0000000`).
Per-depth arrays go to `--out_dir/depth_<d>_all_triplets.npy`, and all triplets go to `--out` as
a list of `(anchor, positive, negative)` row-index tuples. Regenerating from the shipped
`CL_id_map.pkl` and LCA matrix reproduces the shipped `triplet_ind_hard_top50.pkl` exactly
(27,525,118 triplets). The training sampler shuffles triplets, so the depth order is not a
curriculum.

## 5. Model and objectives

**Architecture.** HuggingFace `BertForMaskedLM`, configured as follows:
* 6 layers, hidden size 256, 4 heads, intermediate size 512
* ReLU, dropout 0.02
* vocabulary 19,941, maximum length 2,002 (8.9M parameters)

**Cell embedding.** The last-layer `<cls>` state, written **z** below.

Each micro-batch contains B = 64 triplets per GPU. Anchors, positives and negatives are encoded
in three forward passes, and all three are MLM-masked:
* 15% of gene tokens are selected for masking; `<pad>`, `<cls>` and `<eos>` are never selected.
* Selected tokens are replaced with `<mask>` 80% of the time, a random token 10% of the time, and
  left unchanged 10% of the time.

The loss is

  **L = L_MLM + α · L_triplet + β · L_onto**  with α = β = 1.

* **L_MLM.** Masked-gene cross-entropy on the anchors.
* **Dynamic-margin triplet loss** (`losses.dynamic_triplet_loss`):

  L_triplet = mean( max(0, ‖z_a − z_p‖₂ − ‖z_a − z_n‖₂ + m(c_a, c_n)) )

  m(c_a, c_n) = margin_alpha · ( (1 − Δ̂(c_a, c_n)) + deptĥ(c_a) + 0.1 ),  margin_alpha = 1

  Δ̂ is the LCA distance divided by its maximum, and deptĥ is the ontology depth divided by its
  maximum. The margin grows for ontologically closer (harder) negatives and for deeper, more
  specific anchor types.
* **Ontology alignment loss** (`losses.compute_ontology_alignment_loss`):
  1. For each anchor in the batch, compute its cosine distances 1 − cos(z_i, z_j) to the other
     B − 1 anchors, and the matching LCA distances Δ̂(c_i, c_j).
  2. Convert both rows to differentiable soft ranks (`torchsort.soft_rank`, L2 regularization 0.1).
  3. Correlate the two rank vectors (Pearson on ranks, i.e. Spearman).
  4. L_onto = 1 − mean Spearman over the batch.

  This loss shapes the *global* geometry of the embedding space to follow the ontology, while the
  triplet term preserves *local* neighborhoods.

## 6. Training

```bash
bash pretraining/run_pretrain.sh /path/to/scLLM_onto_dataset 6
# = deepspeed --num_gpus=6 --master_port 29600 pretraining/train_scOntoFM.py --dataset_path ... (+ paths below)
```

| Setting | Value (released model) |
|---|---|
| Hardware | 1 node, 6 × NVIDIA H200 |
| Micro-batch / GPU | 64 triplets |
| Gradient accumulation | 4 (global batch: 64 × 6 × 4 = 1,536 triplets) |
| Epochs | 3 over the 27.5M triplets (17,920 optimizer steps per epoch) |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-3, β = (0.9, 0.999), ε = 1e-8 |
| LR schedule | DeepSpeed `WarmupDecayLR`, 1% linear warm-up, linear decay |
| Precision / parallelism | fp16 (initial loss scale 2¹²), ZeRO stage 2, gradient clipping 1.0 |
| Seed | 42 |
| Checkpoints | `results/checkpoints_<experiment_name>/checkpoint-<micro-step>` (every 1/4 epoch), `results/final_model_<experiment_name>/` |
| Logging | Weights & Biases (`--wandb_project`, disable with `--no_wandb`), per-rank logs in `--log_dir` |

Every hyperparameter is a CLI flag of `train_scOntoFM.py`; the defaults reproduce the released
model. Notes for exact reproduction:

* **LR schedule length.** The schedule length (`total_num_steps`) is computed from the number of
  **rows** in the corpus, including the excluded cells (29,160,112 × 3 / (64 · 6 · 4) = 56,953 steps
  for the released run), not from the number of triplets (17,920 steps per epoch). The learning rate therefore does not fully decay to zero by the
  end of training.
* **Two depth definitions.** The ontology depth used in the margin is the *shortest* path from
  the root. Picking the deepest LCA (step 2) uses the *longest* path.
* **Distance normalization.** The LCA matrix is divided by its NaN-aware maximum. All cell types
  in the corpus are connected to the root, so no NaN reaches the losses.
* **torchsort.** `torchsort` compiles a CUDA extension. Install it against the same torch/CUDA
  version used for training; for H200 GPUs, set `TORCH_CUDA_ARCH_LIST="9.0"` at build time.

The final checkpoint is a standard HuggingFace model directory (`config.json`, `model.safetensors`,
`training_info.pkl`). It loads directly with `scOntoFM.model.load_model` or
`BertForMaskedLM.from_pretrained`.
