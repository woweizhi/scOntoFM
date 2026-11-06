# 2. Cell-type annotation

An MLP probe on frozen cell embeddings, trained and tested on different donors.

## Scripts

| Script | Purpose |
| --- | --- |
| `celltype_annotation_finetune.py` | Donor-split MLP classifier. Reports accuracy and macro-/weighted-F1, and scores each prediction against the Cell Ontology (exact / parent / child / sibling / no match). |

## Workflow

1. Compute cell embeddings with `scripts/run_inference.py`.
2. Split the donors (`--num_test_donors`, `--test_donors` or `--leave_one_donor_out`), train the
   MLP with `--runs_per_split` seeds (default 5), and evaluate on the held-out donors.

## Example (from the repository root)

```bash
python scripts/run_inference.py --input_h5ad data/pantissue/filtered/HIHA.h5ad --run_id scOntoFM

python downstream/2_celltype_annotation/celltype_annotation_finetune.py \
    --h5ad_path data/pantissue/filtered/HIHA.h5ad \
    --embedding_root ./output --model_name scOntoFM \
    --donor_col donor_id --label_col cell_type --num_test_donors 2 \
    --output_dir downstream/2_celltype_annotation/output/HIHA_scOntoFM
```

Main outputs: `run_summary.csv` (one row per split and seed) and
`split_seed_mean_std_summary.csv` (mean and std per split).
