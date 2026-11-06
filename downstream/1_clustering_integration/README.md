# 1. Clustering, batch integration and ontology consistency

Zero-shot evaluation of frozen scOntoFM cell embeddings on nine CELLxGENE tissues
(HIHA, Heart, Lung, Liver, Kidney, Pancreas, Breast_TNBC, Adipose_tissue, Bone_marrow).

## Scripts

| Script | Purpose |
| --- | --- |
| `process_adata.py` | Keeps primary cells and moves raw counts to `adata.X`. |
| `clustering_eval_rapids.py` | scIB metrics on GPU: NMI, ARI, ASW_label, graph connectivity, batch ASW, `overall = 0.6 bio + 0.4 batch`. |
| `visOnto.py` | scOntoSim: correlation between cell-type centroid distances and Cell Ontology LCA (or onClass RWR) distances. |
| `run_rapids_multi.sh` / `run_visOnto.sh` | Run the two evaluations over all tissues and models. |

## Workflow

1. Prepare each tissue `.h5ad` with `process_adata.py`.
2. Compute cell embeddings with `scripts/run_inference.py`.
3. Run the scIB metrics and scOntoSim.

## Example

```bash
cd downstream/1_clustering_integration
python process_adata.py --adata-base-path ../../data/pantissue \
    --output-base-path ../../data/pantissue/filtered --dataset-ids Heart Lung

(cd ../.. && python scripts/run_inference.py \
    --input_h5ad data/pantissue/filtered/Heart.h5ad --run_id scOntoFM --output_dir output)

bash run_rapids_multi.sh scOntoFM donor_id cell_type   # -> output/scOntoFM/<tissue>_scib_metrics_results.csv
bash run_visOnto.sh lca scOntoFM                       # -> output/scOntoGraphSim_SOTA/
```

The shell scripts read `EMB_DIR` (default `../../output`), `ADATA_DIR`, `OUT_DIR` and `DATA_IDS`
from the environment. `clustering_eval_rapids.py` needs `rapids_singlecell` and a GPU.
