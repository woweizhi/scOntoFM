# 4. Novel cell-type identification

Zero-shot labeling of cell types never seen in pretraining. A centroid is computed for each
seen cell type; a centroid for each unseen type is inferred as the `exp(-LCA distance)`-weighted
average of the seen centroids; each query cell takes the label of the nearest unseen centroid.

## Scripts and files

| File | Purpose |
| --- | --- |
| `build_seen_centroids.py` | Averages per-cell-type embeddings into `seen_celltype_centroids.pkl`. |
| `onto_unseenCell_classifier.py` | Library: centroid inference and nearest-centroid classification. |
| `novel_cell_prediction.py` | Classifies unseen cells; writes predictions and per-type accuracy. |

## Input data

One `.h5ad` per cell type, downloaded from the CELLxGENE Census (release 2025-01-30): primary
cells only, `min_genes=200`, `min_cells=5`, at most 10,000 cells per type, Ensembl IDs as
`var_names`, raw counts.

## Workflow

1. Embed every per-cell-type file with `scripts/run_inference.py`.
2. Build the seen centroids.
3. Classify the unseen cells against all candidates and Report macro accuracy over the 99 cell types.

