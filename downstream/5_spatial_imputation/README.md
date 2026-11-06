# 5. Spatial gene imputation

Imputes genes missing from a spatial (ST) panel using a single-cell (SC) reference. Each spot
takes the softmax-weighted average expression of its top-k most similar reference cells
(cosine similarity of embeddings). Evaluated with 10-fold gene-wise cross-validation
(Pearson, Spearman, Kendall tau).

## Scripts

| Script | Purpose |
| --- | --- |
| `spatial_imputation_pipeline.py` | Builds the 10 gene folds and tokenizes the ST data on each fold's train genes. |
| `adata_split.py` | Writes per-fold ST `.h5ad` files (only needed for baseline models). |
| `run_inference_imputation.py` | scOntoFM embeddings of the tokenized ST folds. |
| `run_imputation.py` | Imputes the held-out genes and computes per-gene metrics. |
| `run_imputation_eval.sh` | Runs `run_imputation.py` over all dataset pairs and models. |

## Workflow

1. Build gene folds and tokenize the ST data.
2. Embed the ST folds (`run_inference_imputation.py`) and the SC reference (`scripts/run_inference.py`)
   into the same `--output_dir`.
3. Impute and evaluate.

