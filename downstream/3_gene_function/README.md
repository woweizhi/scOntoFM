# 3. Gene function

GO term prediction and GO-term embedding structure from scOntoFM gene embeddings.

## Scripts

| Script | Purpose |
| --- | --- |
| `GO_classifier.py` | MLP that predicts GO term membership from **static** gene token embeddings (20 CV splits, F1 and AP). |
| `GO_embedding.py` | GO-term embeddings from **static** gene embeddings; KMeans vs. GO namespace (BP/CC/MF), PCA/UMAP plots. |
| `GO_embedding_contextualize.py` | The same analysis with dataset-specific **contextual** gene embeddings. |
| `gene_func.ipynb` | Hallmark pathway scores per cell type from contextual gene embeddings. |

## Workflow

1. Download `c5.go.v2025.1.Hs.symbols.gmt` from
   [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp) into this folder.
2. Static embeddings: run `GO_classifier.py` and `GO_embedding.py`. By default they read
   `resources/weights/scOntoFM` and `resources/vocab/`; the defaults are the paper settings.
3. Contextual embeddings: run inference with `--save_gene_embeddings --save_data`, then run
   `GO_embedding_contextualize.py` or `gene_func.ipynb`.

## Example

```bash
cd downstream/3_gene_function
python GO_classifier.py --output_dir ./go_classifier_out --model_id scOntoFM
python GO_embedding.py  --output_dir ./go_embedding_out  --model_id scOntoFM

cd ../..
python scripts/run_inference.py --input_h5ad data/Kim2020_Lung_convert.h5ad \
    --run_id scOntoFM --save_gene_embeddings --save_data
python downstream/3_gene_function/GO_embedding_contextualize.py \
    --dataset_id Kim2020_Lung_convert --run_id scOntoFM --sota_output_root ./output \
    --output_dir downstream/3_gene_function/go_embedding_context_out
```
