#!/usr/bin/env bash
# Step 4 of spatial imputation: run run_imputation.py for every (ST, SC) dataset
# pair and every model whose embeddings exist under EMB_ROOT.
#
# Usage (from this folder):
#   bash run_imputation_eval.sh
#   DATA_DIR=/path/to/data EMB_ROOT=../../output bash run_imputation_eval.sh
#
# Expected embedding layout (see README.md):
#   $EMB_ROOT/<ST_id>/<model>/fold_<k>_adata/fold_<k>_adata_embeddings.npy
#   $EMB_ROOT/<model>/<SC_id>/<SC_id>_embeddings.npy

DATA_DIR="${DATA_DIR:-../../data}"            # holds <ST_id>.h5ad and <SC_id>.h5ad
EMB_ROOT="${EMB_ROOT:-../../output}"          # --output_dir used for the inference scripts
GENE_SPLITS="${GENE_SPLITS:-./cache/imputation_arrow}"
OUT_DIR="${OUT_DIR:-./imputation_outputs}"
TOP_K="${TOP_K:-50}"

methods=("scOntoFM" "geneformer" "cellPLM" "scgpt" "sccello")

# "<ST_dataset_id> <SC_dataset_id>" pairs
dataset_pairs=(
    "SEAAD_donor_H21_032 SEAAD_ref_donor_H21_032"
    "HumanLungCancerPatient2_st_minGene5 GSE131907_Lung_sc"
    "HumanLiverCancerPatient2_st_minGene50 GSE151530_Liver_sc_minGene5"
    "dataset20_st dataset20_sc"
    "dataset7_st dataset7_sc"
)

for pair in "${dataset_pairs[@]}"; do
    read -r ST_dataset_id SC_dataset_id <<< "$pair"
    for method in "${methods[@]}"; do
        echo "Running imputation: $ST_dataset_id <- $SC_dataset_id, model: $method"

        python run_imputation.py \
            --spatial "${DATA_DIR}/${ST_dataset_id}.h5ad" \
            --reference "${DATA_DIR}/${SC_dataset_id}.h5ad" \
            --model "$method" \
            --top-k "$TOP_K" \
            --gene-splits "$GENE_SPLITS" \
            --output-dir "$OUT_DIR" \
            --sc-dataset-id "${SC_dataset_id}" \
            --spatial-dataset-id "${ST_dataset_id}" \
            --result-embedding-root "$EMB_ROOT"

        echo "Finished: $method"
        echo "------------------------------------"
    done
done
