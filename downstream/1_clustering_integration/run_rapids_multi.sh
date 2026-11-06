#!/usr/bin/env bash
# scIB clustering / integration metrics (clustering_eval_rapids.py) over the pan-tissue datasets.
#
# Usage:  bash run_rapids_multi.sh [MODELS] [BATCH_KEY] [LABEL_KEY]
#   MODELS     space-separated embedding run ids (default: scOntoFM)
#   BATCH_KEY  obs column for batches            (default: donor_id)
#   LABEL_KEY  obs column for cell types         (default: cell_type)
#
# The directory variables below can be overridden from the environment, e.g.
#   EMB_DIR=/path/to/output bash run_rapids_multi.sh "scOntoFM geneformer"
# Relative paths are resolved from this folder.
set -eo pipefail
cd "$(dirname "$0")"

read -r -a MODELS <<< "${1:-scOntoFM}"
BATCH_KEY=${2:-"donor_id"}
LABEL_KEY=${3:-"cell_type"}

# <EMB_DIR>/<model>/<data_id>/<data_id>_embeddings.npy  (output of scripts/run_inference.py --run_id <model>)
EMB_DIR=${EMB_DIR:-"../../output"}
# <ADATA_DIR>/<data_id>.h5ad  (process_adata.py output, row-aligned with the embeddings)
ADATA_DIR=${ADATA_DIR:-"../../data/pantissue/filtered"}
OUT_DIR=${OUT_DIR:-"./output"}
read -r -a DATA_IDS <<< "${DATA_IDS:-HIHA Heart Lung Liver Kidney Pancreas Breast_TNBC Adipose_tissue Bone_marrow}"

for data_id in "${DATA_IDS[@]}"; do
    for model in "${MODELS[@]}"; do
        # results go to <OUT_DIR>/<RUN_ID>/<data_id>_scib_metrics_results.csv; one folder per model
        RUN_ID="$model"
        echo "Running evaluation on data: $data_id | model: $model | run_id: $RUN_ID"

        python clustering_eval_rapids.py --base_dir "$EMB_DIR" \
            --adata_dir "$ADATA_DIR" \
            --model "$model" --data_id "$data_id" \
            --batch_key "$BATCH_KEY" --label_key "$LABEL_KEY" \
            --exclude_label Other \
            --output "$OUT_DIR" \
            --run_id "$RUN_ID"

        echo "Finished: $data_id"
        echo "------------------------------------"
    done
done
