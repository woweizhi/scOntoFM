#!/usr/bin/env bash
# scOntoSim (visOnto.py) over the pan-tissue datasets.
#
# Usage:  bash run_visOnto.sh [MODE] [MODELS]
#   MODE    lca  (default) reference = Cell Ontology LCA distances shipped in
#                ../../assets/celltype_similarity_LCA_deepest_allnodes.pkl   -> scOntoGraphSim
#           rwr  reference = onClass random-walk-with-restart distances
#                (optional external asset, set RWR_REF; default ./onClass_rwr_dist.pkl) -> scGraph-RWR
#   MODELS  space-separated embedding run ids (default: scOntoFM)
#
# The directory variables below can be overridden from the environment, e.g.
#   EMB_DIR=/path/to/output RUN_ID=myrun bash run_visOnto.sh lca "scOntoFM geneformer"
# Relative paths are resolved from this folder.
set -eo pipefail
cd "$(dirname "$0")"

MODE=${1:-"lca"}
read -r -a MODELS <<< "${2:-scOntoFM}"

# <EMB_DIR>/<model>/<data_id>/<data_id>_embeddings.npy  (output of scripts/run_inference.py --run_id <model>)
EMB_DIR=${EMB_DIR:-"../../output"}
# <ADATA_DIR>/<data_id>.h5ad  (process_adata.py output, row-aligned with the embeddings)
ADATA_DIR=${ADATA_DIR:-"../../data/pantissue/filtered"}
OUT_DIR=${OUT_DIR:-"./output"}
TRIM_PROP=${TRIM_PROP:-0.2}
read -r -a DATA_IDS <<< "${DATA_IDS:-HIHA Heart Lung Liver Kidney Pancreas Breast_TNBC Adipose_tissue Bone_marrow}"

case "$MODE" in
    lca)
        REFERENCE=${LCA_REF:-"../../assets/celltype_similarity_LCA_deepest_allnodes.pkl"}
        RUN_ID=${RUN_ID:-"scOntoGraphSim_SOTA"}
        ;;
    rwr)
        REFERENCE=${RWR_REF:-"./onClass_rwr_dist.pkl"}
        RUN_ID=${RUN_ID:-"scGraphRWR_SOTA"}
        ;;
    *)
        echo "Unknown mode '$MODE' (expected: lca | rwr)" >&2
        exit 1
        ;;
esac

if [ ! -f "$REFERENCE" ]; then
    echo "Reference graph not found: $REFERENCE" >&2
    exit 1
fi

for data_id in "${DATA_IDS[@]}"; do
    echo "Running evaluation on data: $data_id | mode: $MODE | models: ${MODELS[*]}"

    # results go to <OUT_DIR>/<RUN_ID>/<data_id>_graph_similarity{,_per_celltype}.csv
    python visOnto.py \
        --base_dir "$EMB_DIR" \
        --adata_dir "$ADATA_DIR" \
        --model "${MODELS[@]}" \
        --data_id "$data_id" \
        --reference_path "$REFERENCE" \
        --trim_prop "$TRIM_PROP" \
        --output "$OUT_DIR" \
        --run_id "$RUN_ID"

    echo "Finished: $data_id"
    echo "------------------------------------"
done
