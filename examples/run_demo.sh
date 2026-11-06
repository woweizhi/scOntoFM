#!/usr/bin/env bash
# scOntoFM demo through the command-line tools (run from anywhere).
#   1) preprocess  2) tokenize  3) embed (+ UMAP)
cd "$(dirname "$0")/.."

DEMO=examples/data/demo_pancreas.h5ad
OUT=examples/demo_output

# 1. Preprocessing: raw counts -> filtered, library-size normalized AnnData
python scripts/preprocess.py --input "$DEMO" --output_dir "$OUT/preprocessed"

# 2. Tokenization: rank-value encoding -> HuggingFace dataset (input_ids)
python scripts/tokenize_h5ad.py --input "$OUT/preprocessed/demo_pancreas.h5ad" \
    --output_dir "$OUT/tokenized" --format hf

# 3. Inference: steps 1-2 are run internally (preprocessing is cached) -> cell embeddings
python scripts/run_inference.py --input_h5ad "$DEMO" \
    --preprocess_cache "$OUT/preprocessed" \
    --output_dir "$OUT" --run_id scOntoFM \
    --color_obs cell_type donor_id --save_data

echo "Cell embeddings: $OUT/scOntoFM/demo_pancreas/demo_pancreas_embeddings.npy"
