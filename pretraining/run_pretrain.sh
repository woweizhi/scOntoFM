#!/usr/bin/env bash
# Single-node scOntoFM pretraining with DeepSpeed.
# The released model was trained on 6 x H200 GPUs for 3 epochs (~2 days).
#
# Usage: bash run_pretrain.sh /path/to/scLLM_onto_dataset [NUM_GPUS]
# On SLURM, wrap this script with sbatch, request NUM_GPUS GPUs on one node,
# ~8 CPUs per GPU, ~350 GB RAM and a wall time of about two days.

DATASET_PATH=${1:?"usage: run_pretrain.sh <tokenized_dataset_dir> [num_gpus]"}
NUM_GPUS=${2:-6}
cd "$(dirname "$0")"

deepspeed --num_gpus="${NUM_GPUS}" --master_port 29600 train_scOntoFM.py \
    --dataset_path "${DATASET_PATH}" \
    --triplet_path ../assets/triplet_ind_hard_top50.pkl \
    --lca_path ../assets/celltype_similarity_LCA_deepest_allnodes.pkl \
    --token_dict_path ../resources/vocab/token_dictionary.pkl \
    --obo_path ../resources/ontology/cl-basic.obo \
    --experiment_name centroid_spearman_dynamicMargin_top50 \
    --output_dir ./results
