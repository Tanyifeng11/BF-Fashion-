#!/bin/bash

#SBATCH -J BF500Total
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/BF-Fashion-/bf_500_total_metrics.log
#SBATCH -e /share/home/u2515283058/BF-Fashion-/bf_500_total_metrics.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate ldm

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/BF-Fashion-}"
MYMODEL_ROOT="${MYMODEL_ROOT:-/share/home/u2515283058/Mymodel}"
DATA_ROOT="${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}"
DATASET_JSON="${DATASET_JSON:-${MYMODEL_ROOT}/data/train_bf_texture.json}"
SPLIT_PATH="${SPLIT_PATH:-${MYMODEL_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/latent-diffusion/mydata_ldm-vq-f8_test.yaml}"
CKPT="${CKPT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/eval_output/benchmark_500_total_metrics}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-openai/clip-vit-large-patch14}"

NUM_SAMPLES="${NUM_SAMPLES:-500}"
EVAL_SEED="${EVAL_SEED:-42}"
GENERATION_SEED="${GENERATION_SEED:-42}"
DDIM_STEPS="${DDIM_STEPS:-200}"
DDIM_ETA="${DDIM_ETA:-0.0}"
DEVICE="${DEVICE:-cuda:0}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-16}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-16}"
METRICS_ONLY="${METRICS_ONLY:-0}"
OVERWRITE="${OVERWRITE:-0}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HTTP_PROXY="${HTTP_PROXY:-http://211.67.63.75:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://211.67.63.75:3128}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -z "${CKPT}" ]]; then
  shopt -s nullglob
  ckpts=(logs/*/checkpoints/last.ckpt)
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "Error: no checkpoint found under logs/*/checkpoints/last.ckpt" >&2
    exit 1
  fi
  CKPT="$(ls -t "${ckpts[@]}" | head -n 1)"
fi

mkdir -p "${OUTPUT_DIR}"

cmd=(
  python -u bf_500_total_metrics.py
  --config "${CONFIG}"
  --ckpt "${CKPT}"
  --dataset-json "${DATASET_JSON}"
  --data-root "${DATA_ROOT}"
  --split-path "${SPLIT_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --num-samples "${NUM_SAMPLES}"
  --seed "${EVAL_SEED}"
  --generation-seed "${GENERATION_SEED}"
  --ddim-steps "${DDIM_STEPS}"
  --ddim-eta "${DDIM_ETA}"
  --device "${DEVICE}"
  --fid-batch-size "${FID_BATCH_SIZE}"
  --clip-batch-size "${CLIP_BATCH_SIZE}"
  --clip-model-path "${CLIP_MODEL_PATH}"
)

if [[ "${METRICS_ONLY}" == "1" ]]; then
  cmd+=(--metrics-only)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "============================================"
echo "BF-Fashion 500-image total metric evaluation"
echo "CKPT=${CKPT}"
echo "DATASET_JSON=${DATASET_JSON}"
echo "SPLIT_PATH=${SPLIT_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "EVAL_SEED=${EVAL_SEED}"
echo "GENERATION_SEED=${GENERATION_SEED}"
echo "DDIM_STEPS=${DDIM_STEPS}"
echo "DEVICE=${DEVICE}"
echo "CLIP_MODEL_PATH=${CLIP_MODEL_PATH}"
echo "METRICS_ONLY=${METRICS_ONLY}"
echo "OVERWRITE=${OVERWRITE}"
echo "============================================"

"${cmd[@]}" 2>&1 | tee "${OUTPUT_DIR}/evaluation.log"

echo "============================================"
echo "Evaluation completed."
echo "Summary: ${OUTPUT_DIR}/total_metrics.txt"
echo "CSV:     ${OUTPUT_DIR}/metrics_summary.csv"
echo "============================================"
