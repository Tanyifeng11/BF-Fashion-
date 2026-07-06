#!/bin/bash

#SBATCH -J BF500Metric
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/BF-Fashion-/bf_500_metrics_cluster.log
#SBATCH -e /share/home/u2515283058/BF-Fashion-/bf_500_metrics_cluster.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh

conda activate ldm

export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/BF-Fashion-}"
DATASET_TEST_DIR="${DATASET_TEST_DIR:-/share/home/u2515283058/datasets/BF/test}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${PROJECT_ROOT}/eval_output}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_OUTPUT}/benchmark_500_metrics}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
PER_CLASS_SAMPLES="${PER_CLASS_SAMPLES:-100}"
EVAL_SEED="${EVAL_SEED:-42}"
REBUILD_SPLIT="${REBUILD_SPLIT:-0}"
DEVICE="${DEVICE:-cuda:0}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "BF 500-image metric evaluation"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATASET_TEST_DIR=${DATASET_TEST_DIR}"
echo "EVAL_OUTPUT=${EVAL_OUTPUT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "PER_CLASS_SAMPLES=${PER_CLASS_SAMPLES}"
echo "EVAL_SEED=${EVAL_SEED}"
echo "REBUILD_SPLIT=${REBUILD_SPLIT}"
echo "DEVICE=${DEVICE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "============================================"

python -u bf_500_metrics.py \
  --dataset-test-dir "${DATASET_TEST_DIR}" \
  --eval-output "${EVAL_OUTPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --per-class-samples "${PER_CLASS_SAMPLES}" \
  --seed "${EVAL_SEED}" \
  --rebuild-split "${REBUILD_SPLIT}" \
  --device "${DEVICE}" \
  2>&1 | tee "${OUTPUT_DIR}/bf_500_metrics.log"

echo "============================================"
echo "BF 500-image metric evaluation done."
echo "Summary: ${OUTPUT_DIR}/summary_metrics.csv"
echo "Split:   ${OUTPUT_DIR}/bf_500_split.json"
echo "Log:     ${OUTPUT_DIR}/bf_500_metrics.log"
echo "============================================"
