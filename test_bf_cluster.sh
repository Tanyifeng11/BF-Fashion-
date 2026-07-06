#!/bin/bash

#SBATCH -J BFTest
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 7
#SBATCH --gres=gpu:2
#SBATCH -o /share/home/u2515283058/BF-Fashion-/test_cluster.log
#SBATCH -e /share/home/u2515283058/BF-Fashion-/test_cluster.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh

conda activate ldm

export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

cd /share/home/u2515283058/BF-Fashion-

EVAL_DIR="eval_output"
EVAL_LOG="$EVAL_DIR/eval.log"
METRICS_DIR="$EVAL_DIR/metrics"
CLASSES=(bag top outwear pants dress)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPUS=(0 1 2 3)
fi

run_class_jobs() {
  local stage="$1"
  shift
  local pids=()

  for i in "${!CLASSES[@]}"; do
    local classname="${CLASSES[$i]}"
    local gpu="${GPUS[$((i % ${#GPUS[@]}))]}"

    (
      echo "[$stage:$classname:gpu$gpu] start"
      CUDA_VISIBLE_DEVICES="$gpu" "$@" "$classname" 2>&1
      echo "[$stage:$classname:gpu$gpu] done"
    ) | sed "s/^/[$stage:$classname:gpu$gpu] /" | tee -a "$EVAL_LOG" &

    pids+=("$!")

    if [ "${#pids[@]}" -eq "${#GPUS[@]}" ]; then
      wait "${pids[@]}"
      pids=()
    fi
  done

  if [ "${#pids[@]}" -gt 0 ]; then
    wait "${pids[@]}"
  fi
}

mkdir -p "$METRICS_DIR"
: > "$EVAL_LOG"

shopt -s nullglob
CKPTS=(logs/*/checkpoints/last.ckpt)
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "Error: no checkpoint found under logs/*/checkpoints/last.ckpt" | tee -a "$EVAL_LOG"
  exit 1
fi

CKPT_PATH=$(ls -t "${CKPTS[@]}" | head -n 1)

echo "Job ID: ${SLURM_JOB_ID:-local}" | tee -a "$EVAL_LOG"
echo "Node: $(hostname)" | tee -a "$EVAL_LOG"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}" | tee -a "$EVAL_LOG"
echo "Using checkpoint: $CKPT_PATH" | tee -a "$EVAL_LOG"
echo "Eval output dir: $EVAL_DIR" | tee -a "$EVAL_LOG"

run_class_jobs "sample" python -u m_class_batch_sample.py --ckpt "$CKPT_PATH" --classes

run_class_jobs "fid_lpips" python -u m_class_fid_lpips.py --classes

run_class_jobs "histo" python -u m_class_histo_loss.py --classes

: > "$EVAL_DIR/m_class_fid_lpips.txt"
: > "$EVAL_DIR/m_class_histo_loss.txt"
for classname in "${CLASSES[@]}"; do
  fid_file="$METRICS_DIR/fid_lpips_${classname}.txt"
  histo_file="$METRICS_DIR/histo_${classname}.txt"

  if [ ! -f "$fid_file" ]; then
    echo "Error: missing $fid_file. Check fid_lpips stage in $EVAL_LOG" | tee -a "$EVAL_LOG"
    exit 1
  fi

  if [ ! -f "$histo_file" ]; then
    echo "Error: missing $histo_file. Check histo stage in $EVAL_LOG" | tee -a "$EVAL_LOG"
    exit 1
  fi

  cat "$fid_file" >> "$EVAL_DIR/m_class_fid_lpips.txt"
  cat "$histo_file" >> "$EVAL_DIR/m_class_histo_loss.txt"
done

echo "Evaluation finished." | tee -a "$EVAL_LOG"
echo "FID/LPIPS: $EVAL_DIR/m_class_fid_lpips.txt" | tee -a "$EVAL_LOG"
echo "Histogram: $EVAL_DIR/m_class_histo_loss.txt" | tee -a "$EVAL_LOG"
