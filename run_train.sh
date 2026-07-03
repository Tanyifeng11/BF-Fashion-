#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="train.log"
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

echo "Starting training in foreground..."
echo "Log file: $LOG_FILE"
echo "Stop training: Ctrl+C"

python -u main.py 2>&1 | tee "$LOG_FILE"

mkdir -p "$METRICS_DIR"
: > "$EVAL_LOG"

echo "Training finished. Starting evaluation..." | tee -a "$EVAL_LOG"
echo "Eval output dir: $EVAL_DIR" | tee -a "$EVAL_LOG"

shopt -s nullglob
CKPTS=(logs/*/checkpoints/last.ckpt)
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "Error: no checkpoint found under logs/*/checkpoints/last.ckpt" | tee -a "$EVAL_LOG"
  exit 1
fi
CKPT_PATH=$(ls -t "${CKPTS[@]}" | head -n 1)
echo "Using checkpoint: $CKPT_PATH" | tee -a "$EVAL_LOG"

run_class_jobs "sample" python -u m_class_batch_sample.py --ckpt "$CKPT_PATH" --classes

run_class_jobs "fid_lpips" python -u m_class_fid_lpips.py --classes

run_class_jobs "histo" python -u m_class_histo_loss.py --classes

: > "$EVAL_DIR/m_class_fid_lpips.txt"
: > "$EVAL_DIR/m_class_histo_loss.txt"
for classname in "${CLASSES[@]}"; do
  cat "$METRICS_DIR/fid_lpips_${classname}.txt" >> "$EVAL_DIR/m_class_fid_lpips.txt"
  cat "$METRICS_DIR/histo_${classname}.txt" >> "$EVAL_DIR/m_class_histo_loss.txt"
done

echo "Evaluation finished." | tee -a "$EVAL_LOG"
echo "FID/LPIPS: $EVAL_DIR/m_class_fid_lpips.txt" | tee -a "$EVAL_LOG"
echo "Histogram: $EVAL_DIR/m_class_histo_loss.txt" | tee -a "$EVAL_LOG"
