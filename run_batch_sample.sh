#!/usr/bin/env bash
set -e

mkdir -p eval_output
LOG_FILE="eval_output/sample.log"

echo "Starting batch sampling..."
nohup python -u m_class_batch_sample.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "Batch sampling started. PID: $PID"
echo "Log file: $LOG_FILE"
echo "View log: tail -f $LOG_FILE"
