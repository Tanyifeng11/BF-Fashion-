#!/usr/bin/env bash
set -e

LOG_FILE="sample.log"

echo "Starting batch sampling..."
nohup python -u m_class_batch_sample.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "Batch sampling started. PID: $PID"
echo "Log file: $LOG_FILE"
echo "View log: tail -f $LOG_FILE"
