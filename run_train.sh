#!/usr/bin/env bash
set -e

LOG_FILE="train.log"

echo "Starting training..."
nohup python -u main.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "Training started. PID: $PID"
echo "Log file: $LOG_FILE"
echo "View log: tail -f $LOG_FILE"
