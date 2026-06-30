#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="train.log"

echo "Starting training in foreground..."
echo "Log file: $LOG_FILE"
echo "Stop training: Ctrl+C"

python -u main.py 2>&1 | tee "$LOG_FILE"
