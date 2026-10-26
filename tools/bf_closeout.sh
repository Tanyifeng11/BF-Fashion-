#!/bin/bash
# 同一冻结输入与原生推理；只补已有图清单中的缺项。
set -euo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate ldm
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
repo=/share/home/u2515283058/BF-Fashion-external-20261026
source_repo=/share/home/u2515283058/BF-Fashion-
out=/share/home/u2515283058/Mymodel/output_eval/m12_external_closeout_20261026
checkpoint=$source_repo/logs/2026-07-01T22-38-33_mydata_ldm-vq-f8/checkpoints/last.ckpt
vq=$source_repo/models/first_stage_models/vq-f8/model.ckpt
cd "$repo"
case "$1" in
  audit)
    python -u tools/bf_confirm_native.py prepare --out "$out" --confirm-archive /share/home/u2515283058/Mymodel/output_eval/m12_gain_independent_confirm_20261020/m12_gain_confirm_review.tar.gz --archive-sha256 1bcd6c0ef4e4ad2291c560b9258a7162b0ea50a769d36a6c60544b19ba52e3bb
    python -u tools/bf_confirm_native.py audit --repo "$repo" --source-repo "$source_repo" --ckpt "$checkpoint" --vq "$vq" --out "$out" --confirm-manifest "$out/BF_confirm_manifest.json" --train-manifest /share/home/u2515283058/Mymodel/data/train_bf_texture.json --training-config "$source_repo/logs/2026-07-01T22-38-33_mydata_ldm-vq-f8/configs/2026-07-01T22-38-33-project.yaml"
    ;;
  evaluate)
    conda activate Mymodel
    python -u tools/bf_common_eval.py
    ;;
  smoke|generate)
    extra=()
    if [ "$1" = smoke ]; then extra=(--limit 3); fi
    python -u tools/bf_confirm_native.py generate --repo "$repo" --ckpt "$checkpoint" --vq "$vq" --out "$out" --confirm-manifest "$out/BF_confirm_manifest.json" --approval "$out/BF_provenance_approval.json" "${extra[@]}"
    ;;
  *) exit 2 ;;
esac
