#!/usr/bin/env bash
set -e

PROJECT_DIR="/share/home/u2515283058/BF-Fashion-"
ENV_NAME="ldm"

cd "$PROJECT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Please load Anaconda/Miniconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env $ENV_NAME already exists. Skip creation."
else
  conda env create -f environment-server.yaml
fi

conda activate "$ENV_NAME"

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install -e .
pip install -e src/taming-transformers

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"

echo "Environment setup finished."
