#!/bin/bash

#SBATCH -J BFTrain
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 7
#SBATCH --gres=gpu:4
#SBATCH -o /share/home/u2515283058/BF-Fashion-/train_cluster.log
#SBATCH -e /share/home/u2515283058/BF-Fashion-/train_cluster.err

set -e

source /share/apps/anaconda3/etc/profile.d/conda.sh

conda activate ldm

export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

MASTER_PORT=29563 bash run_train.sh
