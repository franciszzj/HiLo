#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/logs/%j.out
#SBATCH --job-name=openpsg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=65536
#SBATCH --time=2-00:00

source ~/.bashrc
module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version

CONFIG=$1
MODEL=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/test.py \
  $CONFIG \
  $MODEL \
  --eval sgdet
