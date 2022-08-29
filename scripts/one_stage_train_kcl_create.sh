#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/logs/%j.out
#SBATCH --job-name=openpsg
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=4
#SBATCH --time=2-00:00

source /users/${USER}/.bashrc
module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version

CONFIG=$1
GPUS=4
PORT=29500

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
  --nproc_per_node=$GPUS \
  --master_port=$PORT \
  tools/train.py \
  $CONFIG \
  --launcher pytorch