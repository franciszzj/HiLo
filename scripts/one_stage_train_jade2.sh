#!/bin/bash -l
#SBATCH --output=/jmain02/home/J2AD019/exk01/%u/logs/%j.out
#SBATCH --job-name=openpsg
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=8
#SBATCH --time=6-00:00

source ~/.bashrc
module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version

CONFIG=$1
GPUS=8
PORT=29500

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
EVAL_PAN_RELS=False \
WANDB_MODE="offline" \
python -m torch.distributed.launch \
  --nproc_per_node=$GPUS \
  --master_port=$PORT \
  tools/train.py \
  $CONFIG \
  --auto-resume \
  --launcher pytorch
