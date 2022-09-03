#!/bin/bash -l

module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/train.py \
  $CONFIG 
