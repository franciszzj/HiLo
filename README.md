# HiLo: Exploiting High Low Frequency Relations for Unbiased Panoptic Scene Graph Generation

ICCV 2023, official code implementation

## Preparation

Please follow [OpenPSG](https://github.com/Jingkang50/OpenPSG#get-started).

Pretrained models are directly converted from [Mask2Former](https://github.com/open-mmlab/mmdetection/tree/main/configs/mask2former) using [this code](./tools/change_model.py).
```.bash
python tools/change_model.py path/to/pretrained/model
```

## Train
Train HiLo baseline:
```.bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
EVAL_PAN_RELS=False \
python -m torch.distributed.launch \
  --nproc_per_node=$GPUS \
  --master_port=$PORT \
  tools/train.py \
  path/to/hilo_baseline/config \
  --auto-resume \
  --no-validate \
  --seed 666 \
  --launcher pytorch
```

Obtaining a new training file through IETrans:
```.bash
PYTHONPATH='.':$PYTHONPATH \
python tools/data_prepare/ietrans.py \
  path/to/config \
  path/to/checkpoint \
  path/to/output
```

Train HiLo:
```.bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
EVAL_PAN_RELS=False \
python -m torch.distributed.launch \
  --nproc_per_node=$GPUS \
  --master_port=$PORT \
  tools/train.py \
  path/to/hilo/config \
  --auto-resume \
  --no-validate \
  --seed 666 \
  --launcher pytorch
```

## Test
```.bash
PYTHONPATH='.':$PYTHONPATH \
python tools/test.py \
  path/to/config \
  path/to/checkpoint \
  --eval sgdet_PQ \
  --cfg-options model.bbox_head.test_forward_output_type='merge'
```


## Acknowledgements
HiLo is developed based on [OpenPSG](https://github.com/Jingkang50/OpenPSG) and [MMDetection](https://github.com/open-mmlab/mmdetection). Thanks for their great works!


## Reference
If you find this repository useful, please cite:

```
@article{zhou2023hilo,
  title={HiLo: Exploiting High Low Frequency Relations for Unbiased Panoptic Scene Graph Generation},
  author={Zhou, Zijian and Shi, Miaojing and Caesar, Holger},
  journal={arXiv preprint arXiv:2303.15994},
  year={2023}
}
```