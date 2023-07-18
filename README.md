# HiLo: Exploiting High Low Frequency Relations for Unbiased Panoptic Scene Graph Generation

ICCV 2023, official code implementation

Abstract:

Panoptic Scene Graph generation (PSG) is a recently proposed task in image scene understanding that aims to segment the image and extract triplets of subjects, objects and their relations to build a scene graph.
This task is particularly challenging for two reasons. 
First, it suffers from a long-tail problem in its relation categories, making naive biased methods more inclined to high-frequency relations.
Existing unbiased methods tackle the long-tail problem by data/loss rebalancing to favor low-frequency relations.
Second, a subject-object pair can have two or more semantically overlapping relations.
While existing methods favor one over the other, our proposed HiLo framework lets different network branches specialize on low and high frequency relations, enforce their consistency and fuse the results.
To the best of our knowledge we are the first to propose an explicitly unbiased PSG method.
In extensive experiments we show that our HiLo framework achieves state-of-the-art results on the PSG task. We also apply our method to the Scene Graph Generation task that predicts boxes instead of masks and see improvements over all baseline methods.

## Preparation

Please follow [OpenPSG](https://github.com/Jingkang50/OpenPSG#get-started).

Pretrained models are directly converted from [Mask2Former](https://github.com/open-mmlab/mmdetection/tree/main/configs/mask2former) using [this code](./tools/change_model.py).
```.bash
python tools/change_model.py path/to/pretrained/model
```


## Configs
Config path: ./configs/psgmask2former/
- **R50**: psgmask2former_r50_hilo_baseline.py, psgmask2former_r50_hilo.py
- **Swin Base**: psgmask2former_swin_b_hilo_baseline.py, psgmask2former_swin_b_hilo.py
- **Swin Large**: psgmask2former_swin_l_hilo_baseline.py, psgmask2former_swin_l_hilo.py


## Training
Train HiLo baseline:
```.bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
EVAL_PAN_RELS=True \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
  tools/train.py path/to/hilo_baseline/config --auto-resume --no-validate --seed 666 --launcher pytorch
```

Obtaining a new training file through IETrans:
```.bash
PYTHONPATH='.':$PYTHONPATH \
python tools/data_prepare/ietrans.py path/to/hilo_baseline/config path/to/checkpoint path/to/output
```

Train HiLo:
```.bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
EVAL_PAN_RELS=True \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
  tools/train.py path/to/hilo/config --auto-resume --no-validate --seed 666 --launcher pytorch
```

## Testing and Evaluation

Test and eval HiLo baseline:
```.bash
PYTHONPATH='.':$PYTHONPATH \
EVAL_PAN_RELS=True \
python tools/test.py path/to/hilo_baseline/config path/to/checkpoint --eval sgdet_PQ
```

Test and eval HiLo:
```.bash
PYTHONPATH='.':$PYTHONPATH \
EVAL_PAN_RELS=True \
python tools/test.py path/to/hilo/config path/to/checkpoint --eval sgdet_PQ --cfg-options model.bbox_head.test_forward_output_type='merge'
```


## Acknowledgements
HiLo is developed based on [OpenPSG](https://github.com/Jingkang50/OpenPSG) and [MMDetection](https://github.com/open-mmlab/mmdetection). Thanks for their great works!


## Citation
If you find this repository useful, please cite:

```
@article{zhou2023hilo,
  title={HiLo: Exploiting High Low Frequency Relations for Unbiased Panoptic Scene Graph Generation},
  author={Zhou, Zijian and Shi, Miaojing and Caesar, Holger},
  journal={arXiv preprint arXiv:2303.15994},
  year={2023}
}
```