# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import argparse
import os
import os.path as osp
import time
import warnings
import pprint
import copy
import cv2
import gc
import numpy as np
from skimage.segmentation import find_boundaries
from panopticapi.utils import rgb2id, id2rgb
from prettytable import PrettyTable

import mmcv
import torch
from grade import save_results
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.parallel import DataContainer as DC
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.core import encode_mask_results
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmdet.models import build_detector

from openpsg.datasets import build_dataset
from openpsg.models.relation_heads.psgmask2former_multi_decoder_head import merge_results
from grade import save_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir',
                        help='directory where painted images will be saved')
    parser.add_argument('--show-score-thr',
                        type=float,
                        default=0.3,
                        help='score threshold (default: 0.3)')
    parser.add_argument('--gpu-collect',
                        action='store_true',
                        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument(
        '--submit',
        action='store_true',
        help='save output to a json file and save the panoptic mask as a png image into a folder for grading purpose'
    )
    parser.add_argument('--vis', help='vis results and save them.')

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.submit, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=samples_per_gpu,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=distributed,
                                   shuffle=False)

    # build the model and load checkpoint
    if not (os.getenv('SAVE_PREDICT', 'false').lower() == 'true') and \
            not (os.getenv('MERGE_PREDICT', 'false').lower() == 'true'):
        cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # NOTE:
    if hasattr(dataset, 'PREDICATES'):
        model.PREDICATES = dataset.PREDICATES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        start_time = time.time()
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)

        '''
        # ##### code to test, add labels to test process.
        model.eval()
        prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
        outputs = []
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                device = data['gt_labels'][0].data[0][0].device
                gt_masks = data['gt_masks'][0].data[0][0].to_tensor(
                    torch.uint8, device)
                data['gt_masks'] = [DC([[gt_masks]])]
                result = model(return_loss=False, rescale=True, **data)
            batch_size = len(result)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (bbox_results,
                                                encode_mask_results(mask_results))
            outputs.extend(result)
            for _ in range(batch_size):
                prog_bar.update()
        # ##### code to test, add labels to test process.
        # '''

        '''
        # ##### Use pre saved predict results to after post-processing merge
        predict1 = '/jmain02/home/J2AD019/exk01/zxz35-exk01/workspace/OpenPSG/temp/v15_3_high2low.pkl'
        predict2 = '/jmain02/home/J2AD019/exk01/zxz35-exk01/workspace/OpenPSG/temp/v15_3_low2high.pkl'
        print('load predict1: {}'.format(predict1))
        outputs1 = mmcv.load(predict1)
        print('load predict2: {}'.format(predict2))
        outputs2 = mmcv.load(predict2)
        print('load finish')
        prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
        outputs = []
        for idx, (output1, output2) in enumerate(zip(outputs1, outputs2)):
            if idx >= len(data_loader.dataset):
                break
            # not use gt to find upper bound
            data = None
            # use gt to find upper bound
            # data = data_loader.dataset[idx]
            outputs.append(merge_results(output1, output2, data))
            gc.collect()
            prog_bar.update()
        print('after post-processing merge finish.')
        # ##### Use pre saved predict results to after post-processing merge
        # '''

        duration = time.time() - start_time
        mean_duration = duration / len(data_loader)
        print('\n inference time:', flush=True)
        print(mean_duration, flush=True)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    if args.submit:
        save_results(outputs)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        if args.vis:
            print(f'\nsave vis results to {args.vis}')
            img_file_list = [data['img_metas'][0].data[0][0]['filename']
                             for data in data_loader]
            vis_outputs(outputs, img_file_list, save_dir=args.vis)
            # vis_outputs2(dataset, outputs, img_file_list, save_dir=args.vis)
            # relational_semantic_overlap(
            #     dataset, outputs, img_file_list, save_dir=args.vis)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule', 'dynamic_intervals']:
                eval_kwargs.pop(key, None)

            if args.eval[0] == 'sgdet_PQ':
                eval_kwargs['metric'] = 'sgdet'
                metric_sgdet = dataset.evaluate(outputs, **eval_kwargs)
                eval_kwargs['metric'] = 'PQ'
                metric_pq = dataset.evaluate(outputs, **eval_kwargs)
                final_score = metric_sgdet['sgdet_recall_R_20'] * 0.3 + \
                    metric_sgdet['sgdet_mean_recall_mR_20'] * 0.6 + \
                    metric_pq['PQ'] * 0.01 * 0.1
                metric_results = dict()
                metric_results['sgdet_recall_R_20'] = metric_sgdet['sgdet_recall_R_20']
                metric_results['sgdet_recall_R_50'] = metric_sgdet['sgdet_recall_R_50']
                metric_results['sgdet_recall_R_100'] = metric_sgdet['sgdet_recall_R_100']
                metric_results['sgdet_mean_recall_mR_20'] = metric_sgdet['sgdet_mean_recall_mR_20']
                metric_results['sgdet_mean_recall_mR_50'] = metric_sgdet['sgdet_mean_recall_mR_50']
                metric_results['sgdet_mean_recall_mR_100'] = metric_sgdet['sgdet_mean_recall_mR_100']
                metric_results['PQ'] = metric_pq['PQ']
                metric_results['final_score'] = final_score
            else:
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                metric = dataset.evaluate(outputs, **eval_kwargs)
                metric_results = metric

            pprint.pprint(metric_results)
            epoch = args.checkpoint.split('/')[-1].split('.')[0].split('epoch_')[1]  # noqa
            result_str = 'epoch={}, PQ={:.2f}\nR/mR@20={:.2f}/{:.2f}\nR/mR@50={:.2f}/{:.2f}\nR/mR@100={:.2f}/{:.2f}'.format(
                epoch, metric_results.get('PQ', 0),
                metric_results['sgdet_recall_R_20'] * 100,
                metric_results['sgdet_mean_recall_mR_20'] * 100,
                metric_results['sgdet_recall_R_50'] * 100,
                metric_results['sgdet_mean_recall_mR_50'] * 100,
                metric_results['sgdet_recall_R_100'] * 100,
                metric_results['sgdet_mean_recall_mR_100'] * 100,
            )
            print(args)
            print(result_str)


object_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged'
]
predicate_classes = [
    'over',
    'in front of',
    'beside',
    'on',
    'in',
    'attached to',
    'hanging from',
    'on back of',
    'falling off',
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over',
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning on',
]


def vis_outputs(outputs, img_file_list, save_dir='./vis_results', merge_vis=True, save_top_k=20):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for output, img_file in zip(outputs, img_file_list):
        bboxes = output.refine_bboxes
        labels = output.labels
        rel_pairs = output.rel_pair_idxes
        rel_dists = output.rel_dists
        rel_labels = output.rel_labels
        rel_scores = output.rel_scores
        masks = output.masks
        pan_seg = output.pan_results
        rel_num = rel_labels.shape[0]
        m_h, m_w = masks.shape[-2:]

        img = cv2.imread(img_file)
        if save_top_k == 20:
            row_num = 5
            col_num = 4
        elif save_top_k == 50:
            row_num = 10
            col_num = 5
        elif save_top_k == 100:
            row_num = 10
            col_num = 10
        if merge_vis:
            merge_vis_img = np.zeros((m_h * row_num, m_w * col_num, 3))
        for idx, rel_label in enumerate(rel_labels):
            if idx >= save_top_k:
                continue
            vis_img = copy.deepcopy(img)
            # data prepare
            relation = predicate_classes[rel_label - 1]
            sub_idx = rel_pairs[idx, 0]
            obj_idx = rel_pairs[idx, 1]
            sub_label = labels[sub_idx]
            obj_label = labels[obj_idx]
            subject = object_classes[sub_label - 1]
            object = object_classes[obj_label - 1]
            sub_bbox = bboxes[sub_idx].astype(np.int32)
            obj_bbox = bboxes[obj_idx].astype(np.int32)
            sub_mask = masks[sub_idx].astype(np.int32)
            obj_mask = masks[obj_idx].astype(np.int32)
            i1, i2 = np.where(sub_mask == 1)
            sub_mask_template = np.zeros((m_h, m_w, vis_img.shape[-1]))
            sub_mask_template[i1, i2] = sub_mask_template[i1, i2] + \
                np.array([255, 0, 0])
            i1, i2 = np.where(obj_mask == 1)
            obj_mask_template = np.zeros((m_h, m_w, vis_img.shape[-1]))
            obj_mask_template[i1, i2] = obj_mask_template[i1, i2] + \
                np.array([0, 0, 255])
            # mask
            vis_img = vis_img + sub_mask_template * 0.5
            vis_img = vis_img + obj_mask_template * 0.5
            # subject
            cv2.rectangle(vis_img, (sub_bbox[0], sub_bbox[1]), (sub_bbox[2], sub_bbox[3]),
                          color=(255, 0, 0), thickness=3)
            cv2.putText(vis_img, subject, (sub_bbox[0], sub_bbox[3]),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 128, 0), thickness=2)
            # object
            cv2.rectangle(vis_img, (obj_bbox[0], obj_bbox[1]), (obj_bbox[2], obj_bbox[3]),
                          color=(0, 0, 255), thickness=3)
            cv2.putText(vis_img, object, (obj_bbox[0], obj_bbox[3]),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 128, 255), thickness=2)
            # relation
            cv2.putText(vis_img, relation, ((sub_bbox[0]+obj_bbox[0])//2, (sub_bbox[3]+obj_bbox[3])//2),
                        cv2.FONT_HERSHEY_PLAIN, 2, (128, 255, 128), thickness=2)

            if merge_vis:
                y_idx = idx // col_num
                x_idx = idx % col_num
                merge_vis_img[(m_h * y_idx):(m_h * (y_idx + 1)),
                              (m_w * x_idx):(m_w * (x_idx + 1))] = vis_img
            else:
                img_name = img_file.split('/')[-1]
                save_name = img_name.replace('.jpg', '_{}.jpg'.format(idx))
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, vis_img)

        if merge_vis:
            img_name = img_file.split('/')[-1]
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, merge_vis_img)


'''
PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
           (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
           (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
           (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
           (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
           (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
           (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
           (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
           (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
           (134, 134, 103), (145, 148, 174), (255, 208, 186),
           (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
           (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
           (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
           (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
           (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
           (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
           (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
           (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
           (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
           (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
           (191, 162, 208), (255, 255, 128), (147, 211, 203),
           (150, 100, 100), (168, 171, 172), (146, 112, 198),
           (210, 170, 100), (92, 136, 89), (218, 88, 184), (241, 129, 0),
           (217, 17, 255), (124, 74, 181), (70, 70, 70), (255, 228, 255),
           (154, 208, 0), (193, 0, 92), (76, 91, 113), (255, 180, 195),
           (106, 154, 176),
           (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
           (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
           (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
           (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
           (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
           (146, 139, 141),
           (70, 130, 180), (134, 199, 156), (209, 226, 140), (96, 36, 108),
           (96, 96, 96), (64, 170, 64), (152, 251, 152), (208, 229, 228),
           (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143),
           (102, 102, 156), (250, 141, 255)]


file_client = mmcv.FileClient(**dict(backend='disk'))


def vis_outputs2(dataset, outputs, img_file_list, save_dir='./vis_results', merge_vis=True, save_top_k=20):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fo = open('{}/output.txt'.format(save_dir), 'w')
    for data, output, img_file in zip(dataset, outputs, img_file_list):
        # if '000000496854' not in img_file:
        #     continue

        print(img_file)
        fo.write(img_file+'\n')
        img = cv2.imread(img_file)

        # ground truth
        pan_file = img_file.replace(
            'val2017', 'panoptic_val2017').replace('.jpg', '.png')

        pan_bytes = file_client.get(pan_file)
        pan = mmcv.imfrombytes(pan_bytes,
                               flag='color',
                               channel_order='rgb').squeeze()
        pan_id = rgb2id(pan)
        pan_boundaries = find_boundaries(pan_id)

        new_seg = copy.deepcopy(img)
        segments_info = data['img_metas'][0].data['ann_info']['segments_info']
        for idx, object in enumerate(segments_info):
            object_id = object['id']
            object_label = object['category_id']
            index = np.where(pan_id == object_id)
            new_seg[index] = PALETTE[object_label]

        vis_img_gt = img * 0.5 + new_seg * 0.5
        vis_img_gt[pan_boundaries] = [0, 0, 0]

        img_name = img_file.split('/')[-1].replace('.jpg', '_gt.jpg')
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, vis_img_gt)

        print('ground truth')
        fo.write('ground truth\n')
        table = PrettyTable(['subject', 'object', 'relation'])
        for triplet in data['img_metas'][0].data['ann_info']['rels']:
            sub_idx = triplet[0]
            sub_label = data['img_metas'][0].data['ann_info']['labels'][sub_idx]
            obj_idx = triplet[1]
            obj_label = data['img_metas'][0].data['ann_info']['labels'][obj_idx]
            rel_label = triplet[2]
            subject_name = object_classes[sub_label]
            object_name = object_classes[obj_label]
            relation_name = predicate_classes[rel_label - 1]
            table.add_row(['{}_{}'.format(sub_idx, subject_name),
                           '{}_{}'.format(obj_idx, object_name),
                           relation_name])
        print(table)
        fo.write(table.get_string() + '\n')

        # prediction
        bboxes = output.refine_bboxes
        labels = output.labels
        rel_pairs = output.rel_pair_idxes
        rel_dists = output.rel_dists
        rel_labels = output.rel_labels
        rel_scores = output.rel_scores
        masks = output.masks
        pan_seg = output.pan_results
        rel_num = rel_labels.shape[0]
        m_h, m_w = masks.shape[-2:]

        if len(pan_seg.shape) == 3:
            pan_seg = rgb2id(pan_seg)
        boundaries = find_boundaries(pan_seg)
        new_seg = copy.deepcopy(img)
        for idx, (label, mask) in enumerate(zip(labels, masks)):
            index = np.where(mask > 0)
            new_seg[index] = PALETTE[label - 1]

        vis_img = img * 0.5 + new_seg * 0.5
        vis_img[boundaries] = [0, 0, 0]

        img_name = img_file.split('/')[-1].replace('.jpg', '_pred.jpg')
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, vis_img)

        print('prediction')
        fo.write('prediction\n')
        table = PrettyTable(['subject', 'object', 'relation'])
        for idx, (rel_pair, rel_label) in enumerate(zip(rel_pairs, rel_labels)):
            if idx > save_top_k:
                break
            sub_idx = rel_pair[0]
            sub_label = labels[sub_idx] - 1
            obj_idx = rel_pair[1]
            obj_label = labels[obj_idx] - 1
            subject_name = object_classes[sub_label]
            object_name = object_classes[obj_label]
            relation_name = predicate_classes[rel_label - 1]
            table.add_row(['{}_{}'.format(sub_idx, subject_name),
                           '{}_{}'.format(obj_idx, object_name),
                           relation_name])
        print(table)
        fo.write(table.get_string() + '\n')

    fo.close()


def relational_semantic_overlap(dataset, outputs, img_file_list, save_dir='./vis_results', merge_vis=True, save_top_k=20):
    import pdb; pdb.set_trace()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fo = open('{}/output.txt'.format(save_dir), 'w')
    for data, output, img_file in zip(dataset, outputs, img_file_list):
        print(img_file)
        fo.write(img_file+'\n')
        img = cv2.imread(img_file)

        # ground truth
        pan_file = img_file.replace(
            'val2017', 'panoptic_val2017').replace('.jpg', '.png')

        pan_bytes = file_client.get(pan_file)
        pan = mmcv.imfrombytes(pan_bytes,
                               flag='color',
                               channel_order='rgb').squeeze()
        pan_id = rgb2id(pan)
        pan_boundaries = find_boundaries(pan_id)

        new_seg = copy.deepcopy(img)
        segments_info = data['img_metas'][0].data['ann_info']['segments_info']
        for idx, object in enumerate(segments_info):
            object_id = object['id']
            object_label = object['category_id']
            index = np.where(pan_id == object_id)
            new_seg[index] = PALETTE[object_label]

        vis_img_gt = img * 0.5 + new_seg * 0.5
        vis_img_gt[pan_boundaries] = [0, 0, 0]

        img_name = img_file.split('/')[-1].replace('.jpg', '_gt.jpg')
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, vis_img_gt)

        print('ground truth')
        fo.write('ground truth\n')
        table = PrettyTable(['subject', 'object', 'relation'])
        for triplet in data['img_metas'][0].data['ann_info']['rels']:
            sub_idx = triplet[0]
            sub_label = data['img_metas'][0].data['ann_info']['labels'][sub_idx]
            obj_idx = triplet[1]
            obj_label = data['img_metas'][0].data['ann_info']['labels'][obj_idx]
            rel_label = triplet[2]
            subject_name = object_classes[sub_label]
            object_name = object_classes[obj_label]
            relation_name = predicate_classes[rel_label - 1]
            table.add_row(['{}_{}'.format(sub_idx, subject_name),
                           '{}_{}'.format(obj_idx, object_name),
                           relation_name])
        print(table)
        fo.write(table.get_string() + '\n')

        # prediction
        bboxes = output.refine_bboxes
        labels = output.labels
        rel_pairs = output.rel_pair_idxes
        rel_dists = output.rel_dists
        rel_labels = output.rel_labels
        rel_scores = output.rel_scores
        masks = output.masks
        pan_seg = output.pan_results
        rel_num = rel_labels.shape[0]
        m_h, m_w = masks.shape[-2:]

        if len(pan_seg.shape) == 3:
            pan_seg = rgb2id(pan_seg)
        boundaries = find_boundaries(pan_seg)
        new_seg = copy.deepcopy(img)
        for idx, (label, mask) in enumerate(zip(labels, masks)):
            index = np.where(mask > 0)
            new_seg[index] = PALETTE[label - 1]

        vis_img = img * 0.5 + new_seg * 0.5
        vis_img[boundaries] = [0, 0, 0]

        img_name = img_file.split('/')[-1].replace('.jpg', '_pred.jpg')
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, vis_img)

        print('prediction')
        fo.write('prediction\n')
        table = PrettyTable(['subject', 'object', 'relation'])
        for idx, (rel_pair, rel_label) in enumerate(zip(rel_pairs, rel_labels)):
            if idx > save_top_k:
                break
            sub_idx = rel_pair[0]
            sub_label = labels[sub_idx] - 1
            obj_idx = rel_pair[1]
            obj_label = labels[obj_idx] - 1
            subject_name = object_classes[sub_label]
            object_name = object_classes[obj_label]
            relation_name = predicate_classes[rel_label - 1]
            table.add_row(['{}_{}'.format(sub_idx, subject_name),
                           '{}_{}'.format(obj_idx, object_name),
                           relation_name])
        print(table)
        fo.write(table.get_string() + '\n')

    fo.close()
# '''

if __name__ == '__main__':
    main()
