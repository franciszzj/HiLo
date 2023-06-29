import os.path as osp
import copy
import random
from collections import defaultdict

import mmcv
import numpy as np
import torch
import colorsys
from detectron2.data.detection_utils import read_image
from mmdet.datasets import DATASETS, CocoPanopticDataset
from mmdet.datasets.coco_panoptic import COCOPanoptic
from mmdet.datasets.pipelines import Compose
from panopticapi.utils import rgb2id, id2rgb

from openpsg.evaluation import sgg_evaluation
from openpsg.models.relation_heads.approaches import Result

# Add for visualization
import cv2
from skimage.segmentation import find_boundaries
from polylabel import polylabel
file_client = mmcv.FileClient(**dict(backend='disk'))
OBJ_PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
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


def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))


REL_PALETTE = [hsv2rgb(h/20.0, 0.9, 0.9) for h in range(20)]


@DATASETS.register_module()
class PanopticSceneGraphDataset(CocoPanopticDataset):
    def __init__(
            self,
            ann_file,
            pipeline,
            classes=None,
            data_root=None,
            img_prefix='',
            seg_prefix=None,
            proposal_file=None,
            test_mode=False,
            filter_empty_gt=True,
            file_client_args=dict(backend='disk'),
            # New args
            split: str = 'train',  # {"train", "test"}
            all_bboxes: bool = False,  # load all bboxes (thing, stuff) for SG
            # {'random', 'low2high', 'high2low'}
            overlap_rel_choice_type: str = 'random',
            add_no_rel_triplet: bool = False,
    ):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.overlap_rel_choice_type = overlap_rel_choice_type
        self.add_no_rel_triplet = add_no_rel_triplet

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)

        self.proposal_file = None
        self.proposals = None

        self.all_bboxes = all_bboxes
        self.split = split

        # Load dataset
        dataset = mmcv.load(ann_file)

        # Relation to frequency dict
        self.rel2freq = dict()
        for i in range(len(dataset['predicate_classes'])):
            self.rel2freq[i+1] = 0
        self.so_rel2freq = dict()
        for k in range(len(dataset['thing_classes']) + len(dataset['stuff_classes'])):
            for j in range(len(dataset['thing_classes']) + len(dataset['stuff_classes'])):
                for i in range(len(dataset['predicate_classes'])):
                    self.so_rel2freq[(k, j, i+1)] = 0

        for d in dataset['data']:
            # NOTE: 0-index for object class labels
            # for s in d['segments_info']:
            #     s['category_id'] += 1

            # for a in d['annotations']:
            #     a['category_id'] += 1

            # NOTE: 1-index for predicate class labels
            for r in d['relations']:
                r[2] += 1
                self.rel2freq[r[2]] += 1
                self.so_rel2freq[(d['annotations'][r[0]]['category_id'],
                                  d['annotations'][r[1]]['category_id'], r[2])] += 1

        # NOTE: Filter out images with zero relations
        dataset['data'] = [
            d for d in dataset['data'] if len(d['relations']) != 0
        ]

        # Get split
        assert split in {'train', 'test'}
        if split == 'train':
            self.data = [
                d for d in dataset['data']
                if d['image_id'] not in dataset['test_image_ids']
            ]
            # self.data = self.data[:1000] # for quick debug
        elif split == 'test':
            self.data = [
                d for d in dataset['data']
                if d['image_id'] in dataset['test_image_ids']
            ]
            # If you need to traverse all data
            # self.data = [
            #     d for d in dataset['data']
            # ]
            # self.data = self.data[:1000] # for quick debug

            '''
            # Only test data with relational semantic overlap problems.
            new_data = []
            only_test_dup = True
            for d in self.data:
                has_dup = False
                r_list = []
                if only_test_dup:
                    keep_list = []
                for r in d['relations']:
                    if (r[0], r[1]) not in r_list:
                        r_list.append((r[0], r[1]))
                    else:
                        if only_test_dup:
                            keep_list.append(r)
                        has_dup = True
                if only_test_dup:
                    d['relations'] = keep_list
                if has_dup:
                    new_data.append(d)
            self.data = new_data
            # '''

        # Init image infos
        self.data_infos = []
        for d in self.data:
            self.data_infos.append({
                'filename': d['file_name'],
                'height': d['height'],
                'width': d['width'],
                'id': d['image_id'],
            })
        self.img_ids = [d['id'] for d in self.data_infos]

        # Define classes, 0-index
        # NOTE: Class ids should range from 0 to (num_classes - 1)
        self.THING_CLASSES = dataset['thing_classes']
        self.STUFF_CLASSES = dataset['stuff_classes']
        self.CLASSES = self.THING_CLASSES + self.STUFF_CLASSES
        self.PREDICATES = dataset['predicate_classes']

        # NOTE: For evaluation
        self.coco = self._init_cocoapi()
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.categories = self.coco.cats

        # processing pipeline
        self.pipeline = Compose(pipeline)

        if not self.test_mode:
            self._set_group_flag()

    def _init_cocoapi(self):
        auxcoco = COCOPanoptic()

        annotations = []

        # Create mmdet coco panoptic data format
        for d in self.data:

            annotation = {
                'file_name': d['pan_seg_file_name'],
                'image_id': d['image_id'],
            }
            segments_info = []

            for a, s in zip(d['annotations'], d['segments_info']):

                segments_info.append({
                    'id': s['id'],
                    'category_id': s['category_id'],
                    'iscrowd': s['iscrowd'],
                    'area': int(s['area']),
                    # Convert from xyxy to xywh
                    'bbox': [
                        a['bbox'][0],
                        a['bbox'][1],
                        a['bbox'][2] - a['bbox'][0],
                        a['bbox'][3] - a['bbox'][1],
                    ],
                })

            annotation['segments_info'] = segments_info

            annotations.append(annotation)

        thing_categories = [{
            'id': i,
            'name': name,
            'isthing': 1
        } for i, name in enumerate(self.THING_CLASSES)]
        stuff_categories = [{
            'id': i + len(self.THING_CLASSES),
            'name': name,
            'isthing': 0
        } for i, name in enumerate(self.STUFF_CLASSES)]

        # Create `dataset` attr for for `createIndex` method
        auxcoco.dataset = {
            'images': self.data_infos,
            'annotations': annotations,
            'categories': thing_categories + stuff_categories,
        }
        auxcoco.createIndex()
        auxcoco.img_ann_map = auxcoco.imgToAnns
        auxcoco.cat_img_map = auxcoco.catToImgs

        return auxcoco

    def get_ann_info(self, idx):
        d = self.data[idx]

        # Process bbox annotations
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if self.all_bboxes:
            # NOTE: Get all the bbox annotations (thing + stuff)
            gt_bboxes = np.array([a['bbox'] for a in d['annotations']],
                                 dtype=np.float32)
            gt_labels = np.array([a['category_id'] for a in d['annotations']],
                                 dtype=np.int64)

        else:
            gt_bboxes = []
            gt_labels = []

            # FIXME: Do we have to filter out `is_crowd`?
            # Do not train on `is_crowd`,
            # i.e just follow the mmdet dataset classes
            # Or treat them as stuff classes?
            # Can try and train on datasets with iscrowd
            # and without and see the difference

            for a, s in zip(d['annotations'], d['segments_info']):
                # NOTE: Only thing bboxes are loaded
                if s['isthing']:
                    gt_bboxes.append(a['bbox'])
                    gt_labels.append(a['category_id'])

            if gt_bboxes:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
            else:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.array([], dtype=np.int64)

        # Process segment annotations
        gt_mask_infos = []
        for s in d['segments_info']:
            gt_mask_infos.append({
                'id': s['id'],
                'category': s['category_id'],
                'is_thing': s['isthing']
            })

        # Process relationship annotations
        gt_rels = d['relations'].copy()

        # Filter out dupes!
        if self.split == 'train':
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            random_gt_rels = []
            high2low_gt_rels = []
            low2high_gt_rels = []
            gt_rels = []
            for k, v in all_rel_sets.items():
                random_gt_rels.append((k[0], k[1], np.random.choice(v)))
                v_idx = np.argmin([self.rel2freq[i] for i in v])
                high2low_gt_rels.append((k[0], k[1], v[v_idx]))
                v_idx = np.argmax([self.rel2freq[i] for i in v])
                low2high_gt_rels.append((k[0], k[1], v[v_idx]))
            if self.overlap_rel_choice_type == 'random':
                gt_rels = random_gt_rels
            elif self.overlap_rel_choice_type == 'high2low':
                gt_rels = high2low_gt_rels
            elif self.overlap_rel_choice_type == 'low2high':
                gt_rels = low2high_gt_rels
            random_gt_rels = np.array(random_gt_rels, dtype=np.int32)
            high2low_gt_rels = np.array(high2low_gt_rels, dtype=np.int32)
            low2high_gt_rels = np.array(low2high_gt_rels, dtype=np.int32)
            gt_rels = np.array(gt_rels, dtype=np.int32)
        else:
            # for test or val set, filter the duplicate triplets,
            # but allow multiple labels for each pair
            all_rel_sets = []
            for (o0, o1, r) in gt_rels:
                if (o0, o1, r) not in all_rel_sets:
                    all_rel_sets.append((o0, o1, r))
            gt_rels = np.array(all_rel_sets, dtype=np.int32)

        # add relation to target
        num_box = len(gt_mask_infos)
        relation_map = np.zeros((num_box, num_box), dtype=np.int64)
        for i in range(gt_rels.shape[0]):
            # If already exists a relation?
            if relation_map[int(gt_rels[i, 0]), int(gt_rels[i, 1])] > 0:
                if random.random() > 0.5:
                    relation_map[int(gt_rels[i, 0]),
                                 int(gt_rels[i, 1])] = int(gt_rels[i, 2])
            else:
                relation_map[int(gt_rels[i, 0]),
                             int(gt_rels[i, 1])] = int(gt_rels[i, 2])

        # add no relation triplet
        if self.add_no_rel_triplet:
            sub_obj_pair_list = [[i, j] for i, j, r in gt_rels]
            gt_no_rels = []
            for i in range(len(gt_labels)):  # subject
                for j in range(len(gt_labels)):  # object
                    if i == j:
                        continue
                    if [i, j] not in sub_obj_pair_list:
                        gt_no_rels.append([i, j, 0])
            gt_no_rels = np.array(gt_no_rels)
            gt_rels_cat = copy.deepcopy(gt_rels)
            if gt_rels.shape[0] + gt_no_rels.shape[0] > 100:
                gt_no_rels_ids = [i for i in range(gt_no_rels.shape[0])]
                gt_no_rels_select_ids = random.sample(
                    gt_no_rels_ids, k=100-gt_rels.shape[0])
                gt_no_rels_select = gt_no_rels[gt_no_rels_select_ids]
                if len(gt_no_rels_select.shape) == 2:
                    if gt_no_rels_select.shape[0] != 0 and gt_no_rels_select.shape[1] == 3:
                        gt_rels_cat = np.concatenate(
                            (gt_rels, gt_no_rels_select), axis=0)
            else:
                if len(gt_no_rels.shape) == 2:
                    if gt_no_rels.shape[0] != 0 and gt_no_rels.shape[1] == 3:
                        gt_rels_cat = np.concatenate(
                            (gt_rels, gt_no_rels), axis=0)
            gt_rels = gt_rels_cat

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            rels=gt_rels,
            rel_maps=relation_map,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_mask_infos,
            seg_map=d['pan_seg_file_name'],
            segment_infos=d['segments_info'],
        )
        if self.split == 'train':
            ann['random_gt_rels'] = random_gt_rels
            ann['high2low_gt_rels'] = high2low_gt_rels
            ann['low2high_gt_rels'] = low2high_gt_rels
        else:
            ann['random_gt_rels'] = gt_rels
            ann['high2low_gt_rels'] = gt_rels
            ann['low2high_gt_rels'] = gt_rels

        return ann

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        super().pre_pipeline(results)

        results['rel_fields'] = []

    def prepare_test_img(self, idx):
        # For SGG, since the forward process may need gt_bboxes/gt_labels,
        # we should also load annotation as if in the training mode.
        return self.prepare_train_img(idx)

    def evaluate(
        self,
        results,
        metric='predcls',
        logger=None,
        jsonfile_prefix=None,
        classwise=True,
        multiple_preds=False,
        iou_thrs=0.5,
        nogc_thres_num=None,
        detection_method='bbox',
        **kwargs,
    ):
        """Overwritten evaluate API:

        For each metric in metrics, it checks whether to invoke ps or sg
        evaluation. if the metric is not 'sg', the evaluate method of super
        class is invoked to perform Panoptic Segmentation evaluation. else,
        perform scene graph evaluation.
        """
        metrics = metric if isinstance(metric, list) else [metric]

        # Available metrics
        allowed_sg_metrics = ['predcls', 'sgcls', 'sgdet']
        allowed_od_metrics = ['PQ']

        sg_metrics, od_metrics = [], []
        for m in metrics:
            if m in allowed_od_metrics:
                od_metrics.append(m)
            elif m in allowed_sg_metrics:
                sg_metrics.append(m)
            else:
                raise ValueError('Unknown metric {}.'.format(m))

        if len(od_metrics) > 0:
            # invoke object detection evaluation.
            # Temporarily for bbox
            if not isinstance(results[0], Result):
                # it may be the results from the son classes
                od_results = results
            else:
                od_results = [{'pan_results': r.pan_results} for r in results]
            return super().evaluate(
                od_results,
                metric,
                logger,
                jsonfile_prefix,
                classwise=classwise,
                **kwargs,
            )

        if len(sg_metrics) > 0:
            """Invoke scene graph evaluation.

            prepare the groundtruth and predictions. Transform the predictions
            of key-wise to image-wise. Both the value in gt_results and
            det_results are numpy array.
            """
            if not hasattr(self, 'test_gt_results'):
                print('\nLoading testing groundtruth...\n')
                prog_bar = mmcv.ProgressBar(len(self))
                gt_results = []
                for i in range(len(self)):
                    ann = self.get_ann_info(i)

                    # NOTE: Change to object class labels 1-index here
                    ann['labels'] += 1

                    # load gt pan_seg masks
                    segment_info = ann['masks']
                    gt_img = read_image(self.img_prefix + '/' + ann['seg_map'],
                                        format='RGB')
                    gt_img = gt_img.copy()  # (H, W, 3)

                    seg_map = rgb2id(gt_img)

                    # get separate masks
                    gt_masks = []
                    labels_coco = []
                    for _, s in enumerate(segment_info):
                        label = self.CLASSES[s['category']]
                        labels_coco.append(label)
                        gt_masks.append(seg_map == s['id'])
                    # load gt pan seg masks done

                    gt_results.append(
                        Result(
                            bboxes=ann['bboxes'],
                            labels=ann['labels'],
                            rels=ann['rels'],
                            relmaps=ann['rel_maps'],
                            rel_pair_idxes=ann['rels'][:, :2],
                            rel_labels=ann['rels'][:, -1],
                            masks=gt_masks,
                        ))
                    '''
                    # for visualization
                    self.viz_single(
                        self.data_infos[i], gt_results[i], results[i])
                    # '''
                    prog_bar.update()

                print('\n')
                self.test_gt_results = gt_results

            return sgg_evaluation(
                sg_metrics,
                groundtruths=self.test_gt_results,
                predictions=results,
                iou_thrs=iou_thrs,
                logger=logger,
                ind_to_predicates=['__background__'] + self.PREDICATES,
                multiple_preds=multiple_preds,
                # predicate_freq=self.predicate_freq,
                nogc_thres_num=nogc_thres_num,
                detection_method=detection_method,
            )

    def get_statistics(self):
        freq_matrix = self.get_freq_matrix()
        eps = 1e-3
        freq_matrix += eps
        pred_dist = np.log(freq_matrix / freq_matrix.sum(2)[:, :, None] + eps)

        result = {
            'freq_matrix': torch.from_numpy(freq_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
        }
        if result['pred_dist'].isnan().any():
            print('check pred_dist: nan')
        return result

    def get_freq_matrix(self):
        num_obj_classes = len(self.CLASSES)
        num_rel_classes = len(self.PREDICATES)

        freq_matrix = np.zeros(
            (num_obj_classes, num_obj_classes, num_rel_classes + 1),
            dtype=np.float)
        progbar = mmcv.ProgressBar(len(self.data))

        for d in self.data:
            segments = d['segments_info']
            relations = d['relations']

            for rel in relations:
                object_index = segments[rel[0]]['category_id']
                subject_index = segments[rel[1]]['category_id']
                relation_index = rel[2]

                freq_matrix[object_index, subject_index, relation_index] += 1

            progbar.update()

        return freq_matrix

    def viz_single(self, data_info, groundtruth, prediction):
        # 1. parse data
        image_file = data_info['filename']
        image = mmcv.imread(osp.join(self.img_prefix, image_file))
        h, w, c = image.shape

        gt_bboxes = groundtruth.bboxes
        gt_labels = groundtruth.labels
        gt_masks = groundtruth.masks
        gt_rels = groundtruth.rels
        gt_pan_bytes = file_client.get(
            osp.join(self.img_prefix, data_info['segm_file']))
        gt_pan_seg = mmcv.imfrombytes(gt_pan_bytes,
                                      flag='color',
                                      channel_order='rgb').squeeze()
        gt_pan_id = rgb2id(gt_pan_seg)
        gt_pan_boundaries = find_boundaries(gt_pan_id)

        pred_bboxes = prediction.refine_bboxes
        pred_labels = prediction.labels
        pred_masks = prediction.masks
        pred_rels = prediction.rels
        pred_rel_dists = prediction.rel_dists
        pred_pan_id = prediction.pan_results
        pred_pan_boundaries = find_boundaries(pred_pan_id)
        pred_pan_seg = id2rgb(pred_pan_id)
        
        # 1.5 prepare ious
        gt_masks_array = torch.from_numpy(
            np.stack(gt_masks, axis=0)).flatten(1)
        pred_masks_array = torch.from_numpy(
            np.stack(pred_masks, axis=0)).flatten(1)
        all_masks_array = torch.cat(
            [gt_masks_array, pred_masks_array], dim=0).to(torch.float32)
        ious = all_masks_array.mm(all_masks_array.transpose(0, 1)) / \
            ((all_masks_array+all_masks_array) > 0).sum(-1)
        ious = ious[:gt_masks_array.shape[0], gt_masks_array.shape[0]:]

        # 2. visualize gt
        gt_vis_image = np.zeros((2*h, w, c), dtype=image.dtype) + 225
        gt_pan_seg = copy.deepcopy(image)
        for idx, (label, mask) in enumerate(zip(gt_labels, gt_masks)):
            index = np.where(mask)
            gt_pan_seg[index] = OBJ_PALETTE[label - 1]
        gt_pan_seg[gt_pan_boundaries] = [0, 0, 0]
        gt_vis_image[:h, :, :] = image * 0.5 + gt_pan_seg * 0.5
        for idx, (label, mask) in enumerate(zip(gt_labels, gt_masks)):
            if mask.sum() == 0:
                continue
            # this_boundaries = find_boundaries(mask)
            # poly_index = np.where(this_boundaries)
            # poly_index = [[[x, y] for x, y in zip(poly_index[0], poly_index[1])]]
            # center = polylabel(poly_index)
            index = np.where(mask)
            center = [np.mean(index[1]), np.mean(index[0])]
            cv2.putText(gt_vis_image, '{}_{}'.format(idx, self.CLASSES[label - 1]), (int(
                center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for idx in range(gt_rels.shape[0]):
            rels = gt_rels[idx]
            sub_idx, obj_idx, rel_idx = rels
            sub_cls = gt_labels[sub_idx]
            obj_cls = gt_labels[obj_idx]
            sub_name = self.CLASSES[sub_cls - 1]
            obj_name = self.CLASSES[obj_cls - 1]
            rel_name = self.PREDICATES[rel_idx - 1]
            text = '{}_{} - {} - {}_{}'.format(sub_idx,
                                               sub_name, rel_name, obj_idx, obj_name)
            cv2.putText(gt_vis_image, text, (0, h+h//21*(idx+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, REL_PALETTE[idx % 20], 2)

        # 3. visualize pred
        pred_vis_image = np.zeros((2*h, w, c), dtype=image.dtype) + 225
        pred_pan_seg = copy.deepcopy(image)
        for idx, (label, mask) in enumerate(zip(pred_labels, pred_masks)):
            index = np.where(mask)
            pred_pan_seg[index] = OBJ_PALETTE[label - 1]
        pred_pan_seg[pred_pan_boundaries] = [0, 0, 0]
        pred_vis_image[:h, :, :] = image * 0.5 + pred_pan_seg * 0.5
        for idx, (label, mask) in enumerate(zip(pred_labels, pred_masks)):
            if mask.sum() == 0:
                continue
            # this_boundaries = find_boundaries(mask)
            # poly_index = np.where(this_boundaries)
            # poly_index = [[[x, y] for x, y in zip(poly_index[0], poly_index[1])]]
            # center = polylabel(poly_index)
            index = np.where(mask)
            center = [np.mean(index[1]), np.mean(index[0])]
            cv2.putText(pred_vis_image, '{}_{}'.format(idx, self.CLASSES[label - 1]), (int(
                center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for idx in range(pred_rels.shape[0]):
            if idx > 20:
                break
            rels = pred_rels[idx]
            sub_idx, obj_idx, rel_idx = rels
            sub_cls = pred_labels[sub_idx]
            obj_cls = pred_labels[obj_idx]
            # construct rel_color, rel_thickness
            for gt_idx in range(gt_rels.shape[0]):
                gt_sub_idx, gt_obj_idx, gt_rel_idx = gt_rels[gt_idx]
                gt_sub_cls = gt_labels[gt_sub_idx]
                gt_obj_cls = gt_labels[gt_obj_idx]
                if sub_cls == gt_sub_cls and obj_cls == gt_obj_cls and ious[gt_sub_idx, sub_idx] > 0.5 and ious[gt_obj_idx, obj_idx] > 0.5:
                    rel_color = REL_PALETTE[gt_idx % 20]
                    if gt_rel_idx == rel_idx:
                        rel_thickness = 2
                    else:
                        rel_thickness = 1
                    break
                else:
                    rel_color = (0, 0, 0)
                    rel_thickness = 1
            sub_name = self.CLASSES[sub_cls - 1]
            obj_name = self.CLASSES[obj_cls - 1]
            rel_name = self.PREDICATES[rel_idx - 1]
            text = '{}_{} - {} - {}_{}'.format(
                sub_idx, sub_name, rel_name, obj_idx, obj_name)
            cv2.putText(pred_vis_image, text, (0, h+h//21*(idx+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, rel_color, rel_thickness)

        # 4. save
        vis_image = np.concatenate((gt_vis_image, pred_vis_image), axis=1)
        cv2.imwrite(
            './viz/{}'.format(data_info['filename'].split('/')[-1]), vis_image)
