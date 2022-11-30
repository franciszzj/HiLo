import os
import sys
import cv2
import json
import copy
import numpy as np
from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class SaveIntermediateResults(object):
    def __init__(self, save_json='', save_dir='', save_type='bbox', vis=False, merge_vis=True):
        self.save_json = save_json
        self.save_dir = save_dir
        self.save_type = save_type
        self.vis = vis
        self.merge_vis = merge_vis
        self.fo = open(self.save_json, 'w')

        self.object_classes = [
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
        self.predicate_classes = [
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

    def __call__(self, results):

        filename = results['ori_filename']
        gt_rels = results['gt_rels']
        gt_labels = results['gt_labels']
        gt_bboxes = results['gt_bboxes']
        gt_masks = results['gt_masks']
        img = results['img']

        if self.vis and self.merge_vis:
            h, w = img.shape[:2]
            row_num = 5
            col_num = 4
            merge_vis_img = np.zeros((h * row_num, w * col_num, 3))

        for i in range(gt_rels.shape[0]):
            sub_id = gt_rels[i, 0]
            obj_id = gt_rels[i, 1]
            rel_class_id = gt_rels[i, 2] - 1
            relation = self.predicate_classes[rel_class_id]

            sub_labels = gt_labels[sub_id]
            obj_labels = gt_labels[obj_id]
            sub_bboxes = gt_bboxes[sub_id]
            obj_bboxes = gt_bboxes[obj_id]
            sub_masks = gt_masks.masks[sub_id]
            obj_masks = gt_masks.masks[obj_id]
            subject = self.object_classes[sub_labels]
            object = self.object_classes[obj_labels]
            masks = np.logical_or(sub_masks, obj_masks)

            x1 = int(min(sub_bboxes[0], obj_bboxes[0]))
            y1 = int(min(sub_bboxes[1], obj_bboxes[1]))
            x2 = int(max(sub_bboxes[2], obj_bboxes[2]))
            y2 = int(max(sub_bboxes[3], obj_bboxes[3]))

            if self.vis:
                vis_img = copy.deepcopy(img)
                # data prepare
                sub_bboxes = sub_bboxes.astype(np.int32)
                obj_bboxes = obj_bboxes.astype(np.int32)
                m_h, m_w = sub_masks.shape[-2:]
                sub_masks = sub_masks.astype(np.int32)
                obj_masks = obj_masks.astype(np.int32)
                i1, i2 = np.where(sub_masks == 1)
                sub_masks_template = np.zeros((m_h, m_w, vis_img.shape[-1]))
                sub_masks_template[i1, i2] = sub_masks_template[i1, i2] + \
                    np.array([255, 0, 0])
                i1, i2 = np.where(obj_masks == 1)
                obj_masks_template = np.zeros((m_h, m_w, vis_img.shape[-1]))
                obj_masks_template[i1, i2] = obj_masks_template[i1, i2] + \
                    np.array([0, 0, 255])
                # mask
                vis_img = vis_img + sub_masks_template * 0.5
                vis_img = vis_img + obj_masks_template * 0.5
                # subject
                cv2.rectangle(vis_img, (sub_bboxes[0], sub_bboxes[1]), (sub_bboxes[2], sub_bboxes[3]),
                              color=(255, 0, 0), thickness=3)
                cv2.putText(vis_img, subject, (sub_bboxes[0], sub_bboxes[3]),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 128, 0), thickness=2)
                # object
                cv2.rectangle(vis_img, (obj_bboxes[0], obj_bboxes[1]), (obj_bboxes[2], obj_bboxes[3]),
                              color=(0, 0, 255), thickness=3)
                cv2.putText(vis_img, object, (obj_bboxes[0], obj_bboxes[3]),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 128, 255), thickness=2)
                # relation
                cv2.putText(vis_img, relation, ((sub_bboxes[0]+obj_bboxes[0])//2, (sub_bboxes[3]+obj_bboxes[3])//2),
                            cv2.FONT_HERSHEY_PLAIN, 2, (128, 255, 128), thickness=2)

                if self.merge_vis:
                    if i < 20:
                        y_idx = i // col_num
                        x_idx = i % col_num
                        merge_vis_img[(h * y_idx):(h * (y_idx + 1)),
                                      (w * x_idx):(w * (x_idx + 1))] = vis_img
                else:
                    vis_save_path = os.path.join(
                        self.save_dir, 'vis', filename.replace(
                            '.jpg', '_{}.jpg'.format(i))
                    )
                    vis_save_dir = '/'.join(vis_save_path.split('/')[:-1])
                    if not os.path.exists(vis_save_dir):
                        os.makedirs(vis_save_dir)
                    cv2.imwrite(vis_save_path, vis_img)

            if self.save_type == 'bbox':
                save_img = img[y1:y2, x1:x2]
            elif self.save_type == 'mask':
                save_img = img * masks[:, :, np.newaxis]
                save_img = save_img[y1:y2, x1:x2]

            # save masks
            mask_save_path = os.path.join(
                self.save_dir, 'masks', filename.replace(
                    '.jpg', '_{}.png'.format(i))
            )
            mask_save_dir = '/'.join(mask_save_path.split('/')[:-1])
            if not os.path.exists(mask_save_dir):
                os.makedirs(mask_save_dir)
            cv2.imwrite(mask_save_path,
                        masks[y1:y2, x1:x2].astype(np.uint8) * 255)

            save_path = os.path.join(
                self.save_dir, filename.replace('.jpg', '_{}.jpg'.format(i)))
            save_dir = '/'.join(save_path.split('/')[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            cv2.imwrite(save_path, save_img)
            label_json = {
                'img_path': save_path,
                'sub': subject,
                'obj': object,
                'rel': relation
            }
            save_string = json.dumps(label_json) + '\n'
            self.fo.write(save_string)

        if self.vis and self.merge_vis:
            vis_save_path = os.path.join(self.save_dir, 'vis', filename)
            vis_save_dir = '/'.join(vis_save_path.split('/')[:-1])
            if not os.path.exists(vis_save_dir):
                os.makedirs(vis_save_dir)
            cv2.imwrite(vis_save_path, merge_vis_img)

        return results
