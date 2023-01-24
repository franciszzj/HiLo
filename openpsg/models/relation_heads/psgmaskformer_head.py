# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.cnn import Conv2d, Linear, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import force_fp32

from mmdet.core import (build_assigner, build_sampler, multi_apply, reduce_mean,
                        bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.utils import preprocess_panoptic_gt
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from openpsg.models.relation_heads.psgtr_head import MLP


@HEADS.register_module()
class PSGMaskFormerHead(AnchorFreeHead):

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_classes=133,
                 num_relations=56,
                 object_classes=None,
                 predicate_classes=None,
                 num_queries=100,
                 sync_cls_avg_factor=False,
                 bg_cls_weight=0.02,
                 rel_bg_cls_weight=0.02,
                 use_mask=True,
                 use_self_distillation=False,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 positional_encoding=None,
                 transformer_decoder=None,
                 sub_loss_cls=dict(type='CrossEntropyLoss',
                                   use_sigmoid=False,
                                   loss_weight=1.0,
                                   class_weight=1.0),
                 sub_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 sub_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 sub_loss_focal=dict(type='BCEFocalLoss', loss_weight=1.0),
                 sub_loss_dice=dict(type='psgtrDiceLoss', loss_weight=1.0),
                 obj_loss_cls=dict(type='CrossEntropyLoss',
                                   use_sigmoid=False,
                                   loss_weight=1.0,
                                   class_weight=1.0),
                 obj_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 obj_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 obj_loss_focal=dict(type='BCEFocalLoss', loss_weight=1.0),
                 obj_loss_dice=dict(type='psgtrDiceLoss', loss_weight=1.0),
                 rel_loss_cls=dict(type='CrossEntropyLoss',
                                   use_sigmoid=False,
                                   loss_weight=2.0,
                                   class_weight=1.0),
                 # TODO: mask sure which assigner to use, we need use mask-based assigner.
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        # 1. Config
        self.num_classes = num_classes  # 133 for COCO
        self.num_relations = num_relations  # 56 for PSG Dataset
        self.object_classes = object_classes
        self.predicate_classes = predicate_classes
        self.num_queries = num_queries  # 100
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bg_cls_weight = bg_cls_weight
        self.rel_bg_cls_weight = rel_bg_cls_weight
        self.use_mask = use_mask
        self.use_self_distillation = use_self_distillation
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 2. Head, Pixel and Transformer Decoder
        pixel_decoder.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder)[1]
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        pixel_decoder_type = pixel_decoder.get('type')
        if pixel_decoder_type == 'PixelDecoder' and (
                self.decoder_embed_dims != in_channels[-1]
                or enforce_decoder_input_project):
            self.decoder_input_proj = Conv2d(
                in_channels[-1], self.decoder_embed_dims, kernel_size=1)
        else:
            self.decoder_input_proj = nn.Identity()
        self.decoder_pe = build_positional_encoding(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, out_channels)

        # 3. Pred
        self.sub_cls_out_channels = self.num_classes if sub_loss_cls['use_sigmoid'] \
            else self.num_classes + 1
        self.obj_cls_out_channels = self.num_classes if obj_loss_cls['use_sigmoid'] \
            else self.num_classes + 1
        # self.rel_cls_out_channels = self.num_relations if rel_loss_cls['use_sigmoid'] \
        #     else self.num_relations + 1
        # if rel_loss_cls.use_sigmoid is True, the out_channels is still num_relations + 1
        # because the begin index for relation is 1, not 0. so we can set the out_channels
        # index 0 to be ignored when use_sigmoid=True, and index 0 to be no_relation when
        # use_sigmoid=False.
        self.rel_cls_out_channels = self.num_relations + 1

        self.sub_cls_embed = Linear(
            self.decoder_embed_dims, self.sub_cls_out_channels)
        self.sub_box_embed = MLP(
            self.decoder_embed_dims, self.decoder_embed_dims, 4, 3)
        self.obj_cls_embed = Linear(
            self.decoder_embed_dims, self.obj_cls_out_channels)
        self.obj_box_embed = MLP(
            self.decoder_embed_dims, self.decoder_embed_dims, 4, 3)
        self.rel_cls_embed = Linear(
            self.decoder_embed_dims, self.rel_cls_out_channels)
        self.sub_mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        self.obj_mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        # 4. Assigner and Sampler
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert sub_loss_cls['loss_weight'] == assigner['s_cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert obj_loss_cls['loss_weight'] == assigner['o_cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert rel_loss_cls['loss_weight'] == assigner['r_cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert sub_loss_bbox['loss_weight'] == assigner['s_reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert obj_loss_bbox['loss_weight'] == assigner['o_reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert sub_loss_iou['loss_weight'] == assigner['s_iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            assert obj_loss_iou['loss_weight'] == assigner['o_iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            if train_cfg.assigner.type == 'HTriMatcher':
                if 's_focal_cost' in assigner.keys():
                    del assigner['s_focal_cost']
                if 's_dice_cost' in assigner.keys():
                    del assigner['s_dice_cost']
                if 'o_focal_cost' in assigner.keys():
                    del assigner['o_focal_cost']
                if 'o_dice_cost' in assigner.keys():
                    del assigner['o_dice_cost']
            if train_cfg.assigner.type == 'MaskHTriMatcher':
                assert sub_loss_focal['loss_weight'] == assigner['s_focal_cost']['weight'], \
                    'The mask focal loss weight for loss and matcher should be exactly the same.'
                assert sub_loss_dice['loss_weight'] == assigner['s_dice_cost']['weight'], \
                    'The mask dice loss weight for loss and matcher should be exactly the same.'
                assert obj_loss_focal['loss_weight'] == assigner['o_focal_cost']['weight'], \
                    'The mask focal loss weight for loss and matcher should be exactly the same.'
                assert obj_loss_dice['loss_weight'] == assigner['o_dice_cost']['weight'], \
                    'The mask dice loss weight for loss and matcher should be exactly the same.'
            self.assigner = build_assigner(assigner)
            # following DETR sampling=False, so use PseudoSampler
            sampler_cfg = train_cfg.get('sampler', dict(type='PseudoSampler'))
            self.sampler = build_sampler(sampler_cfg, context=self)

        # 5. Loss
        # NOTE following the official DETR rep0, bg_cls_weight means
        # relative classification weight of the no-object class.
        if not sub_loss_cls.use_sigmoid:
            s_class_weight = sub_loss_cls.get('class_weight', None)
            s_class_weight = torch.ones(num_classes + 1) * s_class_weight
            # NOTE set background class as the last indice
            s_class_weight[-1] = bg_cls_weight
            sub_loss_cls.update({'class_weight': s_class_weight})
        if not obj_loss_cls.use_sigmoid:
            o_class_weight = obj_loss_cls.get('class_weight', None)
            o_class_weight = torch.ones(num_classes + 1) * o_class_weight
            # NOTE set background class as the last indice
            o_class_weight[-1] = bg_cls_weight
            obj_loss_cls.update({'class_weight': o_class_weight})
        if rel_loss_cls.use_sigmoid:
            # When use_sigmoid=True, then ignore the index 0
            rel_bg_cls_weight = 0.
        r_class_weight = rel_loss_cls.get('class_weight', None)
        r_class_weight = torch.ones(num_relations + 1) * r_class_weight
        # NOTE set background class as the first indice for relations as they are 1-based
        r_class_weight[0] = rel_bg_cls_weight
        rel_loss_cls.update({'class_weight': r_class_weight})
        if 'bg_cls_weight' in rel_loss_cls:
            rel_loss_cls.pop('bg_cls_weight')

        self.sub_loss_cls = build_loss(sub_loss_cls)  # cls
        self.sub_loss_bbox = build_loss(sub_loss_bbox)  # bbox
        self.sub_loss_iou = build_loss(sub_loss_iou)  # bbox
        self.sub_loss_focal = build_loss(sub_loss_focal)  # mask
        self.sub_loss_dice = build_loss(sub_loss_dice)  # mask
        self.obj_loss_cls = build_loss(obj_loss_cls)  # cls
        self.obj_loss_bbox = build_loss(obj_loss_bbox)  # bbox
        self.obj_loss_iou = build_loss(obj_loss_iou)  # bbox
        self.obj_loss_focal = build_loss(obj_loss_focal)  # mask
        self.obj_loss_dice = build_loss(obj_loss_dice)  # mask
        self.rel_loss_cls = build_loss(rel_loss_cls)  # rel

    def init_weights(self):
        if isinstance(self.decoder_input_proj, Conv2d):
            caffe2_xavier_init(self.decoder_input_proj, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        convert_dict = {
            'panoptic_head.': 'bbox_head.'
        }
        state_dict_keys = list(state_dict.keys())
        for k in state_dict_keys:
            for ori_key, convert_key in convert_dict.items():
                if ori_key in k:
                    convert_key = k.replace(ori_key, convert_key)
                    state_dict[convert_key] = state_dict[k]
                    del state_dict[k]
        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def _merge_gt_with_gt(self, gt_rels_list1, gt_rels_list2):
        gt_rels_list = []
        for gt_rels1, gt_rels2 in zip(gt_rels_list1, gt_rels_list2):
            assert gt_rels1.shape[0] == gt_rels2.shape[0]
            gt_rels = []
            for idx in range(gt_rels1.shape[0]):
                gt_rel1 = gt_rels1[idx]
                gt_rel2 = gt_rels2[idx]
                assert gt_rel1[0] == gt_rel1[0]
                assert gt_rel1[1] == gt_rel1[1]
                if gt_rel1[2] == gt_rel2[2]:
                    gt_rels.append(gt_rel1)
                else:
                    gt_rels.append(gt_rel1)
                    gt_rels.append(gt_rel2)
            gt_rels = torch.stack(gt_rels, dim=0)
            gt_rels_list.append(gt_rels)
        return gt_rels_list

    def _merge_pred_with_pred(self,
                              s_cls_scores1, o_cls_scores1, r_cls_scores1,
                              s_bbox_preds1, o_bbox_preds1,
                              s_mask_preds1, o_mask_preds1,
                              s_cls_scores2, o_cls_scores2, r_cls_scores2,
                              s_bbox_preds2, o_bbox_preds2,
                              s_mask_preds2, o_mask_preds2, img_metas):

        results = []
        for img_idx in range(len(img_metas)):
            img_shape = img_metas[img_idx]['img_shape']
            scale_factor = img_metas[img_idx]['scale_factor']
            # 1. get result1
            rel_pairs1, labels1, scores1, bboxes1, masks1, r_labels1, r_scores1, r_dists1 = \
                self._get_results_single_easy(
                    s_cls_scores1[img_idx].detach(), o_cls_scores1[img_idx].detach(), r_cls_scores1[img_idx].detach(),  # noqa
                    s_bbox_preds1[img_idx].detach(), o_bbox_preds1[img_idx].detach(),  # noqa
                    s_mask_preds1[img_idx].detach(), o_mask_preds1[img_idx].detach(),  # noqa
                    img_shape, scale_factor, rescale=False)
            num1 = rel_pairs1.shape[0]
            # 2. get result2
            rel_pairs2, labels2, scores2, bboxes2, masks2, r_labels2, r_scores2, r_dists2 = \
                self._get_results_single_easy(
                    s_cls_scores2[img_idx].detach(), o_cls_scores2[img_idx].detach(), r_cls_scores2[img_idx].detach(),  # noqa
                    s_bbox_preds2[img_idx].detach(), o_bbox_preds2[img_idx].detach(),  # noqa
                    s_mask_preds2[img_idx].detach(), o_mask_preds2[img_idx].detach(),  # noqa
                    img_shape, scale_factor, rescale=False)
            num2 = rel_pairs2.shape[0]

            # 3. reshape result to (n, ...)
            labels1 = labels1.reshape((2, num1)).permute((1, 0))
            scores1 = scores1.reshape((2, num1)).permute((1, 0))
            bboxes1 = bboxes1.reshape((2, num1, 5)).permute((1, 0, 2))
            h, w = masks1.shape[-2:]
            masks1 = masks1.reshape((2, num1, h, w)).permute((1, 0, 2, 3))

            labels2 = labels2.reshape((2, num2)).permute((1, 0))
            scores2 = scores2.reshape((2, num2)).permute((1, 0))
            bboxes2 = bboxes2.reshape((2, num2, 5)).permute((1, 0, 2))
            h, w = masks2.shape[-2:]
            masks2 = masks2.reshape((2, num2, h, w)).permute((1, 0, 2, 3))

            # 4. concat result1 and result2
            labels_all = torch.cat((labels1, labels2), dim=0)
            scores_all = torch.cat((scores1, scores2), dim=0)
            bboxes_all = torch.cat((bboxes1, bboxes2), dim=0)
            masks_all = torch.cat((masks1, masks2), dim=0)
            r_labels_all = torch.cat((r_labels1, r_labels2), dim=0)
            r_scores_all = torch.cat((r_scores1, r_scores2), dim=0)
            r_dists_all = torch.cat((r_dists1, r_dists2), dim=0)

            # 5. re-arrange based on r_scores, output r_idxes
            r_idxes = torch.argsort(r_scores_all, dim=0, descending=True)

            # 6. re-arrange based on r_idxes
            labels = labels_all[r_idxes]
            scores = scores_all[r_idxes]
            bboxes = bboxes_all[r_idxes]
            masks = masks_all[r_idxes]
            r_labels = r_labels_all[r_idxes]
            r_scores = r_scores_all[r_idxes]
            r_dists = r_dists_all[r_idxes]

            # 7. dedup
            keep_tri = self._dedup_triplets_based_on_iou(
                labels[:, 0], labels[:, 1], r_labels, masks[:, 0], masks[:, 1])
            rel_pairs = torch.asarray([i for i in range(keep_tri.sum()*2)],
                                      dtype=r_labels.dtype, device=r_labels.device)
            labels = labels[keep_tri]
            scores = scores[keep_tri]
            bboxes = bboxes[keep_tri]
            masks = masks[keep_tri]
            r_labels = r_labels[keep_tri]
            r_scores = r_scores[keep_tri]
            r_dists = r_dists[keep_tri]

            # 8. reshape to (n*2, ...)
            labels = labels.permute((1, 0)).reshape((-1))
            scores = scores.permute((1, 0)).reshape((-1))
            bboxes = bboxes.permute((1, 0, 2)).reshape((-1, 5))
            masks = masks.permute((1, 0, 2, 3)).reshape((-1, h, w))

            results.append([rel_pairs, labels, scores, bboxes,
                           masks, r_labels, r_scores, r_dists])
        return results

    def _merge_gt_with_pred(self, gt_labels_list, gt_bboxes_list, gt_masks_list, gt_rels_list,
                            pred_results, img_metas):
        merge_gt_labels_list = []
        merge_gt_bboxes_list = []
        merge_gt_masks_list = []
        merge_gt_rels_list = []
        for img_idx in range(len(img_metas)):
            rel_pairs, labels, scores, det_bboxes, output_masks, r_labels, r_scores, r_dists = \
                pred_results[img_idx]

            merge_gt_labels = torch.cat(
                (gt_labels_list[img_idx], labels), dim=0)
            merge_gt_bboxes = torch.cat(
                (gt_bboxes_list[img_idx], det_bboxes[:, :4]), dim=0)
            merge_gt_masks = torch.cat(
                (gt_masks_list[img_idx], output_masks), dim=0)
            if rel_pairs.shape[0] > 0 and r_labels.shape[0] > 0:
                rel_pairs += gt_labels_list[img_idx].shape[0]
                rel_pairs = rel_pairs.reshape(2, -1).T
                rels = torch.cat((rel_pairs, r_labels.unsqueeze(1)), dim=1)
                merge_gt_rels = torch.cat((gt_rels_list[img_idx], rels), dim=0)
            else:
                merge_gt_rels = gt_rels_list[img_idx]

            merge_gt_labels_list.append(merge_gt_labels)
            merge_gt_bboxes_list.append(merge_gt_bboxes)
            merge_gt_masks_list.append(merge_gt_masks)
            merge_gt_rels_list.append(merge_gt_rels)
        return merge_gt_labels_list, merge_gt_bboxes_list, merge_gt_masks_list, merge_gt_rels_list

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds', 'all_mask_preds'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             all_mask_preds,
             gt_labels_list,
             gt_bboxes_list,
             gt_masks_list,
             gt_rels_list,
             img_metas,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        all_s_cls_scores = all_cls_scores['sub']
        all_o_cls_scores = all_cls_scores['obj']
        all_r_cls_scores = all_cls_scores['rel']
        all_s_bbox_preds = all_bbox_preds['sub']
        all_o_bbox_preds = all_bbox_preds['obj']
        all_s_mask_preds = all_mask_preds['sub']
        all_o_mask_preds = all_mask_preds['obj']

        # self-distillation
        if self.use_self_distillation:
            pred_results = []
            for img_idx in range(len(img_metas)):
                img_shape = img_metas[img_idx]['img_shape']
                scale_factor = img_metas[img_idx]['scale_factor']
                rel_pairs, labels, scores, bboxes, masks, r_labels, r_scores, r_dists = \
                    self._get_results_single_easy(
                        all_s_cls_scores[-1][img_idx].detach(), all_o_cls_scores[-1][img_idx].detach(), all_r_cls_scores[-1][img_idx].detach(),  # noqa
                        all_s_bbox_preds[-1][img_idx].detach(), all_o_bbox_preds[-1][img_idx].detach(),  # noqa
                        all_s_mask_preds[-1][img_idx].detach(), all_o_mask_preds[-1][img_idx].detach(),  # noqa
                        img_shape, scale_factor, rescale=False)
                pred_results.append([rel_pairs, labels, scores, bboxes,
                                     masks, r_labels, r_scores, r_dists])
            gt_labels_list, gt_bboxes_list, gt_masks_list, gt_rels_list = self._merge_gt_with_pred(
                gt_labels_list, gt_bboxes_list, gt_masks_list, gt_rels_list, pred_results, img_metas)

        num_dec_layers = len(all_s_cls_scores)

        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_rels_list = [gt_rels_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        s_losses_cls, o_losses_cls, r_losses_cls, \
            s_losses_bbox, o_losses_bbox, s_losses_iou, o_losses_iou, \
            s_losses_focal, o_losses_focal, s_losses_dice, o_losses_dice = \
            multi_apply(self.loss_single,
                        all_s_cls_scores, all_o_cls_scores, all_r_cls_scores,
                        all_s_bbox_preds, all_o_bbox_preds,
                        all_s_mask_preds, all_o_mask_preds,
                        all_gt_rels_list, all_gt_bboxes_list, all_gt_labels_list,
                        all_gt_masks_list, img_metas_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['s_loss_cls'] = s_losses_cls[-1]
        loss_dict['o_loss_cls'] = o_losses_cls[-1]
        loss_dict['r_loss_cls'] = r_losses_cls[-1]
        loss_dict['s_loss_bbox'] = s_losses_bbox[-1]
        loss_dict['o_loss_bbox'] = o_losses_bbox[-1]
        loss_dict['s_loss_iou'] = s_losses_iou[-1]
        loss_dict['o_loss_iou'] = o_losses_iou[-1]
        if self.use_mask:
            loss_dict['s_loss_focal'] = s_losses_focal[-1]
            loss_dict['s_loss_dice'] = s_losses_dice[-1]
            loss_dict['o_loss_focal'] = o_losses_focal[-1]
            loss_dict['o_loss_dice'] = o_losses_dice[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for s_loss_cls_i, o_loss_cls_i, r_loss_cls_i, \
            s_loss_bbox_i, o_loss_bbox_i, \
            s_loss_iou_i, o_loss_iou_i, \
            s_loss_focal_i, o_loss_focal_i, \
            s_loss_dice_i, o_loss_dice_i in zip(s_losses_cls[:-1], o_losses_cls[:-1], r_losses_cls[:-1],
                                              s_losses_bbox[:-1], o_losses_bbox[:-1],  # noqa
                                              s_losses_iou[:- 1], o_losses_iou[:-1],  # noqa
                                              s_losses_focal[:- 1], s_losses_dice[:-1],  # noqa
                                              o_losses_focal[:-1], o_losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.s_loss_cls'] = s_loss_cls_i
            loss_dict[f'd{num_dec_layer}.o_loss_cls'] = o_loss_cls_i
            loss_dict[f'd{num_dec_layer}.r_loss_cls'] = r_loss_cls_i
            loss_dict[f'd{num_dec_layer}.s_loss_bbox'] = s_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.o_loss_bbox'] = o_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.s_loss_iou'] = s_loss_iou_i
            loss_dict[f'd{num_dec_layer}.o_loss_iou'] = o_loss_iou_i
            loss_dict[f'd{num_dec_layer}.s_loss_focal'] = s_loss_focal_i
            loss_dict[f'd{num_dec_layer}.o_loss_focal'] = o_loss_focal_i
            loss_dict[f'd{num_dec_layer}.s_loss_dice'] = s_loss_dice_i
            loss_dict[f'd{num_dec_layer}.o_loss_dice'] = o_loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    s_cls_scores,
                    o_cls_scores,
                    r_cls_scores,
                    s_bbox_preds,
                    o_bbox_preds,
                    s_mask_preds,
                    o_mask_preds,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        num_imgs = s_cls_scores.size(0)

        s_cls_scores_list = [s_cls_scores[i] for i in range(num_imgs)]
        o_cls_scores_list = [o_cls_scores[i] for i in range(num_imgs)]
        r_cls_scores_list = [r_cls_scores[i] for i in range(num_imgs)]
        s_bbox_preds_list = [s_bbox_preds[i] for i in range(num_imgs)]
        o_bbox_preds_list = [o_bbox_preds[i] for i in range(num_imgs)]

        if self.use_mask:
            s_mask_preds_list = [s_mask_preds[i] for i in range(num_imgs)]
            o_mask_preds_list = [o_mask_preds[i] for i in range(num_imgs)]
        else:
            s_mask_preds_list = [None for i in range(num_imgs)]
            o_mask_preds_list = [None for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            s_cls_scores_list, o_cls_scores_list, r_cls_scores_list,
            s_bbox_preds_list, o_bbox_preds_list,
            s_mask_preds_list, o_mask_preds_list,
            gt_rels_list, gt_bboxes_list, gt_labels_list,
            gt_masks_list, img_metas, gt_bboxes_ignore_list)

        (s_labels_list, o_labels_list, r_labels_list,
         s_label_weights_list, o_label_weights_list, r_label_weights_list,
         s_bbox_targets_list, o_bbox_targets_list,
         s_bbox_weights_list, o_bbox_weights_list,
         s_mask_targets_list, o_mask_targets_list,
         num_total_pos, num_total_neg,
         s_mask_preds_list, o_mask_preds_list) = cls_reg_targets
        s_labels = torch.cat(s_labels_list, 0)
        o_labels = torch.cat(o_labels_list, 0)
        r_labels = torch.cat(r_labels_list, 0)

        s_label_weights = torch.cat(s_label_weights_list, 0)
        o_label_weights = torch.cat(o_label_weights_list, 0)
        r_label_weights = torch.cat(r_label_weights_list, 0)

        s_bbox_targets = torch.cat(s_bbox_targets_list, 0)
        o_bbox_targets = torch.cat(o_bbox_targets_list, 0)

        s_bbox_weights = torch.cat(s_bbox_weights_list, 0)
        o_bbox_weights = torch.cat(o_bbox_weights_list, 0)

        if self.use_mask:
            s_mask_targets = torch.cat(s_mask_targets_list,
                                       0).float().flatten(1)
            o_mask_targets = torch.cat(o_mask_targets_list,
                                       0).float().flatten(1)

            s_mask_preds = torch.cat(s_mask_preds_list, 0).flatten(1)
            o_mask_preds = torch.cat(o_mask_preds_list, 0).flatten(1)
            num_matches = o_mask_preds.shape[0]

            # mask loss
            s_focal_loss = self.sub_loss_focal(
                s_mask_preds, s_mask_targets, num_matches)
            s_dice_loss = self.sub_loss_dice(
                s_mask_preds, s_mask_targets,
                num_matches)

            o_focal_loss = self.obj_loss_focal(
                o_mask_preds, o_mask_targets, num_matches)
            o_dice_loss = self.obj_loss_dice(
                o_mask_preds, o_mask_targets,
                num_matches)
        else:
            s_focal_loss = None
            s_dice_loss = None
            o_focal_loss = None
            o_dice_loss = None

        # classification loss
        s_cls_scores = s_cls_scores.reshape(-1, self.sub_cls_out_channels)
        o_cls_scores = o_cls_scores.reshape(-1, self.obj_cls_out_channels)
        r_cls_scores = r_cls_scores.reshape(-1, self.rel_cls_out_channels)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                s_cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # NOTE change cls_avg_factor for objects as we do not calculate object classification loss for unmatched ones

        s_loss_cls = self.sub_loss_cls(s_cls_scores,
                                       s_labels,
                                       s_label_weights,
                                       avg_factor=num_total_pos * 1.0)

        o_loss_cls = self.obj_loss_cls(o_cls_scores,
                                       o_labels,
                                       o_label_weights,
                                       avg_factor=num_total_pos * 1.0)

        r_loss_cls = self.rel_loss_cls(r_cls_scores,
                                       r_labels,
                                       r_label_weights,
                                       avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = o_loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, s_bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        s_bbox_preds = s_bbox_preds.reshape(-1, 4)
        s_bboxes = bbox_cxcywh_to_xyxy(s_bbox_preds) * factors
        s_bboxes_gt = bbox_cxcywh_to_xyxy(s_bbox_targets) * factors

        o_bbox_preds = o_bbox_preds.reshape(-1, 4)
        o_bboxes = bbox_cxcywh_to_xyxy(o_bbox_preds) * factors
        o_bboxes_gt = bbox_cxcywh_to_xyxy(o_bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        s_loss_iou = self.sub_loss_iou(s_bboxes,
                                       s_bboxes_gt,
                                       s_bbox_weights,
                                       avg_factor=num_total_pos)
        o_loss_iou = self.obj_loss_iou(o_bboxes,
                                       o_bboxes_gt,
                                       o_bbox_weights,
                                       avg_factor=num_total_pos)

        # regression L1 loss
        s_loss_bbox = self.sub_loss_bbox(s_bbox_preds,
                                         s_bbox_targets,
                                         s_bbox_weights,
                                         avg_factor=num_total_pos)
        o_loss_bbox = self.obj_loss_bbox(o_bbox_preds,
                                         o_bbox_targets,
                                         o_bbox_weights,
                                         avg_factor=num_total_pos)
        return s_loss_cls, o_loss_cls, r_loss_cls, \
            s_loss_bbox, o_loss_bbox, s_loss_iou, o_loss_iou, \
            s_focal_loss, s_dice_loss, o_focal_loss, o_dice_loss

    def get_targets(self,
                    s_cls_scores_list,
                    o_cls_scores_list,
                    r_cls_scores_list,
                    s_bbox_preds_list,
                    o_bbox_preds_list,
                    s_mask_preds_list,
                    o_mask_preds_list,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(s_cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (s_labels_list, o_labels_list, r_labels_list,
         s_label_weights_list, o_label_weights_list, r_label_weights_list,
         s_bbox_targets_list, o_bbox_targets_list,
         s_bbox_weights_list, o_bbox_weights_list,
         s_mask_targets_list, o_mask_targets_list,
         pos_inds_list, neg_inds_list,
         s_mask_preds_list, o_mask_preds_list) = multi_apply(
             self._get_target_single,
             s_cls_scores_list, o_cls_scores_list, r_cls_scores_list,
             s_bbox_preds_list, o_bbox_preds_list,
             s_mask_preds_list, o_mask_preds_list,
             gt_rels_list, gt_bboxes_list, gt_labels_list, gt_masks_list,
             img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (s_labels_list, o_labels_list, r_labels_list,
                s_label_weights_list, o_label_weights_list, r_label_weights_list,
                s_bbox_targets_list, o_bbox_targets_list,
                s_bbox_weights_list, o_bbox_weights_list,
                s_mask_targets_list, o_mask_targets_list,
                num_total_pos, num_total_neg,
                s_mask_preds_list, o_mask_preds_list)

    def _get_target_single(self,
                           s_cls_score,
                           o_cls_score,
                           r_cls_score,
                           s_bbox_pred,
                           o_bbox_pred,
                           s_mask_preds,
                           o_mask_preds,
                           gt_rels,
                           gt_bboxes,
                           gt_labels,
                           gt_masks,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            s_cls_score (Tensor): Subject box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            o_cls_score (Tensor): Object box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            r_cls_score (Tensor): Relation score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            s_bbox_pred (Tensor): Sigmoid outputs of Subject bboxes from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            o_bbox_pred (Tensor): Sigmoid outputs of object bboxes from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            s_mask_preds (Tensor): Logits before sigmoid subject masks from a single decoder layer
                for one image, with shape [num_queries, H, W].
            o_mask_preds (Tensor): Logits before sigmoid object masks from a single decoder layer
                for one image, with shape [num_queries, H, W].
            gt_rels (Tensor): Ground truth relation triplets for one image with
                shape (num_gts, 3) in [gt_sub_id, gt_obj_id, gt_rel_class] format.
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - s/o/r_labels (Tensor): Labels of each image.
                - s/o/r_label_weights (Tensor]): Label weights of each image.
                - s/o_bbox_targets (Tensor): BBox targets of each image.
                - s/o_bbox_weights (Tensor): BBox weights of each image.
                - s/o_mask_targets (Tensor): Mask targets of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
                - s/o_mask_preds (Tensor): Matched mask preds of each image.
        """

        num_bboxes = s_bbox_pred.size(0)
        gt_sub_bboxes = []
        gt_obj_bboxes = []
        gt_sub_labels = []
        gt_obj_labels = []
        gt_rel_labels = []
        if self.use_mask:
            gt_sub_masks = []
            gt_obj_masks = []

        assert len(gt_masks) == len(gt_bboxes)

        for rel_id in range(gt_rels.size(0)):
            gt_sub_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 0])])
            gt_obj_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 1])])
            gt_sub_labels.append(gt_labels[int(gt_rels[rel_id, 0])])
            gt_obj_labels.append(gt_labels[int(gt_rels[rel_id, 1])])
            gt_rel_labels.append(gt_rels[rel_id, 2])
            if self.use_mask:
                gt_sub_masks.append(gt_masks[int(gt_rels[rel_id,
                                                         0])].unsqueeze(0))
                gt_obj_masks.append(gt_masks[int(gt_rels[rel_id,
                                                         1])].unsqueeze(0))

        gt_sub_bboxes = torch.vstack(
            gt_sub_bboxes).type_as(gt_bboxes).reshape(-1, 4)
        gt_obj_bboxes = torch.vstack(
            gt_obj_bboxes).type_as(gt_bboxes).reshape(-1, 4)
        gt_sub_labels = torch.vstack(
            gt_sub_labels).type_as(gt_labels).reshape(-1)
        gt_obj_labels = torch.vstack(
            gt_obj_labels).type_as(gt_labels).reshape(-1)
        gt_rel_labels = torch.vstack(
            gt_rel_labels).type_as(gt_labels).reshape(-1)
        if self.use_mask:
            gt_sub_masks = torch.vstack(
                gt_sub_masks).type_as(gt_masks)
            gt_obj_masks = torch.vstack(
                gt_obj_masks).type_as(gt_masks)
            s_mask_preds = F.interpolate(s_mask_preds[:, None],
                                         size=gt_sub_masks.shape[-2:],
                                         mode='bilinear',
                                         align_corners=False).squeeze(1)
            o_mask_preds = F.interpolate(o_mask_preds[:, None],
                                         size=gt_obj_masks.shape[-2:],
                                         mode='bilinear',
                                         align_corners=False).squeeze(1)

        # assigner and sampler, only return subject&object assign result
        if self.train_cfg.assigner.type == 'HTriMatcher':
            s_assign_result, o_assign_result = self.assigner.assign(
                s_bbox_pred, o_bbox_pred,
                s_cls_score, o_cls_score, r_cls_score,
                gt_sub_bboxes, gt_obj_bboxes,
                gt_sub_labels, gt_obj_labels, gt_rel_labels,
                img_meta, gt_bboxes_ignore)
        elif self.train_cfg.assigner.type == 'MaskHTriMatcher':
            s_assign_result, o_assign_result = self.assigner.assign(
                s_cls_score, o_cls_score, r_cls_score,
                s_bbox_pred, o_bbox_pred,
                s_mask_preds, o_mask_preds,
                gt_sub_labels, gt_obj_labels, gt_rel_labels,
                gt_sub_bboxes, gt_obj_bboxes,
                gt_sub_masks, gt_obj_masks,
                img_meta, gt_bboxes_ignore)
        else:
            assert False, 'Not support model.train_cfg.assigner.type: {}'.format(
                self.train_cfg.assigner.type)

        s_sampling_result = self.sampler.sample(s_assign_result, s_bbox_pred,
                                                gt_sub_bboxes)
        o_sampling_result = self.sampler.sample(o_assign_result, o_bbox_pred,
                                                gt_obj_bboxes)
        pos_inds = o_sampling_result.pos_inds
        neg_inds = o_sampling_result.neg_inds  # no-rel class indices in prediction

        # label targets
        s_labels = gt_sub_bboxes.new_full(
            (num_bboxes, ), self.num_classes,
            dtype=torch.long)  # 0-based, class [num_classes]  as background
        s_labels[pos_inds] = gt_sub_labels[
            s_sampling_result.pos_assigned_gt_inds]
        s_label_weights = gt_sub_bboxes.new_zeros(num_bboxes)
        s_label_weights[pos_inds] = 1.0

        o_labels = gt_obj_bboxes.new_full(
            (num_bboxes, ), self.num_classes,
            dtype=torch.long)  # 0-based, class [num_classes] as background
        o_labels[pos_inds] = gt_obj_labels[
            o_sampling_result.pos_assigned_gt_inds]
        o_label_weights = gt_obj_bboxes.new_zeros(num_bboxes)
        o_label_weights[pos_inds] = 1.0

        r_labels = gt_obj_bboxes.new_full(
            (num_bboxes, ), 0,
            dtype=torch.long)  # 1-based, class 0 as background
        r_labels[pos_inds] = gt_rel_labels[
            o_sampling_result.pos_assigned_gt_inds]
        r_label_weights = gt_obj_bboxes.new_ones(num_bboxes)

        if self.use_mask:

            # gt_sub_masks = torch.cat(gt_sub_masks, axis=0).type_as(gt_masks[0])
            # gt_obj_masks = torch.cat(gt_obj_masks, axis=0).type_as(gt_masks[0])

            assert gt_sub_masks.size() == gt_obj_masks.size()
            # mask targets for subjects and objects
            s_mask_targets = gt_sub_masks[
                s_sampling_result.pos_assigned_gt_inds,
                ...]
            s_mask_preds = s_mask_preds[pos_inds]

            o_mask_targets = gt_obj_masks[
                o_sampling_result.pos_assigned_gt_inds, ...]
            o_mask_preds = o_mask_preds[pos_inds]

            # s_mask_preds = F.interpolate(s_mask_preds[:, None],
            #                              size=gt_sub_masks.shape[-2:],
            #                              mode='bilinear',
            #                              align_corners=False).squeeze(1)

            # o_mask_preds = F.interpolate(o_mask_preds[:, None],
            #                              size=gt_obj_masks.shape[-2:],
            #                              mode='bilinear',
            #                              align_corners=False).squeeze(1)
        else:
            s_mask_targets = None
            s_mask_preds = None
            o_mask_targets = None
            o_mask_preds = None

        # bbox targets for subjects and objects
        s_bbox_targets = torch.zeros_like(s_bbox_pred)
        s_bbox_weights = torch.zeros_like(s_bbox_pred)
        s_bbox_weights[pos_inds] = 1.0

        o_bbox_targets = torch.zeros_like(o_bbox_pred)
        o_bbox_weights = torch.zeros_like(o_bbox_pred)
        o_bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = o_bbox_pred.new_tensor([img_w, img_h, img_w,
                                         img_h]).unsqueeze(0)

        pos_gt_s_bboxes_normalized = s_sampling_result.pos_gt_bboxes / factor
        pos_gt_s_bboxes_targets = bbox_xyxy_to_cxcywh(
            pos_gt_s_bboxes_normalized)
        s_bbox_targets[pos_inds] = pos_gt_s_bboxes_targets

        pos_gt_o_bboxes_normalized = o_sampling_result.pos_gt_bboxes / factor
        pos_gt_o_bboxes_targets = bbox_xyxy_to_cxcywh(
            pos_gt_o_bboxes_normalized)
        o_bbox_targets[pos_inds] = pos_gt_o_bboxes_targets

        return (s_labels, o_labels, r_labels,
                s_label_weights, o_label_weights, r_label_weights,
                s_bbox_targets, o_bbox_targets,
                s_bbox_weights, o_bbox_weights,
                s_mask_targets, o_mask_targets,
                pos_inds, neg_inds,
                s_mask_preds, o_mask_preds
                )  # return the interpolated predicted masks

    def forward(self, feats, img_metas, **kwargs):
        # 1. Forward
        batch_size = len(img_metas)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        padding_mask = feats[-1].new_ones(
            (batch_size, input_img_h, input_img_w), dtype=torch.float32)
        for i in range(batch_size):
            img_h, img_w, _ = img_metas[i]['img_shape']
            padding_mask[i, :img_h, :img_w] = 0
        padding_mask = F.interpolate(
            padding_mask.unsqueeze(1),
            size=feats[-1].shape[-2:],
            mode='nearest').to(torch.bool).squeeze(1)
        # when backbone is swin, memory is output of last stage of swin.
        # when backbone is r50, memory is output of tranformer encoder.
        mask_features, memory = self.pixel_decoder(feats, img_metas)
        pos_embed = self.decoder_pe(padding_mask)
        memory = self.decoder_input_proj(memory)
        # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
        memory = memory.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # shape (batch_size, h * w)
        padding_mask = padding_mask.flatten(1)
        # shape = (num_queries, embed_dims)
        query_embed = self.query_embed.weight
        # shape = (num_queries, batch_size, embed_dims)
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        target = torch.zeros_like(query_embed)
        # shape (num_decoder, num_queries, batch_size, embed_dims)
        out_dec = self.transformer_decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=padding_mask)
        # shape (num_decoder, batch_size, num_queries, embed_dims)
        out_dec = out_dec.transpose(1, 2)

        # 2. Get outputs
        sub_outputs_class = self.sub_cls_embed(out_dec)
        obj_outputs_class = self.obj_cls_embed(out_dec)
        rel_outputs_class = self.rel_cls_embed(out_dec)
        all_cls_scores = dict(sub=sub_outputs_class,
                              obj=obj_outputs_class,
                              rel=rel_outputs_class)

        sub_outputs_coord = self.sub_box_embed(out_dec).sigmoid()
        obj_outputs_coord = self.obj_box_embed(out_dec).sigmoid()
        all_bbox_preds = dict(sub=sub_outputs_coord,
                              obj=obj_outputs_coord)

        sub_mask_embed = self.sub_mask_embed(out_dec)
        sub_outputs_mask = torch.einsum(
            'lbqc,bchw->lbqhw', sub_mask_embed, mask_features)
        obj_mask_embed = self.obj_mask_embed(out_dec)
        obj_outputs_mask = torch.einsum(
            'lbqc,bchw->lbqhw', obj_mask_embed, mask_features)
        all_mask_preds = dict(sub=sub_outputs_mask,
                              obj=obj_outputs_mask)

        return all_cls_scores, all_bbox_preds, all_mask_preds

    def forward_train(self,
                      feats,
                      img_metas,
                      gt_rels,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # not consider ignoring bboxes
        assert gt_bboxes_ignore is None

        # forward
        all_cls_scores, all_bbox_preds, all_mask_preds = self(feats, img_metas)

        # loss
        loss_inputs = (all_cls_scores, all_bbox_preds, all_mask_preds) + \
            (gt_labels, gt_bboxes, gt_masks, gt_rels, img_metas, gt_bboxes_ignore)
        losses = self.loss(*loss_inputs)

        return losses

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'mask_preds'))
    def get_results(self, cls_scores, bbox_preds, mask_preds, img_metas, rescale=False):

        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.

        result_list = []
        for img_id in range(len(img_metas)):
            s_cls_score = cls_scores['sub'][-1, img_id, ...]
            o_cls_score = cls_scores['obj'][-1, img_id, ...]
            r_cls_score = cls_scores['rel'][-1, img_id, ...]
            s_bbox_pred = bbox_preds['sub'][-1, img_id, ...]
            o_bbox_pred = bbox_preds['obj'][-1, img_id, ...]
            s_mask_pred = mask_preds['sub'][-1, img_id, ...]
            o_mask_pred = mask_preds['obj'][-1, img_id, ...]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            triplets = self._get_results_single(s_cls_score, o_cls_score,
                                                r_cls_score, s_bbox_pred,
                                                o_bbox_pred, s_mask_pred,
                                                o_mask_pred, img_shape,
                                                scale_factor, rescale)
            result_list.append(triplets)

        return result_list

    def _get_results_single(self,
                            s_cls_score,
                            o_cls_score,
                            r_cls_score,
                            s_bbox_pred,
                            o_bbox_pred,
                            s_mask_pred,
                            o_mask_pred,
                            img_shape,
                            scale_factor,
                            rescale=False):

        assert len(s_cls_score) == len(o_cls_score)
        assert len(s_cls_score) == len(s_bbox_pred)
        assert len(s_cls_score) == len(o_bbox_pred)

        mask_size = (round(img_shape[0] / scale_factor[1]),
                     round(img_shape[1] / scale_factor[0]))
        max_per_img = self.test_cfg.get('max_per_img', self.num_queries)

        assert self.sub_loss_cls.use_sigmoid == False
        assert self.obj_loss_cls.use_sigmoid == False
        # assert self.rel_loss_cls.use_sigmoid == False
        assert len(s_cls_score) == len(r_cls_score)

        # 0-based label input for objects and self.num_classes as default background cls
        s_logits = F.softmax(s_cls_score, dim=-1)[..., :-1]
        o_logits = F.softmax(o_cls_score, dim=-1)[..., :-1]

        s_scores, s_labels = s_logits.max(-1)
        o_scores, o_labels = o_logits.max(-1)

        if self.rel_loss_cls.use_sigmoid:
            r_cls_score[:, 0] = -9999
            r_cls_score = F.sigmoid(r_cls_score)
            max_rel_index = torch.argmax(r_cls_score, dim=1)
            offset = max_rel_index.new_tensor(
                [i * r_cls_score.shape[-1] for i in range(r_cls_score.shape[0])])
            max_rel_index += offset
            # No relationship is in index 0.
            r_cls_score[:, 0] = 1. - torch.take(r_cls_score, max_rel_index)
            r_lgs = r_cls_score
        else:
            r_lgs = F.softmax(r_cls_score, dim=-1)
        r_logits = r_lgs[..., 1:]
        r_scores, r_indexes = r_logits.reshape(-1).topk(max_per_img)
        r_labels = r_indexes % self.num_relations + 1
        triplet_index = r_indexes // self.num_relations

        s_scores = s_scores[triplet_index]
        s_labels = s_labels[triplet_index] + 1
        s_bbox_pred = s_bbox_pred[triplet_index]

        o_scores = o_scores[triplet_index]
        o_labels = o_labels[triplet_index] + 1
        o_bbox_pred = o_bbox_pred[triplet_index]

        r_dists = r_lgs.reshape(
            -1, self.num_relations +
            1)[triplet_index]  # NOTE: to match the evaluation in vg

        if self.use_mask:
            s_mask_pred = s_mask_pred[triplet_index]
            o_mask_pred = o_mask_pred[triplet_index]
            s_mask_pred = F.interpolate(s_mask_pred.unsqueeze(1),
                                        size=mask_size).squeeze(1)
            o_mask_pred = F.interpolate(o_mask_pred.unsqueeze(1),
                                        size=mask_size).squeeze(1)

            s_mask_pred_logits = s_mask_pred
            o_mask_pred_logits = o_mask_pred

            s_mask_pred = torch.sigmoid(s_mask_pred) > 0.85
            o_mask_pred = torch.sigmoid(o_mask_pred) > 0.85
            ### triplets deduplicate####
            relation_classes = defaultdict(lambda: [])
            for k, (s_l, o_l, r_l) in enumerate(zip(s_labels, o_labels, r_labels)):
                relation_classes[(s_l.item(), o_l.item(),
                                  r_l.item())].append(k)
            s_binary_masks = s_mask_pred.to(torch.float).flatten(1)
            o_binary_masks = o_mask_pred.to(torch.float).flatten(1)

            def dedup_triplets(triplets_ids, s_binary_masks, o_binary_masks, keep_tri):
                while len(triplets_ids) > 1:
                    base_s_mask = s_binary_masks[triplets_ids[0]].unsqueeze(0)
                    base_o_mask = o_binary_masks[triplets_ids[0]].unsqueeze(0)
                    other_s_masks = s_binary_masks[triplets_ids[1:]]
                    other_o_masks = o_binary_masks[triplets_ids[1:]]
                    # calculate ious
                    s_ious = base_s_mask.mm(other_s_masks.transpose(
                        0, 1))/((base_s_mask+other_s_masks) > 0).sum(-1)
                    o_ious = base_o_mask.mm(other_o_masks.transpose(
                        0, 1))/((base_o_mask+other_o_masks) > 0).sum(-1)
                    ids_left = []
                    for s_iou, o_iou, other_id in zip(s_ious[0], o_ious[0], triplets_ids[1:]):
                        if (s_iou > 0.5) & (o_iou > 0.5):
                            keep_tri[other_id] = False
                        else:
                            ids_left.append(other_id)
                    triplets_ids = ids_left
                return keep_tri

            keep_tri = torch.ones_like(r_labels, dtype=torch.bool)
            for triplets_ids in relation_classes.values():
                if len(triplets_ids) > 1:
                    keep_tri = dedup_triplets(
                        triplets_ids, s_binary_masks, o_binary_masks, keep_tri)

            s_labels = s_labels[keep_tri]
            o_labels = o_labels[keep_tri]
            s_mask_pred = s_mask_pred[keep_tri]
            o_mask_pred = o_mask_pred[keep_tri]

            complete_labels = torch.cat((s_labels, o_labels), 0)
            output_masks = torch.cat((s_mask_pred, o_mask_pred), 0)
            r_scores = r_scores[keep_tri]
            r_labels = r_labels[keep_tri]
            r_dists = r_dists[keep_tri]
            rel_pairs = torch.arange(keep_tri.sum()*2,
                                     dtype=torch.int).reshape(2, -1).T
            complete_r_scores = r_scores
            complete_r_labels = r_labels
            complete_r_dists = r_dists

            s_binary_masks = s_binary_masks[keep_tri]
            o_binary_masks = o_binary_masks[keep_tri]

            s_mask_pred_logits = s_mask_pred_logits[keep_tri]
            o_mask_pred_logits = o_mask_pred_logits[keep_tri]

            ###end triplets deduplicate####

            #### for panoptic postprocessing ####
            keep = (s_labels != (s_logits.shape[-1] - 1)) & (
                o_labels != (s_logits.shape[-1] - 1)) & (
                    s_scores[keep_tri] > 0.5) & (o_scores[keep_tri] > 0.5) & (r_scores > 0.3)  # the threshold is set to 0.85
            r_scores = r_scores[keep]
            r_labels = r_labels[keep]
            r_dists = r_dists[keep]

            labels = torch.cat((s_labels[keep], o_labels[keep]), 0) - 1
            masks = torch.cat((s_mask_pred[keep], o_mask_pred[keep]), 0)
            binary_masks = masks.to(torch.float).flatten(1)
            s_mask_pred_logits = s_mask_pred_logits[keep]
            o_mask_pred_logits = o_mask_pred_logits[keep]
            mask_logits = torch.cat(
                (s_mask_pred_logits, o_mask_pred_logits), 0)

            h, w = masks.shape[-2:]

            if labels.numel() == 0:
                pan_img = torch.ones(mask_size).cpu().to(torch.long)
                pan_masks = pan_img.unsqueeze(0).cpu().to(torch.long)
                pan_rel_pairs = torch.arange(len(labels), dtype=torch.int).to(
                    masks.device).reshape(2, -1).T
                rels = torch.tensor([0, 0, 0]).view(-1, 3)
                pan_labels = torch.tensor([0])
            else:
                stuff_equiv_classes = defaultdict(lambda: [])
                thing_classes = defaultdict(lambda: [])
                thing_dedup = defaultdict(lambda: [])
                for k, label in enumerate(labels):
                    if label.item() >= 80:
                        stuff_equiv_classes[label.item()].append(k)
                    else:
                        thing_classes[label.item()].append(k)

                pan_rel_pairs = torch.arange(
                    len(labels), dtype=torch.int).to(masks.device)

                def dedup_things(pred_ids, binary_masks):
                    while len(pred_ids) > 1:
                        base_mask = binary_masks[pred_ids[0]].unsqueeze(0)
                        other_masks = binary_masks[pred_ids[1:]]
                        # calculate ious
                        ious = base_mask.mm(other_masks.transpose(
                            0, 1))/((base_mask+other_masks) > 0).sum(-1)
                        ids_left = []
                        thing_dedup[pred_ids[0]].append(pred_ids[0])
                        for iou, other_id in zip(ious[0], pred_ids[1:]):
                            if iou > 0.5:
                                thing_dedup[pred_ids[0]].append(other_id)
                            else:
                                ids_left.append(other_id)
                        pred_ids = ids_left
                    if len(pred_ids) == 1:
                        thing_dedup[pred_ids[0]].append(pred_ids[0])

                # create dict that groups duplicate masks
                for thing_pred_ids in thing_classes.values():
                    if len(thing_pred_ids) > 1:
                        dedup_things(thing_pred_ids, binary_masks)
                    else:
                        thing_dedup[thing_pred_ids[0]].append(
                            thing_pred_ids[0])

                def get_ids_area(masks, pan_rel_pairs, r_labels, r_dists, dedup=False):
                    # This helper function creates the final panoptic segmentation image
                    # It also returns the area of the masks that appears on the image
                    masks = masks.flatten(1)
                    m_id = masks.transpose(0, 1).softmax(-1)

                    if m_id.shape[-1] == 0:
                        # We didn't detect any mask :(
                        m_id = torch.zeros((h, w),
                                           dtype=torch.long,
                                           device=m_id.device)
                    else:
                        m_id = m_id.argmax(-1).view(h, w)

                    if dedup:
                        # Merge the masks corresponding to the same stuff class
                        for equiv in stuff_equiv_classes.values():
                            if len(equiv) > 1:
                                for eq_id in equiv:
                                    m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
                                    pan_rel_pairs[eq_id] = equiv[0]
                        # Merge the masks corresponding to the same thing instance
                        for equiv in thing_dedup.values():
                            if len(equiv) > 1:
                                for eq_id in equiv:
                                    m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
                                    pan_rel_pairs[eq_id] = equiv[0]

                    m_ids_remain, _ = m_id.unique().sort()

                    pan_rel_pairs = pan_rel_pairs.reshape(2, -1).T
                    no_obj_filter = torch.zeros(
                        pan_rel_pairs.shape[0], dtype=torch.bool)
                    for triplet_id in range(pan_rel_pairs.shape[0]):
                        if pan_rel_pairs[triplet_id, 0] in m_ids_remain and pan_rel_pairs[triplet_id, 1] in m_ids_remain:
                            no_obj_filter[triplet_id] = True
                    pan_rel_pairs = pan_rel_pairs[no_obj_filter]
                    r_labels, r_dists = r_labels[no_obj_filter], r_dists[no_obj_filter]
                    pan_labels = []
                    pan_masks = []
                    for i, m_id_remain in enumerate(m_ids_remain):
                        pan_masks.append(m_id.eq(m_id_remain).unsqueeze(0))
                        pan_labels.append(labels[m_id_remain].unsqueeze(0))
                        m_id.masked_fill_(m_id.eq(m_id_remain), i)
                        pan_rel_pairs.masked_fill_(
                            pan_rel_pairs.eq(m_id_remain), i)
                    pan_masks = torch.cat(pan_masks, 0)
                    pan_labels = torch.cat(pan_labels, 0)
                    seg_img = m_id * INSTANCE_OFFSET + pan_labels[m_id]
                    seg_img = seg_img.view(h, w).cpu().to(torch.long)
                    m_id = m_id.view(h, w).cpu()
                    area = []
                    for i in range(len(masks)):
                        area.append(m_id.eq(i).sum().item())
                    return area, seg_img, pan_rel_pairs, pan_masks, r_labels, r_dists, pan_labels

                area, pan_img, pan_rel_pairs, pan_masks, r_labels, r_dists, pan_labels = get_ids_area(
                    mask_logits, pan_rel_pairs, r_labels, r_dists, dedup=True)
                if r_labels.numel() == 0:
                    rels = torch.tensor([0, 0, 0]).view(-1, 3)
                else:
                    rels = torch.cat(
                        (pan_rel_pairs, r_labels.unsqueeze(-1)), -1)
                # if labels.numel() > 0:
                #     # We know filter empty masks as long as we find some
                #     while True:
                #         filtered_small = torch.as_tensor(
                #             [area[i] <= 4 for i, c in enumerate(labels)],
                #             dtype=torch.bool,
                #             device=keep.device)
                #         if filtered_small.any().item():
                #             scores = scores[~filtered_small]
                #             labels = labels[~filtered_small]
                #             masks = masks[~filtered_small]
                #             area, pan_img = get_ids_area(masks, scores)
                #         else:
                #             break

        s_det_bboxes = bbox_cxcywh_to_xyxy(s_bbox_pred)
        s_det_bboxes[:, 0::2] = s_det_bboxes[:, 0::2] * img_shape[1]
        s_det_bboxes[:, 1::2] = s_det_bboxes[:, 1::2] * img_shape[0]
        s_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        s_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            s_det_bboxes /= s_det_bboxes.new_tensor(scale_factor)
        s_det_bboxes = torch.cat((s_det_bboxes, s_scores.unsqueeze(1)), -1)

        o_det_bboxes = bbox_cxcywh_to_xyxy(o_bbox_pred)
        o_det_bboxes[:, 0::2] = o_det_bboxes[:, 0::2] * img_shape[1]
        o_det_bboxes[:, 1::2] = o_det_bboxes[:, 1::2] * img_shape[0]
        o_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        o_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            o_det_bboxes /= o_det_bboxes.new_tensor(scale_factor)
        o_det_bboxes = torch.cat((o_det_bboxes, o_scores.unsqueeze(1)), -1)

        det_bboxes = torch.cat(
            (s_det_bboxes[keep_tri], o_det_bboxes[keep_tri]), 0)

        if self.use_mask:
            return det_bboxes, complete_labels, rel_pairs, output_masks, pan_rel_pairs, \
                pan_img, complete_r_scores, complete_r_labels, complete_r_dists, r_scores, r_labels, r_dists, pan_masks, rels, pan_labels
        else:
            return det_bboxes, labels, rel_pairs, r_labels, r_dists

    def simple_test(self, feats, img_metas, rescale=False, **kwargs):
        """Test without augmentaton.

        Args:
            feats (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two tensors.

            - mask_cls_results (Tensor): Mask classification logits,\
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            - mask_pred_results (Tensor): Mask logits, shape \
                (batch_size, num_queries, h, w).
        """
        outs = self(feats, img_metas, **kwargs)
        results_list = self.get_results(*outs, img_metas, rescale=rescale)
        return results_list

    def _dedup_triplets_based_on_iou(self, s_labels, o_labels, r_labels, s_mask_pred, o_mask_pred):
        relation_classes = defaultdict(lambda: [])
        for k, (s_l, o_l, r_l) in enumerate(zip(s_labels, o_labels, r_labels)):
            relation_classes[(s_l.item(), o_l.item(),
                              r_l.item())].append(k)
        s_binary_masks = s_mask_pred.to(torch.float).flatten(1)
        o_binary_masks = o_mask_pred.to(torch.float).flatten(1)

        def dedup_triplets(triplets_ids, s_binary_masks, o_binary_masks, keep_tri):
            while len(triplets_ids) > 1:
                base_s_mask = s_binary_masks[triplets_ids[0]].unsqueeze(0)
                base_o_mask = o_binary_masks[triplets_ids[0]].unsqueeze(0)
                other_s_masks = s_binary_masks[triplets_ids[1:]]
                other_o_masks = o_binary_masks[triplets_ids[1:]]
                # calculate ious
                s_ious = base_s_mask.mm(other_s_masks.transpose(
                    0, 1))/((base_s_mask+other_s_masks) > 0).sum(-1)
                o_ious = base_o_mask.mm(other_o_masks.transpose(
                    0, 1))/((base_o_mask+other_o_masks) > 0).sum(-1)
                ids_left = []
                for s_iou, o_iou, other_id in zip(s_ious[0], o_ious[0], triplets_ids[1:]):
                    if (s_iou > 0.5) & (o_iou > 0.5):
                        keep_tri[other_id] = False
                    else:
                        ids_left.append(other_id)
                triplets_ids = ids_left
            return keep_tri

        keep_tri = torch.ones_like(
            r_labels, dtype=torch.bool, device=r_labels.device)
        for triplets_ids in relation_classes.values():
            if len(triplets_ids) > 1:
                keep_tri = dedup_triplets(
                    triplets_ids, s_binary_masks, o_binary_masks, keep_tri)

        return keep_tri

    def _get_results_single_easy(self,
                                 s_cls_score, o_cls_score, r_cls_score,
                                 s_bbox_pred, o_bbox_pred,
                                 s_mask_pred, o_mask_pred,
                                 img_shape, scale_factor, rescale=False):

        # because input is half size of mask, here should follow pre-process, not post-process.
        # mask_size = (round(img_shape[0] / scale_factor[1]),
        #              round(img_shape[1] / scale_factor[0]))
        mask_size = (img_shape[0] // 2, img_shape[1] // 2)
        max_per_img = self.num_queries

        ###################
        # sub/obj/rel cls #
        ###################
        s_logits = F.softmax(s_cls_score, dim=-1)[..., :-1]
        o_logits = F.softmax(o_cls_score, dim=-1)[..., :-1]
        s_scores, s_labels = s_logits.max(-1)
        o_scores, o_labels = o_logits.max(-1)

        r_lgs = F.softmax(r_cls_score, dim=-1)
        r_logits = r_lgs[..., 1:]
        r_scores, r_indexes = r_logits.reshape(-1).topk(max_per_img)
        r_labels = r_indexes % self.num_relations + 1
        triplet_index = r_indexes // self.num_relations

        s_scores = s_scores[triplet_index]
        s_labels = s_labels[triplet_index]
        s_bbox_pred = s_bbox_pred[triplet_index]
        s_mask_pred = s_mask_pred[triplet_index]

        o_scores = o_scores[triplet_index]
        o_labels = o_labels[triplet_index]
        o_bbox_pred = o_bbox_pred[triplet_index]
        o_mask_pred = o_mask_pred[triplet_index]

        r_dists = r_lgs.reshape(-1, self.num_relations + 1)[triplet_index]

        # same as post-process
        keep = (s_scores > 0.5) & (o_scores > 0.5) & (r_scores > 0.3)
        s_scores = s_scores[keep]
        s_labels = s_labels[keep]
        s_bbox_pred = s_bbox_pred[keep]
        s_mask_pred = s_mask_pred[keep]
        o_scores = o_scores[keep]
        o_labels = o_labels[keep]
        o_bbox_pred = o_bbox_pred[keep]
        o_mask_pred = o_mask_pred[keep]
        r_scores = r_scores[keep]
        r_labels = r_labels[keep]
        r_dists = r_dists[keep]

        ################
        # sub/obj bbox #
        ################
        s_det_bboxes = bbox_cxcywh_to_xyxy(s_bbox_pred)
        s_det_bboxes[:, 0::2] = s_det_bboxes[:, 0::2] * img_shape[1]
        s_det_bboxes[:, 1::2] = s_det_bboxes[:, 1::2] * img_shape[0]
        s_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        s_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            s_det_bboxes /= s_det_bboxes.new_tensor(scale_factor)
        s_det_bboxes = torch.cat((s_det_bboxes, s_scores.unsqueeze(1)), -1)

        o_det_bboxes = bbox_cxcywh_to_xyxy(o_bbox_pred)
        o_det_bboxes[:, 0::2] = o_det_bboxes[:, 0::2] * img_shape[1]
        o_det_bboxes[:, 1::2] = o_det_bboxes[:, 1::2] * img_shape[0]
        o_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        o_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            o_det_bboxes /= o_det_bboxes.new_tensor(scale_factor)
        o_det_bboxes = torch.cat((o_det_bboxes, o_scores.unsqueeze(1)), -1)

        ################
        # sub/obj mask #
        ################
        s_mask_pred = F.interpolate(s_mask_pred.unsqueeze(1),
                                    size=mask_size).squeeze(1)
        o_mask_pred = F.interpolate(o_mask_pred.unsqueeze(1),
                                    size=mask_size).squeeze(1)
        s_mask_pred = torch.sigmoid(s_mask_pred) > 0.85
        o_mask_pred = torch.sigmoid(o_mask_pred) > 0.85

        ### triplets deduplicate ###
        keep_tri = self._dedup_triplets_based_on_iou(
            s_labels, o_labels, r_labels, s_mask_pred, o_mask_pred)

        scores = torch.cat((s_scores[keep_tri], o_scores[keep_tri]), 0)
        # object, (2*n)
        labels = torch.cat((s_labels[keep_tri], o_labels[keep_tri]), 0)
        # object bbox, (2*n, 5)
        det_bboxes = torch.cat(
            (s_det_bboxes[keep_tri], o_det_bboxes[keep_tri]), 0)
        # object mask, (2*n, h, w)
        output_masks = torch.cat(
            (s_mask_pred[keep_tri], o_mask_pred[keep_tri]), 0)
        # relation (n)
        r_labels = r_labels[keep_tri]
        r_scores = r_scores[keep_tri]
        r_dists = r_dists[keep_tri]
        # (n, 2)
        rel_pairs = torch.arange(keep_tri.sum()*2,
                                 dtype=r_labels.dtype,
                                 device=r_labels.device).reshape(2, -1).T

        return rel_pairs, labels, scores, det_bboxes, output_masks, r_labels, r_scores, r_dists
