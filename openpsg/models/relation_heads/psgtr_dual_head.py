# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply, reduce_mean,
                        bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from openpsg.models.relation_heads.psgtr_head import (
    PSGTrHead, MLP, MaskHeadSmallConv, MHAttentionMap)


@HEADS.register_module()
class PSGTrDualHead(PSGTrHead):
    def __init__(
            self,
            in_channels,
            num_classes=133,
            num_relations=56,
            object_classes=None,
            predicate_classes=None,
            num_query=100,
            sync_cls_avg_factor=False,
            bg_cls_weight=0.02,
            use_mask=True,
            num_reg_fcs=2,
            n_heads=8,
            swin_backbone=None,
            dual_head=dict(use_object_cls=False,
                           use_relation_cls=False,
                           use_bbox=False,
                           use_mask=True),
            positional_encoding=dict(type='SinePositionalEncoding',
                                     num_feats=128,
                                     normalize=True),
            transformer=None,
            sub_loss_cls=dict(type='CrossEntropyLoss',
                              use_sigmoid=False,
                              loss_weight=1.0,
                              class_weight=1.0),
            sub_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            sub_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            sub_focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
            sub_dice_loss=dict(type='psgtrDiceLoss', loss_weight=1.0),
            obj_loss_cls=dict(type='CrossEntropyLoss',
                              use_sigmoid=False,
                              loss_weight=1.0,
                              class_weight=1.0),
            obj_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            obj_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            obj_focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
            obj_dice_loss=dict(type='psgtrDiceLoss', loss_weight=1.0),
            rel_loss_cls=dict(type='CrossEntropyLoss',
                              use_sigmoid=False,
                              loss_weight=2.0,
                              class_weight=1.0),
            train_cfg=dict(assigner=dict(
                type='HTriMatcher',
                s_cls_cost=dict(type='ClassificationCost', weight=1.),
                s_reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                s_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                o_cls_cost=dict(type='ClassificationCost', weight=1.),
                o_reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                o_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                r_cls_cost=dict(type='ClassificationCost', weight=2.))),
            test_cfg=dict(max_per_img=100),
            init_cfg=None,
            **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        # 1. Config
        self.num_classes = num_classes  # 133 for COCO
        self.num_relations = num_relations  # 56 for PSG Dataset
        self.object_classes = object_classes
        self.predicate_classes = predicate_classes
        self.num_query = num_query  # 100
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bg_cls_weight = bg_cls_weight
        self.use_mask = use_mask
        self.num_reg_fcs = num_reg_fcs  # not use
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.swin = swin_backbone
        self.dual_head = dual_head

        # 2. Head, Transformer
        self.n_heads = n_heads
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)  # not use
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self.input_proj = Conv2d(in_channels,
                                 self.embed_dims,
                                 kernel_size=1)
        self.query_embed = nn.Embedding(self.num_query, self.embed_dims)

        # Dual Head
        self.dual_transformer = build_transformer(transformer)
        self.dual_embed_dims = self.dual_transformer.embed_dims
        self.dual_input_dropout = nn.Dropout(p=0.2)
        self.dual_query_embed = nn.Embedding(
            self.num_query, self.dual_embed_dims)

        # 3. Pred
        self.sub_cls_out_channels = self.num_classes if sub_loss_cls['use_sigmoid'] \
            else self.num_classes + 1
        self.obj_cls_out_channels = self.num_classes if obj_loss_cls['use_sigmoid'] \
            else self.num_classes + 1
        self.rel_cls_out_channels = self.num_relations if rel_loss_cls['use_sigmoid'] \
            else self.num_relations + 1

        self.sub_cls_embed = Linear(
            self.embed_dims, self.sub_cls_out_channels)
        self.sub_box_embed = MLP(
            self.embed_dims, self.embed_dims, 4, 3)
        self.obj_cls_embed = Linear(
            self.embed_dims, self.obj_cls_out_channels)
        self.obj_box_embed = MLP(
            self.embed_dims, self.embed_dims, 4, 3)
        self.rel_cls_embed = Linear(
            self.embed_dims, self.rel_cls_out_channels)
        if self.use_mask:
            self.sub_bbox_attention = MHAttentionMap(
                self.embed_dims,
                self.embed_dims,
                self.n_heads,
                dropout=0.0)
            self.obj_bbox_attention = MHAttentionMap(
                self.embed_dims,
                self.embed_dims,
                self.n_heads,
                dropout=0.0)
            if not self.swin:
                self.sub_mask_head = MaskHeadSmallConv(
                    self.embed_dims + self.n_heads,
                    [1024, 512, 256],
                    self.embed_dims)
                self.obj_mask_head = MaskHeadSmallConv(
                    self.embed_dims + self.n_heads,
                    [1024, 512, 256],
                    self.embed_dims)
            elif self.swin:
                self.sub_mask_head = MaskHeadSmallConv(
                    self.embed_dims + self.n_heads,
                    self.swin,
                    self.embed_dims)
                self.obj_mask_head = MaskHeadSmallConv(
                    self.embed_dims + self.n_heads,
                    self.swin,
                    self.embed_dims)

            # Dual Mask Pred
            self.dual_sub_bbox_attention = MHAttentionMap(
                self.dual_embed_dims,
                self.dual_embed_dims,
                self.n_heads,
                dropout=0.0)
            self.dual_obj_bbox_attention = MHAttentionMap(
                self.dual_embed_dims,
                self.dual_embed_dims,
                self.n_heads,
                dropout=0.0)
            if not self.swin:
                self.dual_sub_mask_head = MaskHeadSmallConv(
                    self.dual_embed_dims + self.n_heads,
                    [1024, 512, 256],
                    self.dual_embed_dims)
                self.dual_obj_mask_head = MaskHeadSmallConv(
                    self.dual_embed_dims + self.n_heads, [1024, 512, 256],
                    self.dual_embed_dims)
            elif self.swin:
                self.dual_sub_mask_head = MaskHeadSmallConv(
                    self.dual_embed_dims + self.n_heads,
                    self.swin,
                    self.dual_embed_dims)
                self.obj_mask_head = MaskHeadSmallConv(
                    self.dual_embed_dims + self.n_heads,
                    self.swin,
                    self.dual_embed_dims)

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
                assert sub_focal_loss['loss_weight'] == assigner['s_focal_cost']['weight'], \
                    'The mask focal loss weight for loss and matcher should be exactly the same.'
                assert sub_dice_loss['loss_weight'] == assigner['s_dice_cost']['weight'], \
                    'The mask dice loss weight for loss and matcher should be exactly the same.'
                assert obj_focal_loss['loss_weight'] == assigner['o_focal_cost']['weight'], \
                    'The mask focal loss weight for loss and matcher should be exactly the same.'
                assert obj_dice_loss['loss_weight'] == assigner['o_dice_cost']['weight'], \
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
        if not rel_loss_cls.use_sigmoid:
            r_class_weight = rel_loss_cls.get('class_weight', None)
            r_class_weight = torch.ones(num_relations + 1) * r_class_weight
            # NOTE set background class as the first indice for relations as they are 1-based
            r_class_weight[0] = bg_cls_weight
            rel_loss_cls.update({'class_weight': r_class_weight})
            if 'bg_cls_weight' in rel_loss_cls:
                rel_loss_cls.pop('bg_cls_weight')

        self.sub_loss_cls = build_loss(sub_loss_cls)  # cls
        self.sub_loss_bbox = build_loss(sub_loss_bbox)  # bbox
        self.sub_loss_iou = build_loss(sub_loss_iou)  # bbox
        self.obj_loss_cls = build_loss(obj_loss_cls)  # cls
        self.obj_loss_bbox = build_loss(obj_loss_bbox)  # bbox
        self.obj_loss_iou = build_loss(obj_loss_iou)  # bbox
        if self.use_mask:
            self.obj_focal_loss = build_loss(obj_focal_loss)  # mask
            self.obj_dice_loss = build_loss(obj_dice_loss)  # mask
            self.sub_focal_loss = build_loss(sub_focal_loss)  # mask
            self.sub_dice_loss = build_loss(sub_dice_loss)  # mask
        self.rel_loss_cls = build_loss(rel_loss_cls)  # rel

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        # Dual Head
        self.dual_transformer.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        version = local_metadata.get('version', None)
        if (version is None or version < 2):
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.',
                '.query_embedding.': '.query_embed.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]
        # Dual Head
        copy_dict = {
            'transformer.': 'dual_transformer.'
        }
        state_dict_keys = list(state_dict.keys())
        for k in state_dict_keys:
            for ori_key, copy_key in copy_dict.items():
                if ori_key in k:
                    copy_key = k.replace(ori_key, copy_key)
                    state_dict[copy_key] = state_dict[k]
        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_rels_list,
             gt_bboxes_list,
             gt_labels_list,
             gt_masks_list,
             img_metas,
             gt_bboxes_ignore=None):
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list
        all_bbox_preds = all_bbox_preds_list
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        all_s_cls_scores = all_cls_scores['sub']
        all_o_cls_scores = all_cls_scores['obj']

        all_s_bbox_preds = all_bbox_preds['sub']
        all_o_bbox_preds = all_bbox_preds['obj']

        num_dec_layers = len(all_s_cls_scores)

        if self.use_mask:
            all_s_mask_preds = all_bbox_preds['sub_seg']
            all_o_mask_preds = all_bbox_preds['obj_seg']
            all_s_mask_preds = [
                all_s_mask_preds for _ in range(num_dec_layers)]
            all_o_mask_preds = [
                all_o_mask_preds for _ in range(num_dec_layers)]
            # Dual Mask Pred
            all_dual_s_mask_preds = all_bbox_preds['dual_sub_seg']
            all_dual_o_mask_preds = all_bbox_preds['dual_obj_seg']
            all_dual_s_mask_preds = [
                all_dual_s_mask_preds for _ in range(num_dec_layers)]
            all_dual_o_mask_preds = [
                all_dual_o_mask_preds for _ in range(num_dec_layers)]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_rels_list = [gt_rels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        all_r_cls_scores = [None for _ in range(num_dec_layers)]
        all_r_cls_scores = all_cls_scores['rel']

        if self.use_mask:
            s_losses_cls, o_losses_cls, r_losses_cls, \
                s_losses_bbox, o_losses_bbox, s_losses_iou, o_losses_iou, \
                s_focal_losses, s_dice_losses, o_focal_losses, o_dice_losses, \
                dual_s_focal_losses, dual_s_dice_losses, dual_o_focal_losses, dual_o_dice_losses = \
                multi_apply(self.loss_single,
                            all_s_cls_scores, all_o_cls_scores, all_r_cls_scores,
                            all_s_bbox_preds, all_o_bbox_preds,
                            all_s_mask_preds, all_o_mask_preds,
                            all_dual_s_mask_preds, all_dual_o_mask_preds,
                            all_gt_rels_list, all_gt_bboxes_list, all_gt_labels_list,
                            all_gt_masks_list, img_metas_list, all_gt_bboxes_ignore_list)
        else:
            all_s_mask_preds = [None for _ in range(num_dec_layers)]
            all_o_mask_preds = [None for _ in range(num_dec_layers)]
            all_dual_s_mask_preds = [None for _ in range(num_dec_layers)]
            all_dual_o_mask_preds = [None for _ in range(num_dec_layers)]
            s_losses_cls, o_losses_cls, r_losses_cls, \
                s_losses_bbox, o_losses_bbox, s_losses_iou, o_losses_iou, \
                s_focal_losses, s_dice_losses, o_focal_losses, o_dice_losses, \
                dual_s_focal_losses, dual_s_dice_losses, dual_o_focal_losses, dual_o_dice_losses = \
                multi_apply(self.loss_single,
                            all_s_cls_scores, all_o_cls_scores, all_r_cls_scores,
                            all_s_bbox_preds, all_o_bbox_preds,
                            all_s_mask_preds, all_o_mask_preds,
                            all_dual_s_mask_preds, all_dual_o_mask_preds,
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
            loss_dict['s_focal_losses'] = s_focal_losses[-1]
            loss_dict['o_focal_losses'] = o_focal_losses[-1]
            loss_dict['s_dice_losses'] = s_dice_losses[-1]
            loss_dict['o_dice_losses'] = o_dice_losses[-1]
            # Dual Mask Pred
            loss_dict['dual_s_focal_losses'] = dual_s_focal_losses[-1]
            loss_dict['dual_o_focal_losses'] = dual_o_focal_losses[-1]
            loss_dict['dual_s_dice_losses'] = dual_s_dice_losses[-1]
            loss_dict['dual_o_dice_losses'] = dual_o_dice_losses[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for s_loss_cls_i, o_loss_cls_i, r_loss_cls_i, \
            s_loss_bbox_i, o_loss_bbox_i, \
            s_loss_iou_i, o_loss_iou_i in zip(s_losses_cls[:-1], o_losses_cls[:-1], r_losses_cls[:-1],
                                              s_losses_bbox[:-1], o_losses_bbox[:-1],  # noqa
                                              s_losses_iou[:-1], o_losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.s_loss_cls'] = s_loss_cls_i
            loss_dict[f'd{num_dec_layer}.o_loss_cls'] = o_loss_cls_i
            loss_dict[f'd{num_dec_layer}.r_loss_cls'] = r_loss_cls_i
            loss_dict[f'd{num_dec_layer}.s_loss_bbox'] = s_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.o_loss_bbox'] = o_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.s_loss_iou'] = s_loss_iou_i
            loss_dict[f'd{num_dec_layer}.o_loss_iou'] = o_loss_iou_i
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
                    dual_s_mask_preds,
                    dual_o_mask_preds,
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
            dual_s_mask_preds_list = [dual_s_mask_preds[i]
                                      for i in range(num_imgs)]
            dual_o_mask_preds_list = [dual_o_mask_preds[i]
                                      for i in range(num_imgs)]
        else:
            s_mask_preds_list = [None for i in range(num_imgs)]
            o_mask_preds_list = [None for i in range(num_imgs)]
            dual_s_mask_preds_list = [None for i in range(num_imgs)]
            dual_o_mask_preds_list = [None for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            s_cls_scores_list, o_cls_scores_list, r_cls_scores_list,
            s_bbox_preds_list, o_bbox_preds_list,
            s_mask_preds_list, o_mask_preds_list,
            dual_s_mask_preds_list, dual_o_mask_preds_list,
            gt_rels_list, gt_bboxes_list, gt_labels_list, gt_masks_list,
            img_metas, gt_bboxes_ignore_list)

        (s_labels_list, o_labels_list, r_labels_list,
         s_label_weights_list, o_label_weights_list, r_label_weights_list,
         s_bbox_targets_list, o_bbox_targets_list,
         s_bbox_weights_list, o_bbox_weights_list,
         s_mask_targets_list, o_mask_targets_list,
         num_total_pos, num_total_neg,
         s_mask_preds_list, o_mask_preds_list,
         dual_s_mask_preds_list, dual_o_mask_preds_list) = cls_reg_targets
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
            dual_s_mask_preds = torch.cat(dual_s_mask_preds_list, 0).flatten(1)
            dual_o_mask_preds = torch.cat(dual_o_mask_preds_list, 0).flatten(1)
            num_matches = o_mask_preds.shape[0]

            # mask loss
            s_focal_loss = self.sub_focal_loss(
                s_mask_preds, s_mask_targets, num_matches)
            s_dice_loss = self.sub_dice_loss(
                s_mask_preds, s_mask_targets,
                num_matches)

            o_focal_loss = self.obj_focal_loss(
                o_mask_preds, o_mask_targets, num_matches)
            o_dice_loss = self.obj_dice_loss(
                o_mask_preds, o_mask_targets,
                num_matches)

            # Dual Mask Loss
            dual_s_focal_loss = self.sub_focal_loss(
                dual_s_mask_preds, s_mask_targets, num_matches)
            dual_s_dice_loss = self.sub_dice_loss(
                dual_s_mask_preds, s_mask_targets, num_matches)
            dual_o_focal_loss = self.obj_focal_loss(
                dual_o_mask_preds, o_mask_targets, num_matches)
            dual_o_dice_loss = self.obj_dice_loss(
                dual_o_mask_preds, o_mask_targets, num_matches)

        else:
            s_focal_loss = None
            s_dice_loss = None
            o_focal_loss = None
            o_dice_loss = None
            dual_s_focal_loss = None
            dual_s_dice_loss = None
            dual_o_focal_loss = None
            dual_o_dice_loss = None

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
            s_focal_loss, s_dice_loss, o_focal_loss, o_dice_loss, \
            dual_s_focal_loss, dual_s_dice_loss, dual_o_focal_loss, dual_o_dice_loss

    def get_targets(self,
                    s_cls_scores_list,
                    o_cls_scores_list,
                    r_cls_scores_list,
                    s_bbox_preds_list,
                    o_bbox_preds_list,
                    s_mask_preds_list,
                    o_mask_preds_list,
                    dual_s_mask_preds_list,
                    dual_o_mask_preds_list,
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
         s_mask_preds_list, o_mask_preds_list,
         dual_s_mask_preds_list, dual_o_mask_preds_list) = \
            multi_apply(self._get_target_single,
                        s_cls_scores_list, o_cls_scores_list, r_cls_scores_list,
                        s_bbox_preds_list, o_bbox_preds_list,
                        s_mask_preds_list, o_mask_preds_list,
                        dual_s_mask_preds_list, dual_o_mask_preds_list,
                        gt_rels_list, gt_bboxes_list, gt_labels_list, gt_masks_list,
                        img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (s_labels_list, o_labels_list, r_labels_list,
                s_label_weights_list, o_label_weights_list,
                r_label_weights_list, s_bbox_targets_list, o_bbox_targets_list,
                s_bbox_weights_list, o_bbox_weights_list,
                s_mask_targets_list, o_mask_targets_list,
                num_total_pos, num_total_neg,
                s_mask_preds_list, o_mask_preds_list,
                dual_s_mask_preds_list, dual_o_mask_preds_list)

    def _get_target_single(self,
                           s_cls_score,
                           o_cls_score,
                           r_cls_score,
                           s_bbox_pred,
                           o_bbox_pred,
                           s_mask_preds,
                           o_mask_preds,
                           dual_s_mask_preds,
                           dual_o_mask_preds,
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
                for one image. Shape [num_query, cls_out_channels].
            o_cls_score (Tensor): Object box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            r_cls_score (Tensor): Relation score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            s_bbox_pred (Tensor): Sigmoid outputs of Subject bboxes from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            o_bbox_pred (Tensor): Sigmoid outputs of object bboxes from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            s_mask_preds (Tensor): Logits before sigmoid subject masks from a single decoder layer
                for one image, with shape [num_query, H, W].
            o_mask_preds (Tensor): Logits before sigmoid object masks from a single decoder layer
                for one image, with shape [num_query, H, W].
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
            dual_s_mask_preds = F.interpolate(dual_s_mask_preds[:, None],
                                              size=gt_sub_masks.shape[-2:],
                                              mode='bilinear',
                                              align_corners=False).squeeze(1)
            dual_o_mask_preds = F.interpolate(dual_o_mask_preds[:, None],
                                              size=gt_obj_masks.shape[-2:],
                                              mode='bilinear',
                                              align_corners=False).squeeze(1)

        # assigner and sampler, only return subject&object assign result
        if self.train_cfg.assigner.type == 'HTriMatcher':
            s_assign_result, o_assign_result = self.assigner.assign(
                s_bbox_pred, o_bbox_pred, s_cls_score, o_cls_score, r_cls_score,
                gt_sub_bboxes, gt_obj_bboxes, gt_sub_labels, gt_obj_labels,
                gt_rel_labels, img_meta, gt_bboxes_ignore)
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
            dual_s_mask_preds = dual_s_mask_preds[pos_inds]

            o_mask_targets = gt_obj_masks[
                o_sampling_result.pos_assigned_gt_inds, ...]
            o_mask_preds = o_mask_preds[pos_inds]
            dual_o_mask_preds = dual_o_mask_preds[pos_inds]

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
            dual_s_mask_preds = None
            o_mask_targets = None
            o_mask_preds = None
            dual_o_mask_preds = None

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

        return (s_labels, o_labels, r_labels, s_label_weights, o_label_weights,
                r_label_weights, s_bbox_targets, o_bbox_targets,
                s_bbox_weights, o_bbox_weights, s_mask_targets, o_mask_targets,
                pos_inds, neg_inds, s_mask_preds, o_mask_preds,
                dual_s_mask_preds, dual_o_mask_preds
                )  # return the interpolated predicted masks

    def forward(self, feats, img_metas):
        # 1. Forward
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        last_features = feats[-1]
        batch_size = last_features.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = last_features.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        last_features = self.input_proj(last_features)
        # interpolate masks to have the same spatial shape with feats
        masks = F.interpolate(masks.unsqueeze(1),
                              size=last_features.shape[-2:]).to(
                                  torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, memory = self.transformer(
            last_features, masks, self.query_embed.weight, pos_embed)

        last_features_dropout = self.dual_input_dropout(last_features)
        dual_outs_dec, dual_memory = self.dual_transformer(
            last_features_dropout, masks, self.dual_query_embed.weight, pos_embed)

        # 2. Get outputs
        sub_outputs_class = self.sub_cls_embed(outs_dec)
        sub_outputs_coord = self.sub_box_embed(outs_dec).sigmoid()
        obj_outputs_class = self.obj_cls_embed(outs_dec)
        obj_outputs_coord = self.obj_box_embed(outs_dec).sigmoid()

        all_cls_scores = dict(sub=sub_outputs_class, obj=obj_outputs_class)
        rel_outputs_class = self.rel_cls_embed(outs_dec)
        all_cls_scores['rel'] = rel_outputs_class
        if self.use_mask:
            ###########for segmentation#################
            sub_bbox_mask = self.sub_bbox_attention(outs_dec[-1],
                                                    memory,
                                                    mask=masks)
            obj_bbox_mask = self.obj_bbox_attention(outs_dec[-1],
                                                    memory,
                                                    mask=masks)
            sub_seg_masks = self.sub_mask_head(last_features, sub_bbox_mask,
                                               [feats[2], feats[1], feats[0]])
            outputs_sub_seg_masks = sub_seg_masks.view(batch_size,
                                                       self.num_query,
                                                       sub_seg_masks.shape[-2],
                                                       sub_seg_masks.shape[-1])
            obj_seg_masks = self.obj_mask_head(last_features, obj_bbox_mask,
                                               [feats[2], feats[1], feats[0]])
            outputs_obj_seg_masks = obj_seg_masks.view(batch_size,
                                                       self.num_query,
                                                       obj_seg_masks.shape[-2],
                                                       obj_seg_masks.shape[-1])

            # Dual Mask Pred
            dual_sub_bbox_mask = self.dual_sub_bbox_attention(dual_outs_dec[-1],
                                                              dual_memory,
                                                              mask=masks)
            dual_obj_bbox_mask = self.dual_obj_bbox_attention(dual_outs_dec[-1],
                                                              dual_memory,
                                                              mask=masks)
            dual_sub_seg_masks = self.dual_sub_mask_head(last_features, dual_sub_bbox_mask,
                                                         [feats[2], feats[1], feats[0]])
            dual_outputs_sub_seg_masks = dual_sub_seg_masks.view(batch_size,
                                                                 self.num_query,
                                                                 dual_sub_seg_masks.shape[-2],
                                                                 dual_sub_seg_masks.shape[-1])
            dual_obj_seg_masks = self.dual_obj_mask_head(last_features, dual_obj_bbox_mask,
                                                         [feats[2], feats[1], feats[0]])
            dual_outputs_obj_seg_masks = dual_obj_seg_masks.view(batch_size,
                                                                 self.num_query,
                                                                 dual_obj_seg_masks.shape[-2],
                                                                 dual_obj_seg_masks.shape[-1])

            all_bbox_preds = dict(sub=sub_outputs_coord,
                                  obj=obj_outputs_coord,
                                  sub_seg=outputs_sub_seg_masks,
                                  obj_seg=outputs_obj_seg_masks,
                                  dual_sub_seg=dual_outputs_sub_seg_masks,
                                  dual_obj_seg=dual_outputs_obj_seg_masks)
        else:
            all_bbox_preds = dict(sub=sub_outputs_coord, obj=obj_outputs_coord)
        return all_cls_scores, all_bbox_preds
