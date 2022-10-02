# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply, reduce_mean,
                        bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from openpsg.models.relation_heads.psgtr_head import (
    PSGTrHead,  MLP, MaskHeadSmallConv, MHAttentionMap)


@HEADS.register_module()
class PSGTrODbeRHead(PSGTrHead):
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
            positional_encoding=dict(type='SinePositionalEncoding',
                                     num_feats=128,
                                     normalize=True),
            transformer=None,
            relation_decoder=None,
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

        # Relation Decoder
        self.relation_decoder = build_transformer_layer_sequence(
            relation_decoder)

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
        for p in self.relation_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

        # 2. Get outputs
        sub_outputs_class = self.sub_cls_embed(outs_dec)
        sub_outputs_coord = self.sub_box_embed(outs_dec).sigmoid()
        obj_outputs_class = self.obj_cls_embed(outs_dec)
        obj_outputs_coord = self.obj_box_embed(outs_dec).sigmoid()

        all_cls_scores = dict(sub=sub_outputs_class, obj=obj_outputs_class)

        # Relation Decoder
        # (bs, token, c) -> (token, bs, c)
        rel_query_embed = outs_dec[-1].permute(1, 0, 2)
        rel_target = torch.zeros_like(rel_query_embed)
        # (bs, c, h, w) -> (h*w, bs, c)
        rel_memory = memory.flatten(2).permute(2, 0, 1)
        # (bs, c, h, w) -> (h*w, bs, c)
        rel_pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # (bs, h, w) -> (bs, h*w)
        rel_masks = masks.flatten(1)
        # shape (num_decoder, num_queries, batch_size, embed_dims)
        rel_outs_dec = self.relation_decoder(
            query=rel_query_embed,
            key=rel_memory,
            value=rel_memory,
            key_pos=rel_pos_embed,
            query_pos=rel_target,
            key_padding_mask=rel_masks)
        # shape (num_decoder, batch_size, num_queries, embed_dims)
        rel_outs_dec = rel_outs_dec.permute(0, 2, 1, 3)
        rel_outputs_class = self.rel_cls_embed(rel_outs_dec)
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

            all_bbox_preds = dict(sub=sub_outputs_coord,
                                  obj=obj_outputs_coord,
                                  sub_seg=outputs_sub_seg_masks,
                                  obj_seg=outputs_obj_seg_masks)
        else:
            all_bbox_preds = dict(sub=sub_outputs_coord, obj=obj_outputs_coord)
        return all_cls_scores, all_bbox_preds
