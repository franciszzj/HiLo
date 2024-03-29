# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import os
import sys
import cv2
import copy
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import force_fp32, ModuleList

from mmdet.core import build_assigner, build_sampler
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from openpsg.models.relation_heads.psgmaskformer_head import PSGMaskFormerHead
from openpsg.models.relation_heads.psgtr_head import MLP
from openpsg.models.losses.rel_losses import InconsistencyLoss

if (os.getenv('SAVE_PREDICT', 'false').lower() == 'true') or \
        (os.getenv('MERGE_PREDICT', 'false').lower() == 'true'):
    import dbm
    kv_db = dbm.open('./db/kv_db', 'c')


@HEADS.register_module()
class PSGMask2FormerHead(PSGMaskFormerHead):

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_classes=133,
                 num_relations=56,
                 object_classes=None,
                 predicate_classes=None,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 sync_cls_avg_factor=False,
                 bg_cls_weight=0.02,
                 rel_bg_cls_weight=0.02,
                 use_mask=True,
                 use_self_distillation=False,
                 use_relation_query=False,
                 use_mlp_before_dot_product=False,
                 use_decoder_for_relation_query=True,
                 mask_self_attn_interact_of_diff_query_types=True,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 positional_encoding=None,
                 transformer_decoder=None,
                 use_separate_head=False,
                 decoder_cfg=dict(use_query_pred=True,
                                  ignore_masked_attention_layers=[]),
                 use_inconsistency_loss=False,
                 inconsistency_loss_weight=1.0,
                 aux_loss_list=[0, 1, 2, 3, 4, 5, 6, 7],
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
        self.num_transformer_feat_level = num_transformer_feat_level
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bg_cls_weight = bg_cls_weight
        self.rel_bg_cls_weight = rel_bg_cls_weight
        self.use_mask = use_mask
        self.use_self_distillation = use_self_distillation
        self.use_relation_query = use_relation_query
        self.use_mlp_before_dot_product = use_mlp_before_dot_product
        self.use_decoder_for_relation_query = use_decoder_for_relation_query
        self.mask_self_attn_interact_of_diff_query_types = mask_self_attn_interact_of_diff_query_types
        self.use_separate_head = use_separate_head
        self.decoder_cfg = decoder_cfg
        self.use_inconsistency_loss = use_inconsistency_loss
        self.inconsistency_loss_weight = inconsistency_loss_weight
        self.aux_loss_list = aux_loss_list
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 2. Head, Pixel and Transformer Decoder
        assert pixel_decoder.encoder.transformerlayers.\
            attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]

        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_pe = build_positional_encoding(positional_encoding)
        self.query_embed = nn.Embedding(
            self.num_queries, out_channels)
        self.query_feat = nn.Embedding(
            self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)
        if self.use_relation_query:
            self.relation_query_embed = nn.Embedding(
                self.num_relations + 1, out_channels)
            self.relation_query_feat = nn.Embedding(
                self.num_relations + 1, feat_channels)

        if self.use_separate_head:
            self.sub_head = MLP(
                self.decoder_embed_dims, self.decoder_embed_dims, self.decoder_embed_dims, 3)
            self.obj_head = MLP(
                self.decoder_embed_dims, self.decoder_embed_dims, self.decoder_embed_dims, 3)
            self.rel_head = MLP(
                self.decoder_embed_dims, self.decoder_embed_dims, self.decoder_embed_dims, 3)

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
        if self.use_relation_query:
            if self.use_mlp_before_dot_product:
                self.object_query_mlp = MLP(
                    self.decoder_embed_dims, self.decoder_embed_dims, self.decoder_embed_dims, 3)
                self.relation_query_mlp = MLP(
                    self.decoder_embed_dims, self.decoder_embed_dims, self.decoder_embed_dims, 3)
        else:
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
            # NOTE: Not use
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

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

        if self.use_inconsistency_loss:
            self.inconsistency_loss = InconsistencyLoss(
                loss_weight=self.inconsistency_loss_weight)

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def loss_aux(self, all_inconsistency_feat, **kwargs):
        loss_dict = dict()
        if self.use_inconsistency_loss and all_inconsistency_feat is not None:
            loss_dict['inconsistency_loss'] = self.inconsistency_loss(
                all_inconsistency_feat[-1])
            # for i, inconsistency_feat in enumerate(all_inconsistency_feat[:-1]):
            #     loss = self.inconsistency_loss(inconsistency_feat)
            #     loss_dict['d{}.inconsistency_loss'.format(i)] = loss
        return loss_dict

    def forward_head(self, decoder_out, relation_decoder_out, mask_feature, attn_mask_target_size, decoder_layer_idx=0):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            relation_decoder_out (Tensor or None): in shape
                (num_relations + 1, batch_size, c). If it is None, then not use
                relation_decoder_out.
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.
            decoder_layer_idx (int): the index number of decoder layer.
                0 layer means the embedding output from encoder and before
                feed into the decoder.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h*w). \
                This attn_mask is used for cross attention.
        """
        debug = False

        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        if self.use_relation_query:
            if self.use_decoder_for_relation_query:
                relation_decoder_out = self.transformer_decoder.post_norm(
                    relation_decoder_out)
            relation_decoder_out = relation_decoder_out.transpose(0, 1)

        if self.use_separate_head:
            sub_decoder_out = self.sub_head(decoder_out)
            obj_decoder_out = self.obj_head(decoder_out)
            rel_decoder_out = self.rel_head(decoder_out)
        else:
            sub_decoder_out = decoder_out
            obj_decoder_out = decoder_out
            rel_decoder_out = decoder_out

        sub_output_class = self.sub_cls_embed(sub_decoder_out)
        obj_output_class = self.obj_cls_embed(obj_decoder_out)
        if self.use_relation_query:
            if self.use_mlp_before_dot_product:
                _decoder_out = self.object_query_mlp(rel_decoder_out)
                _relation_decoder_out = self.relation_query_mlp(
                    relation_decoder_out)
                rel_output_class = torch.einsum(
                    'bqc,brc->bqr', _decoder_out, _relation_decoder_out)
            else:
                rel_output_class = torch.einsum(
                    'bqc,brc->bqr', rel_decoder_out, relation_decoder_out)
        else:
            rel_output_class = self.rel_cls_embed(rel_decoder_out)
        all_cls_score = dict(sub=sub_output_class,
                             obj=obj_output_class,
                             rel=rel_output_class)

        sub_output_coord = self.sub_box_embed(sub_decoder_out).sigmoid()
        obj_output_coord = self.obj_box_embed(obj_decoder_out).sigmoid()
        all_bbox_pred = dict(sub=sub_output_coord,
                             obj=obj_output_coord)

        sub_mask_embed = self.sub_mask_embed(sub_decoder_out)
        sub_output_mask = torch.einsum(
            'bqc,bchw->bqhw', sub_mask_embed, mask_feature)
        obj_mask_embed = self.obj_mask_embed(obj_decoder_out)
        obj_output_mask = torch.einsum(
            'bqc,bchw->bqhw', obj_mask_embed, mask_feature)
        all_mask_pred = dict(sub=sub_output_mask,
                             obj=obj_output_mask)

        sub_attn_mask = F.interpolate(
            sub_output_mask,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (batch_size, num_queries, h, w) ->
        #   (batch_size * num_head, num_queries, h*w)
        sub_attn_mask = sub_attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        sub_attn_mask = sub_attn_mask.sigmoid() < 0.5
        sub_attn_mask = sub_attn_mask.detach()

        if debug:
            for i in range(sub_attn_mask.shape[0]):
                for j in range(sub_attn_mask.shape[1]):
                    mask = sub_attn_mask[i, j].reshape(
                        attn_mask_target_size).float().cpu().numpy() * 255
                    cv2.imwrite('vis/decoder_layer_{}_sub_attn_mask_head_{}_query_{}.png'.format(
                        decoder_layer_idx, i, j), mask)

        obj_attn_mask = F.interpolate(
            obj_output_mask,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (batch_size, num_queries, h, w) ->
        #   (batch_size * num_head, num_queries, h*w)
        obj_attn_mask = obj_attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        obj_attn_mask = obj_attn_mask.sigmoid() < 0.5
        obj_attn_mask = obj_attn_mask.detach()

        if debug:
            for i in range(obj_attn_mask.shape[0]):
                for j in range(obj_attn_mask.shape[1]):
                    mask = obj_attn_mask[i, j].reshape(
                        attn_mask_target_size).float().cpu().numpy() * 255
                    cv2.imwrite('vis/decoder_layer_{}_obj_attn_mask_head_{}_query_{}.png'.format(
                        decoder_layer_idx, i, j), mask)

        attn_mask = torch.mul(sub_attn_mask, obj_attn_mask)

        if debug:
            for i in range(attn_mask.shape[0]):
                for j in range(attn_mask.shape[1]):
                    mask = attn_mask[i, j].reshape(
                        attn_mask_target_size).float().cpu().numpy() * 255
                    cv2.imwrite('vis/decoder_layer_{}_attn_mask_head_{}_query_{}.png'.format(
                        decoder_layer_idx, i, j), mask)

        if self.use_relation_query and self.use_decoder_for_relation_query:
            relation_attn_mask = attn_mask.new_zeros(
                (attn_mask.shape[0], self.num_relations+1, attn_mask.shape[2]))
            all_attn_mask = torch.concat(
                (attn_mask, relation_attn_mask), dim=1)
        else:
            all_attn_mask = attn_mask

        return all_cls_score, all_bbox_pred, all_mask_pred, all_attn_mask

    def forward(self, feats, img_metas, **kwargs):
        debug = False
        if debug:
            print('\n img_metas for this test image: {}'.format(img_metas))

        # 1. Forward
        batch_size = len(img_metas)

        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_pe(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        if self.use_relation_query:
            relation_query_feat = self.relation_query_feat.weight.unsqueeze(1).repeat(
                (1, batch_size, 1))
            relation_query_embed = self.relation_query_embed.weight.unsqueeze(1).repeat(
                (1, batch_size, 1))
            if self.use_decoder_for_relation_query:
                # (num_queries, batch_size, c) and (num_relations+1， batch_size, c) ->
                # (num_queries+num_relations+1, batch_size, c)
                all_query_feat = torch.cat((
                    query_feat, relation_query_feat), dim=0)
                all_query_embed = torch.cat(
                    (query_embed, relation_query_embed), dim=0)
            else:
                all_query_feat = query_feat
                all_query_embed = query_embed
        else:
            all_query_feat = query_feat
            all_query_embed = query_embed
            relation_query_feat = None

        cls_pred_list = []
        bbox_pred_list = []
        mask_pred_list = []
        if self.decoder_cfg.get('use_query_pred', True):
            cls_pred, bbox_pred, mask_pred, cross_attn_mask = self.forward_head(
                query_feat, relation_query_feat, mask_features, multi_scale_memorys[0].shape[-2:], decoder_layer_idx=0)
            cls_pred_list.append(cls_pred)
            bbox_pred_list.append(bbox_pred)
            mask_pred_list.append(mask_pred)
        else:
            cross_attn_mask = None

        ignore_masked_attention_layers = self.decoder_cfg.get(
            'ignore_masked_attention_layers', [])

        if self.use_inconsistency_loss:
            inconsistency_feat_list = []

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            if cross_attn_mask is not None:
                cross_attn_mask[torch.where(
                    cross_attn_mask.sum(-1) == cross_attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            # cross_attn_mask is prepared for cross attention
            # if we use relation_query, then we need setup another self_attn_mask
            # for self attention, to decide whether interact triplet query
            # with relation query.
            if i in ignore_masked_attention_layers:
                cross_attn_mask = None
            if self.use_relation_query and \
                    self.use_decoder_for_relation_query and \
                    self.mask_self_attn_interact_of_diff_query_types:
                self_attn_mask = cross_attn_mask.new_zeros(
                    (cross_attn_mask.shape[0], cross_attn_mask.shape[1], cross_attn_mask.shape[1])).bool()
                # the interact area of triplet query and relation query are masked.
                self_attn_mask[:, self.num_queries:, :self.num_queries] = True
                self_attn_mask[:, :self.num_queries, self.num_queries:] = True
            else:
                self_attn_mask = None
            attn_masks = [cross_attn_mask, self_attn_mask]
            all_query_feat = layer(
                query=all_query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=all_query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)

            if self.use_relation_query:
                if self.use_decoder_for_relation_query:
                    query_feat, relation_query_feat = torch.split(
                        all_query_feat,
                        split_size_or_sections=[
                            self.num_queries, self.num_relations+1],
                        dim=0)
                else:
                    query_feat = all_query_feat
            else:
                query_feat = all_query_feat

            if (i in self.aux_loss_list) or (i+1 == self.num_transformer_decoder_layers):
                cls_pred, bbox_pred, mask_pred, cross_attn_mask = self.forward_head(
                    query_feat, relation_query_feat, mask_features, multi_scale_memorys[
                        (i + 1) % self.num_transformer_feat_level].shape[-2:], decoder_layer_idx=i+1)
                cls_pred_list.append(cls_pred)
                bbox_pred_list.append(bbox_pred)
                mask_pred_list.append(mask_pred)
            else:
                cross_attn_mask = None

            if (i in self.aux_loss_list) or (i+1 == self.num_transformer_decoder_layers):
                if self.use_inconsistency_loss:
                    inconsistency_cat_list = []
                    query_feat_ = torch.reshape(
                        query_feat, (self.num_queries, -1))
                    inconsistency_cat_list.append(query_feat_)
                    inconsistency_feat = torch.cat(
                        inconsistency_cat_list, dim=0)
                    inconsistency_feat_list.append(inconsistency_feat)

        all_s_cls_scores = []
        all_o_cls_scores = []
        all_r_cls_scores = []
        all_s_bbox_preds = []
        all_o_bbox_preds = []
        all_s_mask_preds = []
        all_o_mask_preds = []
        for cls_pred, bbox_pred, mask_pred in zip(cls_pred_list, bbox_pred_list, mask_pred_list):
            sub_cls_pred = cls_pred['sub']
            obj_cls_pred = cls_pred['obj']
            rel_cls_pred = cls_pred['rel']
            sub_bbox_pred = bbox_pred['sub']
            obj_bbox_pred = bbox_pred['obj']
            sub_mask_pred = mask_pred['sub']
            obj_mask_pred = mask_pred['obj']
            all_s_cls_scores.append(sub_cls_pred)
            all_o_cls_scores.append(obj_cls_pred)
            all_r_cls_scores.append(rel_cls_pred)
            all_s_bbox_preds.append(sub_bbox_pred)
            all_o_bbox_preds.append(obj_bbox_pred)
            all_s_mask_preds.append(sub_mask_pred)
            all_o_mask_preds.append(obj_mask_pred)
        all_s_cls_scores = torch.stack(all_s_cls_scores, dim=0)
        all_o_cls_scores = torch.stack(all_o_cls_scores, dim=0)
        all_r_cls_scores = torch.stack(all_r_cls_scores, dim=0)
        all_s_bbox_preds = torch.stack(all_s_bbox_preds, dim=0)
        all_o_bbox_preds = torch.stack(all_o_bbox_preds, dim=0)
        all_s_mask_preds = torch.stack(all_s_mask_preds, dim=0)
        all_o_mask_preds = torch.stack(all_o_mask_preds, dim=0)
        all_cls_scores = dict(
            sub=all_s_cls_scores,
            obj=all_o_cls_scores,
            rel=all_r_cls_scores)
        all_bbox_preds = dict(
            sub=all_s_bbox_preds,
            obj=all_o_bbox_preds)
        all_mask_preds = dict(
            sub=all_s_mask_preds,
            obj=all_o_mask_preds)

        if os.getenv('SAVE_PREDICT', 'false').lower() == 'true':
            for i, img_meta in enumerate(img_metas):
                # config = 'v14_2_jade2_epoch_42'
                config = 'v14_3_jade2_epoch_43'
                key = config + '#' + img_metas[i]['filename']
                value = {
                    'all_cls_scores': dict(
                        sub=all_s_cls_scores[-1:, i:(i+1)].cpu().detach().numpy(),  # noqa
                        obj=all_o_cls_scores[-1:, i:(i+1)].cpu().detach().numpy(),  # noqa
                        rel=all_r_cls_scores[-1:, i:(i+1)].cpu().detach().numpy()),  # noqa
                    'all_bbox_preds': dict(
                        sub=all_s_bbox_preds[-1:, i:(i+1)].cpu().detach().numpy(),  # noqa
                        obj=all_o_bbox_preds[-1:, i:(i+1)].cpu().detach().numpy()),  # noqa
                    'all_mask_preds': dict(
                        sub=all_s_mask_preds[-1:, i:(i+1)].cpu().detach().numpy(),  # noqa
                        obj=all_o_mask_preds[-1:, i:(i+1)].cpu().detach().numpy()),  # noqa
                }
                value = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                kv_db[key] = value

        if os.getenv('MERGE_PREDICT', 'false').lower() == 'true':
            value = {
                'all_cls_scores': dict(
                    sub=all_s_cls_scores[-1:, i:(i+1)].cpu().detach().numpy(),  # noqa
                    obj=all_o_cls_scores[-1:, i:(i+1)].cpu().detach().numpy(),  # noqa
                    rel=all_r_cls_scores[-1:, i:(i+1)].cpu().detach().numpy()),  # noqa
                'all_bbox_preds': dict(
                    sub=all_s_bbox_preds[-1:, i:(i+1)].cpu().detach().numpy(),  # noqa
                    obj=all_o_bbox_preds[-1:, i:(i+1)].cpu().detach().numpy()),  # noqa
                'all_mask_preds': dict(
                    sub=all_s_mask_preds[-1:, i:(i+1)].cpu().detach().numpy(),  # noqa
                    obj=all_o_mask_preds[-1:, i:(i+1)].cpu().detach().numpy()),  # noqa
            }

            config1 = 'v14_2_jade2_epoch_42'
            key1 = config1 + '#' + img_metas[0]['filename']
            value1 = kv_db.get(key1, value)
            if isinstance(value1, (str, bytes)):
                value1 = pickle.loads(value1)
            else:
                print('miss {}'.format(key1))

            value1 = {
                'all_cls_scores': dict(
                    sub=torch.asarray(value1['all_cls_scores']['sub'], dtype=all_s_cls_scores.dtype, device=all_s_cls_scores.device),  # noqa
                    obj=torch.asarray(value1['all_cls_scores']['obj'], dtype=all_o_cls_scores.dtype, device=all_o_cls_scores.device),  # noqa
                    rel=torch.asarray(value1['all_cls_scores']['rel'], dtype=all_r_cls_scores.dtype, device=all_r_cls_scores.device),  # noqa
                ),
                'all_bbox_preds': dict(
                    sub=torch.asarray(value1['all_bbox_preds']['sub'], dtype=all_s_bbox_preds.dtype, device=all_s_bbox_preds.device),  # noqa
                    obj=torch.asarray(value1['all_bbox_preds']['obj'], dtype=all_o_bbox_preds.dtype, device=all_o_bbox_preds.device),  # noqa
                ),
                'all_mask_preds': dict(
                    sub=torch.asarray(value1['all_mask_preds']['sub'], dtype=all_s_mask_preds.dtype, device=all_s_mask_preds.device),  # noqa
                    obj=torch.asarray(value1['all_mask_preds']['obj'], dtype=all_o_mask_preds.dtype, device=all_o_mask_preds.device),  # noqa
                ),
            }

            config2 = 'v14_3_jade2_epoch_43'
            key2 = config2 + '#' + img_metas[0]['filename']
            value2 = kv_db.get(key2, value)
            if isinstance(value2, (str, bytes)):
                value2 = pickle.loads(value2)
            else:
                print('miss {}'.format(key2))

            value2 = {
                'all_cls_scores': dict(
                    sub=torch.asarray(value2['all_cls_scores']['sub'], dtype=all_s_cls_scores.dtype, device=all_s_cls_scores.device),  # noqa
                    obj=torch.asarray(value2['all_cls_scores']['obj'], dtype=all_o_cls_scores.dtype, device=all_o_cls_scores.device),  # noqa
                    rel=torch.asarray(value2['all_cls_scores']['rel'], dtype=all_r_cls_scores.dtype, device=all_r_cls_scores.device),  # noqa
                ),
                'all_bbox_preds': dict(
                    sub=torch.asarray(value2['all_bbox_preds']['sub'], dtype=all_s_bbox_preds.dtype, device=all_s_bbox_preds.device),  # noqa
                    obj=torch.asarray(value2['all_bbox_preds']['obj'], dtype=all_o_bbox_preds.dtype, device=all_o_bbox_preds.device),  # noqa
                ),
                'all_mask_preds': dict(
                    sub=torch.asarray(value2['all_mask_preds']['sub'], dtype=all_s_mask_preds.dtype, device=all_s_mask_preds.device),  # noqa
                    obj=torch.asarray(value2['all_mask_preds']['obj'], dtype=all_o_mask_preds.dtype, device=all_o_mask_preds.device),  # noqa
                ),
            }

            pred1 = (value1['all_cls_scores'],
                     value1['all_bbox_preds'], value1['all_mask_preds'])
            pred2 = (value2['all_cls_scores'],
                     value2['all_bbox_preds'], value2['all_mask_preds'])

            gt_labels = kwargs['gt_labels'][0]
            gt_bboxes = kwargs['gt_bboxes'][0]
            gt_masks = kwargs['gt_masks'][0]
            gt_rels = kwargs['gt_rels'][0]
            gt_bboxes_ignore = None
            gt = (gt_labels, gt_bboxes, gt_masks,
                  gt_rels, img_metas, gt_bboxes_ignore)

            input1 = pred1 + gt
            input2 = pred2 + gt
            loss1 = self.loss(*input1)
            loss2 = self.loss(*input2)

            loss1_value = sum([x.cpu().detach().item()
                              for x in loss1.values()])
            loss2_value = sum([x.cpu().detach().item()
                              for x in loss2.values()])

            if loss1_value < loss2_value:
                w1 = 1.0
                w2 = 0.0
            else:
                w1 = 0.0
                w2 = 1.0

            # if random.random() > 0.5:
            #     w1 = 1.0
            #     w2 = 0.0
            # else:
            #     w1 = 0.0
            #     w2 = 1.0

            all_cls_scores = dict(
                sub=value1['all_cls_scores']['sub'] * w1 + value2['all_cls_scores']['sub'] * w2,  # noqa
                obj=value1['all_cls_scores']['obj'] * w1 + value2['all_cls_scores']['obj'] * w2,  # noqa
                rel=value1['all_cls_scores']['rel'] * w1 + value2['all_cls_scores']['rel'] * w2,  # noqa
            )
            all_bbox_preds = dict(
                sub=value1['all_bbox_preds']['sub'] * w1 + value2['all_bbox_preds']['sub'] * w2,  # noqa
                obj=value1['all_bbox_preds']['obj'] * w1 + value2['all_bbox_preds']['obj'] * w2,  # noqa
            )
            all_mask_preds = dict(
                sub=value1['all_mask_preds']['sub'] * w1 + value2['all_mask_preds']['sub'] * w2,  # noqa
                obj=value1['all_mask_preds']['obj'] * w1 + value2['all_mask_preds']['obj'] * w2,  # noqa
            )

        output_dict = dict()
        output_dict['all_cls_scores'] = all_cls_scores
        output_dict['all_bbox_preds'] = all_bbox_preds
        output_dict['all_mask_preds'] = all_mask_preds

        if self.use_inconsistency_loss:
            output_dict['all_inconsistency_feat'] = inconsistency_feat_list
        else:
            output_dict['all_inconsistency_feat'] = None

        return output_dict
