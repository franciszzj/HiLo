# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import AssignResult, BaseAssigner, bbox_cxcywh_to_xyxy
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HTriMatcher(BaseAssigner):
    def __init__(self,
                 s_cls_cost=dict(type='ClassificationCost', weight=1.),
                 s_reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 s_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 o_cls_cost=dict(type='ClassificationCost', weight=1.),
                 o_reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 o_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 r_cls_cost=dict(type='ClassificationCost', weight=1.)):
        self.s_cls_cost = build_match_cost(s_cls_cost)
        self.s_reg_cost = build_match_cost(s_reg_cost)
        self.s_iou_cost = build_match_cost(s_iou_cost)
        self.o_cls_cost = build_match_cost(o_cls_cost)
        self.o_reg_cost = build_match_cost(o_reg_cost)
        self.o_iou_cost = build_match_cost(o_iou_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(self,
               sub_bbox_pred,
               obj_bbox_pred,
               sub_cls_score,
               obj_cls_score,
               rel_cls_score,
               gt_sub_bboxes,
               gt_obj_bboxes,
               gt_sub_labels,
               gt_obj_labels,
               gt_rel_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):

        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_sub_bboxes.size(0), sub_bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = sub_bbox_pred.new_full((num_bboxes, ),
                                                  -1,
                                                  dtype=torch.long)
        assigned_s_labels = sub_bbox_pred.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        assigned_o_labels = sub_bbox_pred.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_s_labels), AssignResult(
                                    num_gts,
                                    assigned_gt_inds,
                                    None,
                                    labels=assigned_o_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_sub_bboxes.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        s_cls_cost = self.s_cls_cost(sub_cls_score, gt_sub_labels)
        o_cls_cost = self.o_cls_cost(obj_cls_score, gt_obj_labels)
        r_cls_cost = self.r_cls_cost(rel_cls_score, gt_rel_labels)
        # regression L1 cost
        normalize_gt_sub_bboxes = gt_sub_bboxes / factor
        normalize_gt_obj_bboxes = gt_obj_bboxes / factor
        s_reg_cost = self.s_reg_cost(sub_bbox_pred, normalize_gt_sub_bboxes)
        o_reg_cost = self.o_reg_cost(obj_bbox_pred, normalize_gt_obj_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        sub_bboxes = bbox_cxcywh_to_xyxy(sub_bbox_pred) * factor
        obj_bboxes = bbox_cxcywh_to_xyxy(obj_bbox_pred) * factor
        s_iou_cost = self.s_iou_cost(sub_bboxes, gt_sub_bboxes)
        o_iou_cost = self.o_iou_cost(obj_bboxes, gt_obj_bboxes)
        # weighted sum of above three costs
        beta_1, beta_2 = 1.2, 1
        alpha_s, alpha_o, alpha_r = 1, 1, 1
        cls_cost = (alpha_s * s_cls_cost + alpha_o * o_cls_cost +
                    alpha_r * r_cls_cost) / (alpha_s + alpha_o + alpha_r)
        bbox_cost = (s_reg_cost + o_reg_cost + s_iou_cost + o_iou_cost) / 2
        cost = beta_1 * cls_cost + beta_2 * bbox_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            sub_bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            sub_bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_labels[matched_col_inds]
        assigned_o_labels[matched_row_inds] = gt_obj_labels[matched_col_inds]
        return AssignResult(num_gts,
                            assigned_gt_inds,
                            None,
                            labels=assigned_s_labels), AssignResult(
                                num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_o_labels)


@BBOX_ASSIGNERS.register_module()
class IdMatcher(BaseAssigner):
    def __init__(self,
                 sub_id_cost=dict(type='ClassificationCost', weight=1.),
                 obj_id_cost=dict(type='ClassificationCost', weight=1.),
                 r_cls_cost=dict(type='ClassificationCost', weight=1.)):
        self.sub_id_cost = build_match_cost(sub_id_cost)
        self.obj_id_cost = build_match_cost(obj_id_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(self,
               sub_match_score,
               obj_match_score,
               rel_cls_score,
               gt_sub_ids,
               gt_obj_ids,
               gt_rel_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """gt_ids are mapped from previous Hungarian matchinmg results.

        ~[0,99]
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_rel_labels.size(0), rel_cls_score.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = rel_cls_score.new_full((num_bboxes, ),
                                                  -1,
                                                  dtype=torch.long)
        assigned_s_labels = rel_cls_score.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        assigned_o_labels = rel_cls_score.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_s_labels), AssignResult(
                                    num_gts,
                                    assigned_gt_inds,
                                    None,
                                    labels=assigned_o_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        sub_id_cost = self.sub_id_cost(sub_match_score, gt_sub_ids)
        obj_id_cost = self.obj_id_cost(obj_match_score, gt_obj_ids)
        r_cls_cost = self.r_cls_cost(rel_cls_score, gt_rel_labels)

        # weighted sum of above three costs
        cost = sub_id_cost + obj_id_cost + r_cls_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            rel_cls_score.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            rel_cls_score.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_ids[matched_col_inds]
        assigned_o_labels[matched_row_inds] = gt_obj_ids[matched_col_inds]
        return AssignResult(num_gts,
                            assigned_gt_inds,
                            None,
                            labels=assigned_s_labels), AssignResult(
                                num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_o_labels)


@BBOX_ASSIGNERS.register_module()
class MaskHTriMatcher(BaseAssigner):
    def __init__(self,
                 s_cls_cost=dict(type='ClassificationCost', weight=1.),
                 s_reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 s_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 s_focal_cost=dict(type='FocalLossCost',
                                   weight=1.0, binary_input=True),
                 s_dice_cost=dict(type='DiceCost', weight=1.0,
                                  pred_act=True, eps=1.0),
                 o_cls_cost=dict(type='ClassificationCost', weight=1.),
                 o_reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 o_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 o_focal_cost=dict(type='FocalLossCost',
                                   weight=1.0, binary_input=True),
                 o_dice_cost=dict(type='DiceCost', weight=1.0,
                                  pred_act=True, eps=1.0),
                 r_cls_cost=dict(type='ClassificationCost', weight=1.)):
        self.s_cls_cost_weight = s_cls_cost['weight']
        self.s_reg_cost_weight = s_reg_cost['weight']
        self.s_iou_cost_weight = s_iou_cost['weight']
        self.s_focal_cost_weight = s_focal_cost['weight']
        self.s_dice_cost_weight = s_dice_cost['weight']
        self.o_cls_cost_weight = o_cls_cost['weight']
        self.o_reg_cost_weight = o_reg_cost['weight']
        self.o_iou_cost_weight = o_iou_cost['weight']
        self.o_focal_cost_weight = o_focal_cost['weight']
        self.o_dice_cost_weight = o_dice_cost['weight']
        self.r_cls_cost_weight = r_cls_cost['weight']
        self.s_cls_cost = build_match_cost(s_cls_cost)
        self.s_reg_cost = build_match_cost(s_reg_cost)
        self.s_iou_cost = build_match_cost(s_iou_cost)
        self.o_cls_cost = build_match_cost(o_cls_cost)
        self.o_reg_cost = build_match_cost(o_reg_cost)
        self.o_iou_cost = build_match_cost(o_iou_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)
        self.s_focal_cost = build_match_cost(s_focal_cost)
        self.s_dice_cost = build_match_cost(s_dice_cost)
        self.o_focal_cost = build_match_cost(o_focal_cost)
        self.o_dice_cost = build_match_cost(o_dice_cost)

    def assign(self,
               sub_cls_score,
               obj_cls_score,
               rel_cls_score,
               sub_bbox_pred,
               obj_bbox_pred,
               sub_mask_pred,
               obj_mask_pred,
               gt_sub_labels,
               gt_obj_labels,
               gt_rel_labels,
               gt_sub_bboxes,
               gt_obj_bboxes,
               gt_sub_masks,
               gt_obj_masks,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):

        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_sub_bboxes.size(0), sub_bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = sub_bbox_pred.new_full((num_bboxes, ),
                                                  -1,
                                                  dtype=torch.long)
        assigned_s_labels = sub_bbox_pred.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        assigned_o_labels = sub_bbox_pred.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_s_labels), AssignResult(
                                    num_gts,
                                    assigned_gt_inds,
                                    None,
                                    labels=assigned_o_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_sub_bboxes.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        s_cls_cost = self.s_cls_cost(
            sub_cls_score, gt_sub_labels) if self.s_cls_cost_weight > 1e-8 else 0.0
        o_cls_cost = self.o_cls_cost(
            obj_cls_score, gt_obj_labels) if self.o_cls_cost_weight > 1e-8 else 0.0
        r_cls_cost = self.r_cls_cost(
            rel_cls_score, gt_rel_labels) if self.r_cls_cost_weight > 1e-8 else 0.0
        # regression L1 cost
        if self.s_reg_cost_weight > 1e-8:
            normalize_gt_sub_bboxes = gt_sub_bboxes / factor
            s_reg_cost = self.s_reg_cost(
                sub_bbox_pred, normalize_gt_sub_bboxes)
        else:
            s_reg_cost = 0.0
        if self.o_reg_cost_weight > 1e-8:
            normalize_gt_obj_bboxes = gt_obj_bboxes / factor
            o_reg_cost = self.o_reg_cost(
                obj_bbox_pred, normalize_gt_obj_bboxes)
        else:
            o_reg_cost = 0.0
        # regression iou cost, defaultly giou is used in official DETR.
        if self.s_iou_cost_weight > 1e-8:
            sub_bboxes = bbox_cxcywh_to_xyxy(sub_bbox_pred) * factor
            s_iou_cost = self.s_iou_cost(sub_bboxes, gt_sub_bboxes)
        else:
            s_iou_cost = 0.0
        if self.o_iou_cost_weight > 1e-8:
            obj_bboxes = bbox_cxcywh_to_xyxy(obj_bbox_pred) * factor
            o_iou_cost = self.o_iou_cost(obj_bboxes, gt_obj_bboxes)
        else:
            o_iou_cost = 0.0
        # mask cost
        s_focal_cost = self.s_focal_cost(
            sub_mask_pred, gt_sub_masks) if self.s_focal_cost_weight > 1e-8 else 0.0
        s_dice_cost = self.s_dice_cost(
            sub_mask_pred, gt_sub_masks) if self.s_dice_cost_weight > 1e-8 else 0.0
        o_focal_cost = self.o_focal_cost(
            obj_mask_pred, gt_obj_masks) if self.o_focal_cost_weight > 1e-8 else 0.0
        o_dice_cost = self.o_dice_cost(
            obj_mask_pred, gt_obj_masks) if self.o_dice_cost_weight > 1e-8 else 0.0
        # weighted sum of above three costs
        beta_1, beta_2, beta_3 = 1.2, 1, 1
        alpha_s, alpha_o, alpha_r = 1, 1, 1
        cls_cost = (alpha_s * s_cls_cost + alpha_o * o_cls_cost +
                    alpha_r * r_cls_cost) / (alpha_s + alpha_o + alpha_r)
        bbox_cost = (s_reg_cost + o_reg_cost + s_iou_cost + o_iou_cost) / 2
        mask_cost = (s_focal_cost + o_focal_cost +
                     s_dice_cost + o_dice_cost) / 2
        cost = beta_1 * cls_cost + beta_2 * bbox_cost + beta_3 * mask_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            sub_bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            sub_bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_labels[matched_col_inds]
        assigned_o_labels[matched_row_inds] = gt_obj_labels[matched_col_inds]
        return AssignResult(num_gts,
                            assigned_gt_inds,
                            None,
                            labels=assigned_s_labels), AssignResult(
                                num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_o_labels)
