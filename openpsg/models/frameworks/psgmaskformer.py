# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings

import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS, SingleStageDetector

from openpsg.models.relation_heads.approaches import Result


@DETECTORS.register_module()
class PSGMaskFormer(SingleStageDetector):
    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PSGMaskFormer, self).__init__(backbone, None, bbox_head, train_cfg,
                                            test_cfg, pretrained, init_cfg)
        self.CLASSES = self.bbox_head.object_classes
        self.PREDICATES = self.bbox_head.predicate_classes
        self.num_classes = self.bbox_head.num_classes

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(batch_input_shape=(height, width),
                 img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_rels,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None):
        # add batch_input_shape to img_metas.
        super(SingleStageDetector, self).forward_train(img, img_metas)

        x = self.extract_feat(img)
        # prepare gt_masks, for segmentation tasks
        if self.bbox_head.use_mask:
            BS, C, H, W = img.shape
            new_gt_masks = []
            for each in gt_masks:
                mask = torch.tensor(each.to_ndarray(), device=x[0].device)
                _, h, w = mask.shape
                padding = (0, W - w, 0, H - h)
                mask = F.interpolate(F.pad(mask, padding).unsqueeze(1),
                                     size=(H // 2, W // 2),
                                     mode='nearest').squeeze(1)
                # mask = F.pad(mask, padding)
                new_gt_masks.append(mask)

            gt_masks = new_gt_masks

        losses = self.bbox_head.forward_train(x, img_metas, gt_rels, gt_bboxes,
                                              gt_labels, gt_masks,
                                              gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):

        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat,
                                                  img_metas,
                                                  rescale=rescale,
                                                  **kwargs)
        sg_results = [
            triplet2Result(triplets, self.bbox_head.use_mask)
            for triplets in results_list
        ]
        # print(time.time() - s)
        return sg_results


def triplet2Result(triplets, use_mask, eval_pan_rels=os.getenv('EVAL_PAN_RELS', 'false').lower() == 'true'):
    if isinstance(triplets, Result):
        return triplets
    if use_mask:
        complete_labels, complete_bboxes, complete_masks_binary, complete_rel_pairs, complete_r_scores, complete_r_labels, complete_r_dists, complete_triplets, \
            new_labels, new_bboxes, new_masks_binary, new_rel_pairs, new_r_scores, new_r_labels, new_r_dists, new_triplets, panoptic_seg = triplets
        complete_labels = complete_labels.detach().cpu().numpy()
        complete_bboxes = complete_bboxes.detach().cpu().numpy()
        complete_masks_binary = complete_masks_binary.detach().cpu().numpy()
        complete_rel_pairs = complete_rel_pairs.detach().cpu().numpy()
        complete_r_scores = complete_r_scores.detach().cpu().numpy()
        complete_r_labels = complete_r_labels.detach().cpu().numpy()
        complete_r_dists = complete_r_dists.detach().cpu().numpy()
        complete_triplets = complete_triplets.detach().cpu().numpy()
        new_labels = new_labels.detach().cpu().numpy()
        new_bboxes = new_bboxes.detach().cpu().numpy()
        new_masks_binary = new_masks_binary.detach().cpu().numpy()
        new_rel_pairs = new_rel_pairs.detach().cpu().numpy()
        new_r_scores = new_r_scores.detach().cpu().numpy()
        new_r_labels = new_r_labels.detach().cpu().numpy()
        new_r_dists = new_r_dists.detach().cpu().numpy()
        new_triplets = new_triplets.detach().cpu().numpy()
        panoptic_seg = panoptic_seg.detach().cpu().numpy()
        if eval_pan_rels:
            return Result(refine_bboxes=new_bboxes,  # (2*n, 5)
                          labels=new_labels+1,  # (2*n)
                          formatted_masks=dict(
                              pan_results=panoptic_seg),  # (h, w)
                          rel_pair_idxes=new_rel_pairs,  # (n, 2)
                          rel_scores=new_r_scores,  # (n)
                          rel_labels=new_r_labels,  # (n)
                          rel_dists=new_r_dists,  # (n, 57)
                          pan_results=panoptic_seg,  # (h, w)
                          masks=new_masks_binary,  # (2*n, h, w)
                          rels=new_triplets)  # (n, 3)
        else:
            return Result(refine_bboxes=complete_bboxes,  # (2*n, 5)
                          labels=complete_labels+1,  # (2*n)
                          formatted_masks=dict(
                              pan_results=panoptic_seg),  # (h, w)
                          rel_pair_idxes=complete_rel_pairs,  # (n, 2)
                          rel_scores=complete_r_scores,  # (n)
                          rel_labels=complete_r_labels,  # (n)
                          rel_dists=complete_r_dists,  # (n, 57)
                          pan_results=panoptic_seg,  # (h, w)
                          masks=complete_masks_binary,  # (2*n, h, w)
                          rels=complete_triplets)  # (n, 3)
    else:
        bboxes, labels, rel_pairs, r_labels, r_dists = triplets
        labels = labels.detach().cpu().numpy()
        bboxes = bboxes.detach().cpu().numpy()
        rel_pairs = rel_pairs.detach().cpu().numpy()
        r_labels = r_labels.detach().cpu().numpy()
        r_dists = r_dists.detach().cpu().numpy()
        return Result(
            refine_bboxes=bboxes,
            labels=labels,
            formatted_masks=dict(pan_results=None),
            rel_pair_idxes=rel_pairs,
            rel_dists=r_dists,
            rel_labels=r_labels,
            pan_results=None,
        )
