# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS, SingleStageDetector

from openpsg.models.frameworks.psgmaskformer import triplet2Result


@DETECTORS.register_module()
class PSGMask2Former(SingleStageDetector):
    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PSGMask2Former, self).__init__(backbone, None, bbox_head, train_cfg,
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
