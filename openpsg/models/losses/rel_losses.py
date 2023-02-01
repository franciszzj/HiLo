import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class MultiLabelLoss(nn.Module):
    def __init__(self, loss_weight=1.0, loss_alpha=1.0, **kwargs):
        super(MultiLabelLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_alpha = loss_alpha
        self.use_sigmoid = kwargs.get('use_sigmoid', True)

    def forward(self, inputs, targets, weights, avg_factor):
        '''
        inputs: (100, 57)
        targets: (100, )
        weights: (100, )
        avg_factor: float
        '''
        # (100, ) -> (100, 57)
        targets = F.one_hot(targets.to(torch.int64),
                            num_classes=inputs.shape[1])
        # (100, 57) -> (57, 100) -> (56, 100)
        inputs = torch.permute(inputs, (1, 0))[1:]
        # (100, 57) -> (57, 100) -> (56, 100)
        targets = torch.permute(targets, (1, 0))[1:]
        loss = self.multilabel_categorical_crossentropy(targets, inputs)
        inst_weights = (loss / loss.max()) ** self.loss_alpha
        loss = loss * inst_weights * self.loss_weight
        loss = torch.mean(loss)
        return loss

    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """https://kexue.fm/archives/7359
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 9999
        y_pred_pos = y_pred - (1 - y_true) * 9999
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss


class InconsistencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=0.01, beta=0.01):
        super(InconsistencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta

    def forward(self, inconsistency_feat):
        '''
        inconsistency_feat: (N, C)
        '''
        inconsistency_feat = F.normalize(inconsistency_feat)
        inconsistency_dist = inconsistency_feat @ inconsistency_feat.transpose(0, 1)  # noqa
        nan_idx = torch.isnan(inconsistency_dist)
        inconsistency_dist[nan_idx] = 0.
        inf_idx = torch.isinf(inconsistency_dist)
        inconsistency_dist[inf_idx] = 0.
        inconsistency_dist = inconsistency_dist - torch.diag_embed(inconsistency_dist)  # noqa
        inconsistency_dist = inconsistency_dist + torch.eye(inconsistency_dist.shape[0],
                                                            dtype=inconsistency_dist.dtype,
                                                            device=inconsistency_dist.device)

        loss = (-torch.log(inconsistency_dist+self.alpha) +
                self.beta) * self.loss_weight
        loss = loss.mean()
        return loss
