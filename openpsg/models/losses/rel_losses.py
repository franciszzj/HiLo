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
