

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(input_values, gamma):
    """Computes the focal loss
    
    Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    """
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss


class FocalLoss(nn.Module):
    """Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, weight=self.weight, reduction=self.reduction), self.gamma)
