import torch
from torch import nn
from mmdet.registry import MODELS

# code based on:
#   https://github.com/Arche-Logos/bev_lane_det/blob/main/models/loss/__init__.py

@MODELS.register_module()
class SoftIOULoss(nn.Module):
    def __init__(self, ignore_index=255, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps
    
    def forward(self, outputs, targets):
        mask = (targets != self.ignore_index).float()
        
        outputs = outputs.float()
        targets = targets.float()
        
        intersection = torch.sum(outputs * targets * mask)
        
        union = torch.sum(
            outputs * mask +
            targets * mask -
            outputs * targets * mask
        )
        
        return 1 - intersection / (union + self.eps)
