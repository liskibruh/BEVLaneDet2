from torch import nn
from mmcv.cnn import build_conv_layer, ConvModule
from mmcv.cnn.bricks import Dropout
from mmengine.model import BaseModule, Sequential
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from ..utils.common_modules import naive_init_module, Residual
from mmdet.registry import MODELS

@MODELS.register_module()
class BinOccHead(BaseDenseHead):
    def __init__(self, in_channels, hidden_channels, num_z, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.conv2 = ConvModule(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.lateral = ConvModule(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=dict(type='BN'),
            act_cfg=None
        )
        
        self.refine = ConvModule(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.occ_pred = build_conv_layer(
            dict(type='Conv2d'),
            in_channels=hidden_channels,
            out_channels=num_z,
            kernel_size=1,
            stride=1,
            padding=0
        )
    
    def forward(self, bev_feat):
        bev_feat1 = self.conv1(bev_feat)
        bev_feat2 = self.conv2(bev_feat)
        
        bev_feat2 = self.lateral(bev_feat2)
        
        bev_feat2 = nn.functional.interpolate(
            bev_feat2,
            bev_feat1.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        feat = bev_feat1 + bev_feat2
        feat = self.refine(feat)
        
        occ_logits = self.occ_pred(feat).permute(
            0, 2, 1, 3
        ).unsqueeze(1).contiguous()
        
        return occ_logits
        
        
    
    def loss(self, bev_feat):
        occ_logits = self.forward(bev_feat)
        
        return occ_logits
    
    def loss_by_feat(self, **kwargs):
        
        pass
        