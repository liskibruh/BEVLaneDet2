import logging
import torch
from torch import nn
from ..utils.common_modules import Residual

from mmengine.model import kaiming_init, constant_init, Sequential
from mmengine.logging import print_log
from mmcv.cnn import ConvModule
from mmdet.models import BaseDetector
from mmdet.registry import MODELS

@MODELS.register_module()
class BEVLaneDet(BaseDetector):
    def __init__(self, data_preprocessor, backbone, s32_view_transformer, s64_view_transformer, lane_head, bb_out_ch=512, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.backbone = MODELS.build(backbone)
        
        self.down = Residual(
            module = Sequential(
                ConvModule(
                    in_channels=bb_out_ch, 
                    out_channels=1024, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    norm_cfg=dict(type='BN'), 
                    act_cfg=dict(type='ReLU')
                    ),
                ConvModule(
                    in_channels=1024, 
                    out_channels=1024, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    norm_cfg=dict(type='BN'), 
                    act_cfg=None
                    )
            ),
            downsample = nn.Conv2d(bb_out_ch, 1024, kernel_size=3, stride=2, padding=1)
        )
        
        self.s32_view_transformer = MODELS.build(s32_view_transformer)
        self.s64_view_transformer = MODELS.build(s64_view_transformer)
        
        self.lane_head = MODELS.build(lane_head)
    
    def extract_feat(self, input):
        feat = self.backbone(input)
        return feat
    
    def _forward(self, inputs, data_samples=None):
        img_s32 = self.extract_feat(inputs)[0]      # only last stage feature
        print(f"img_s32.shape: {img_s32.shape}")
        img_s64 = self.down(img_s32)
        print(f"img_s64.shape: {img_s64.shape}")
        bev_32 = self.s32_view_transformer(img_s32)
        print(f"bev_32.shape: {bev_32.shape}")
        bev_64 = self.s64_view_transformer(img_s64)
        print(f"bev_64.shape: {bev_64.shape}")
        bev = torch.cat([bev_32, bev_64], dim=1)
        print(f"bev.shape: {bev.shape}")
                
        out = self.lane_head(img_s32)
        
        
        return out
    
    def loss(self):
        pass
    
    def predict(self):
        pass