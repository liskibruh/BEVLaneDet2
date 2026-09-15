import logging
from torch import nn
from ..utils.common_modules import Residual

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, kaiming_init, constant_init, Sequential
from mmengine.logging import print_log
from mmdet.registry import MODELS

@MODELS.register_module()
class FCTransform_(BaseModule):
    def __init__(self, image_featmap_size, space_featmap_size, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        ic, ih, iw = image_featmap_size
        sc, sh, sw = space_featmap_size
        
        self.image_featmpa_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        
        self.fc_transform = Sequential(
            nn.Linear(ih*iw, sh*sw),
            nn.ReLU(),
            nn.Linear(sh*sw, sh*sw),
            nn.ReLU()
        )
        
        self.conv1 = ConvModule(
            in_channels=ic,
            out_channels=sc,
            kernel_size=1,
            stride=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.residual = Residual(
            module= ConvModule(
                in_channels=sc,
                out_channels=sc,
                kernel_size=1,
                stride=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=None
            )
        )
    
    def forward(self, x):
        x = x.view(list(x.size()[:2]) + [self.image_featmpa_size[1] * self.image_featmpa_size[2], ]) # (B,C,H,W) -> (B,C,H*W)
        bev_view = self.fc_transform(x)
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])
        bev_view = self.conv1(bev_view)
        bev_view = self.residual(bev_view)
        
        return bev_view