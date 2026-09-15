
import logging
from torch import nn
from mmcv.cnn import build_upsample_layer, build_conv_layer, ConvModule
from mmcv.cnn.bricks import Dropout
from mmengine.model import BaseModule, kaiming_init, constant_init, Sequential
from mmengine.logging import print_log
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from ..utils.common_modules import naive_init_module, Residual
from mmdet.registry import MODELS

@MODELS.register_module()
class InstanceEmbedding(BaseModule):
    def __init__(self, ci, co=1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.neck = Sequential(
            ConvModule(
                in_channels=ci, 
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels=128,
                out_channels=ci,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            )
        )
        
        self.ms = Sequential(
            ConvModule(
                in_channels=ci,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            ),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=64,
                out_channels=1,     # co?
                kernel_size=3,
                stride=1,
                padding=1
            )
            
        )
        
        self.me = Sequential(
            ConvModule(
                in_channels=ci,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            ),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=64,
                out_channels=co,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
    
    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)

@MODELS.register_module()
class FVLaneHead(BaseDenseHead):
    def __init__(self, output_size, embed_net_cfg, input_channels=256, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.fv_up = Sequential(
            build_upsample_layer(dict(type='bilinear', scale_factor=2)),
            Residual(
                module=Sequential(
                    ConvModule(
                        in_channels=input_channels,
                        out_channels=64,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='ReLU')
                    ),
                    Dropout(drop_prob=0.2, inplace=False),
                    ConvModule(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm_cfg=dict(type='BN'),
                        act_cfg=None
                    )
                ),
                downsample=build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=input_channels,
                    out_channels=128,
                    kernel_size=1
                )
            ),
            build_upsample_layer(dict(type='bilinear', scale_factor=2)),
            Residual(
                module=Sequential(
                    ConvModule(
                        in_channels=128,
                        out_channels=64,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='ReLU')
                    ),
                    Dropout(drop_prob=0.2, inplace=False),
                    ConvModule(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm_cfg=dict(type='BN'),
                        act_cfg=None
                    )
                ),
                downsample=build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=128,
                    out_channels=32,
                    kernel_size=1
                )
            ),
            build_upsample_layer(dict(type='bilinear', size=output_size)),
            Residual(
                module=Sequential(
                    ConvModule(
                        in_channels=32,
                        out_channels=16,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='ReLU')
                    ),
                    Dropout(drop_prob=0.2, inplace=False),
                    ConvModule(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm_cfg=dict(type='BN'),
                        act_cfg=None
                    )
                )
            )
        )
        
        self.head = MODELS.build(embed_net_cfg)
    
    def forward(self, fv_x):
        fv_feat = self.fv_up(fv_x)
        return self.head(fv_feat)
    
    def loss(self, fv_x):
        fv_out = self.forward(fv_x)
        losses = self.loss_by_feat(nothing=0)
        
        # return losses
        return fv_out
    
    def loss_by_feat(self, **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head."""
        pass

@MODELS.register_module()
class InstanceEmbeddingWithOffset(BaseModule):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        pass

@MODELS.register_module()
class BEVLaneHead(BaseDenseHead):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
    
    def forward(sef, bev_x):
        pass
