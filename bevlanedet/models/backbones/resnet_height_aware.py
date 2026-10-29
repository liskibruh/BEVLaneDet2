import torch
from mmdet.models.backbones.resnet import ResNet
from mmcv.cnn import ConvModule, build_conv_layer, build_activation_layer
from mmengine.model import BaseModule, Sequential
from mmdet.registry import MODELS
from ..utils.common_modules import Residual

class HeightBranch(BaseModule):
    """
    Implementation based on HeightBranch in https://github.com/Zhou-Renjie/HeightAware-BEV/blob/main/nets/segnet.py#L224
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=128,
                 num_blocks=2):
        super().__init__()
        
        self.conv_bn_gelu1 = ConvModule(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='GELU')
        )
        
        self.residual_blocks = Sequential(
            *[
                Residual(
                module = Sequential(
                    ConvModule(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='GELU')
                    ),
                    ConvModule(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=dict(type='GELU'),
                        act_cfg=None
                    )
                )
            )
            for _ in range(num_blocks)
            ]
        )
        
        self.height_net = Sequential(
            build_conv_layer(
                type=dict('Conv2d'),
                in_channels=hidden_channels,
                out_channels=out_channels + 1,
                kernel_size=1
            ),
            build_activation_layer(
                type=dict('Softmax'),
                dim=1
            )
        )
    
    def forward(self, x):
        out = self.conv_bn_gelu1(x)
        out = self.residual_blocks(out)
        out = self.height_net(out)
        return out

@MODELS.register_module()
class ResNet_HeightAware(ResNet):
    def __init__(self, 
                 feat_in_channels, 
                 height_in_channels, 
                 height_out_channels, 
                 height_hidden_channels=128, 
                 height_aware=True, 
                 **kwargs):
        
        super().__init__(
            in_channels=feat_in_channels,
            **kwargs,
            )
        
        self.height_aware = height_aware
        
        self.height_branch = HeightBranch(
            in_channels=height_in_channels,
            hidden_channels=height_hidden_channels,
            out_channels=height_out_channels
        )
    
    def forward(self, x):
        f = super().forward(x)
        
        if self.height_aware:
            h = self.height_branch(f).clone()
            return (f, h)
        
        return (f, )