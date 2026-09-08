from torch import nn
from ..utils.common_modules import naive_init_module, Residual

from mmdet.models import BaseDetector
from mmdet.registry import MODELS

@MODELS.register_module()
class BEVLaneDet(BaseDetector):
    def __init__(self, data_preprocessor=None, backbone=None, bb_out_ch=512):
        super().__init__(data_preprocessor=data_preprocessor)
        
        self.backbone = MODELS.build(backbone)
        
        self.down = naive_init_module(Residual(
            module = nn.Sequential(
                nn.Conv2d(bb_out_ch, 1024, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024)
            ),
            downsample = nn.Conv2d(bb_out_ch, 1024, kernel_size=3, stride=2, padding=1)
        ))
    
    def extract_feat(self, input):
        feat = self.backbone(input)
        
        return feat
    
    def _forward(self, inputs, data_sample=None):
        img_s32 = self.extract_feat(inputs)
        print(f"img_s32.shape: {img_s32[0].shape}")
        img_s64 = self.down(img_s32[0])
        print(f"img_s64.shape: {img_s64.shape}")
        
        # todo: continue from here
        
        out = img_s64
        
        return out
    
    def loss(self):
        pass
    
    def predict(self):
        pass