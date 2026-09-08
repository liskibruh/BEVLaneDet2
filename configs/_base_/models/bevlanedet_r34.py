from mmengine.config import read_base

with read_base():
    from ..commons import *

model = dict(
    type="BEVLaneDet",
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.288, 103.53],     # replace by 'mean' and 'std' of Apollo Lanes
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=1
    ),
    backbone=dict(
        type="ResNet",
        depth=34,
        in_channels=3,
        num_stages=4,
        out_indices=(3, ),  # only last stage features
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34'),
    ),
    bb_out_ch=512,      # depends on backbone's (resenet) "out_indices" parameter
                        # [64, 128, 256, 512] for out_indices (0, 1, 2, 3)
)