from mmengine.config import read_base

with read_base():
    from ..commons import *

model = dict(
    type="BEVLaneDet",
    # init_cfg=dict(type='Pretrained', checkpoint='path/to/bevlanedet_checkpoint.pth'),
    init_cfg=[
        dict(type='Kaiming', layer=['Conv2d', 'Linear'], mode='fan_out', nonlinearity='relu'),     # this also applies to FCTransform_ (the orig impl. does not apply it to this module)
        dict(type='Constant', layer=['_BatchNorm', 'GroupNorm'], val=1, bias=0),
    ],
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
    s32_view_transformer=dict(
        type="FCTransform_",
        image_featmap_size=(512, input_shape[0]//32, input_shape[1]//32),
        space_featmap_size=(256, 25, 5),
        init_cfg=None
    ),
    s64_view_transformer=dict(
        type="FCTransform_",
        image_featmap_size=(1024, input_shape[0]//64, input_shape[1]//64),
        space_featmap_size=(256, 25, 5),
        init_cfg=None
    ),
    fv_lane_head=dict(
        type="FVLaneHead",
        output_size=output_2d_shape,
        input_channels=512,
        embed_net_cfg=dict(
            type="InstanceEmbedding",
            ci=32,
            co=2
        )
    ),
    bev_lane_head=dict(
        type="BEVLaneHead",
        output_size=bev_shape,
        input_channels=512,
        embed_net_cfg=dict(
            type="InstanceEmbeddingWithOffset",
            ci=64,
            co=2
        )
    ),
    # lane_head=dict(
    #     fv_lane_head=dict(
    #         type="FVLaneHead",
    #         output_size=output_2d_shape,
    #         input_channels=512,
    #         embed_net_cfg=dict(
    #             type="InstanceEmbedding",
    #             ci=32,
    #             co=2
    #         )
    #     ),
    #     bev_lane_head=dict(
    #         type="BEVLaneHead",
    #         output_size=bev_shape,
    #         input_channels=512,
    #         embed_net_cfg=dict(
    #             type="InstanceEmbedding_offset_y_z",
    #             ci=64,
    #             co=2
    #         )
    #     )
    # ),
    bb_out_ch=512,      # depends on backbone's (resenet) "out_indices" parameter
                        # [64, 128, 256, 512] for out_indices (0, 1, 2, 3)
)