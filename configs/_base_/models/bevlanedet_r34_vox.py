from mmengine.config import read_base

with read_base():
    from ..commons import *

model = dict(
    type="BEVLaneDetVox",
    # init_cfg=dict(type='Pretrained', checkpoint='path/to/bevlanedet_checkpoint.pth'),
    init_cfg=[
        dict(type='Kaiming', layer=['Conv2d', 'Linear'], mode='fan_out', nonlinearity='relu'),     # this also applies to FCTransform_ (the orig impl. does not apply it to this module)
        dict(type='Constant', layer=['_BatchNorm', 'GroupNorm'], val=1, bias=0),
    ],
    data_preprocessor=dict(
        type='CustomDataPreprocessor',
        # mean=[124.67  138.17 145.12]      # mean and std computed for Apollo
        # std=[56.87 58.62 62.50]
        mean=[123.675, 116.288, 103.53],    # mean and std of ImageNet
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=1
    ),
    backbone=dict(
        type="ResNet",
        depth=34,
        in_channels=3,
        num_stages=4,
        out_indices=(3,),  # only last stage features
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34'),
    ),
    view_transformer=dict(
        type="ExplicitVoxelProjection",
        z_range=z_range,
        roi_x=x_range,
        roi_y=y_range,
        grid_res=(meter_per_pixel, meter_per_pixel, meter_per_pixel),
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
    # bev_lane_head=dict(
    #     type="BEVLaneHeadVox",
    #     output_size=bev_shape,
    #     input_channels=512,
    #     embed_net_cfg=dict(
    #         type="InstanceEmbeddingWithOffset",
    #         ci=64,
    #         co=2
    #     )
    # ),
    bev_lane_head=dict(
        type="BEVLaneHeadVoxMod",
        output_size=bev_shape,
        input_channels=64,  # 96 for global sptial module exp
        embed_net_cfg=dict(
            type="InstanceEmbeddingWithOffset",
            ci=64,
            co=2
        )
    ),
    loss_seg=dict(
        type='SegLoss',
        bce_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            class_weight=[10.0],
            loss_weight=1.0
        ),
        iou_loss=dict(
            type='SoftIOULoss',
            ignore_index=255,
            eps=1e-6
        )
    ),
    loss_embed=dict(
        type='NDPushPullLoss',
        var_weight=1.0,
        dist_weight=1.0,
        margin_var=1.0,
        margin_dist=5.0,
        ignore_label=200
    ),
    loss_offset=dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        loss_weight=1.0
    ),
    loss_z=dict(
        type='MSELoss',
        reduction='mean',
        loss_weight=1.0
    ),
    # loss weights
    bev_loss_seg_weight=3,
    bev_loss_embed_weight=0.5,
    bev_loss_offset_weight=60,
    bev_loss_z_weight=30,
    fv_loss_seg_weight=3,
    fv_loss_embed_weight=0.5,
    bb_out_ch=512,      # depends on backbone's (resenet) "out_indices" parameter
                        # [64, 128, 256, 512] for out_indices (0, 1, 2, 3)
    # bev_feat_ch=int(64*((2*z_range)/meter_per_pixel)),
    bev_feat_ch=64,
)