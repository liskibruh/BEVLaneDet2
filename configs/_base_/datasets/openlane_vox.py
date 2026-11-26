from mmengine.config import read_base

with read_base():
    from ..commons import *

data_root = "./data/OpenLane/"
train_ann_path = "lane3d_300/training/"
val_ann_path = "lane3d_300/validation/"

dataset = "OpenLaneDataset"


vc_config = dict(
    use_virtual_cam=True,
    vc_intrinsic=[
        [
            [2081.5212033927246, 0.0, 934.7111248349433],
            [0.0, 2081.5212033927246, 646.3389987785433],
            [0.0, 0.0, 1.0]
        ]
    ],
    vc_extrinsic=[
        [-0.002122161262459438, 0.010697496358766389, 0.9999405282331697, 1.5441039498273286],
        [-0.9999378331046326, -0.010968621415360667, -0.0020048117763292747, -0.023774034344867204],
        [0.010946522625388108, -0.9998826195688676, 0.01072010851209982, 2.1157397903843567],
        [0.0, 0.0, 0.0, 1.0]
    ],
    vc_im_shape=(1920, 1280)
)

# # virtual camera
# vc_config = dict(
#     use_virtual_cam=True,
#     vc_height=1.7860000133514404,      # same as BEVLaneDet
#     vc_pitch=0.04325083977888603,
#     vc_im_shape=(1920, 1080)           # todo: use consistent axes order
# )

albu_train_transforms = [
    dict(type="MotionBlur", p=0.2),
    dict(type="RandomBrightnessContrast"),
    dict(type="ColorJitter", p=0.1)
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="GenerateOpenLaneTargets",
        x_range=x_range,
        y_range=y_range,
        meter_per_pixel=meter_per_pixel,
        vc_config=vc_config
        ),
    dict(
        type="Voxel_Generator_Cont", 
        base_height=base_height,
        z_range=z_range,
        roi_x=x_range,
        roi_y=y_range,
        grid_res=(meter_per_pixel, meter_per_pixel, meter_per_pixel),
        input_shape=input_shape,
        feat_downscale=feat_downscale,
        ),
    dict(
        type="MultiTargetResize",
        image_size=input_shape,
        image_gt_size=output_2d_shape
        ),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        keymap={'img': 'image'}
        ),
    dict(type="PackLaneDataVox")
    # todo: add normlization in data_preprocessor in model config
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="GenerateOpenLaneTargets",     # todo: this warps the image in vc and also generates gts, the gts part is not required for evaluation
        x_range=x_range,
        y_range=y_range,
        meter_per_pixel=meter_per_pixel,
        vc_config=vc_config
        ),
    dict(
        type="Voxel_Generator_Cont", 
        base_height=base_height,
        z_range=z_range,
        roi_x=x_range,
        roi_y=y_range,
        grid_res=(meter_per_pixel, meter_per_pixel, meter_per_pixel),
        input_shape=input_shape,
        feat_downscale=feat_downscale,
        ),
    dict(type="Resize", scale=(input_shape[1], input_shape[0]), keep_ratio=False),
    dict(type="PackLaneDataVox")
    # dict(type="RandomFlip", prob=0.5),
    
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    shuffle=True,
    dataset=dict(
        type=dataset,
        data_root=data_root,
        ann_dir=train_ann_path,
        pipeline=train_pipeline,
        lazy_init=False # todo: dummy for now
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    shuffle=False,
    dataset=dict(
        type=dataset,
        data_root=data_root,
        ann_dir=val_ann_path,
        pipeline=val_pipeline,
        lazy_init=False # todo: dummy for now
    )
)

val_evaluator = dict(
    type="LaneEvalVis",
    post_conf=-0.7,
    post_emb_margin=6.0,
    post_min_cluster_size=15,
    x_range=x_range,
    meter_per_pixel=meter_per_pixel,
    data_root=data_root,
    visualize=True,
    vis_max_num=0,
)
