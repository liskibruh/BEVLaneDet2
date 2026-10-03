from mmengine.config import read_base

with read_base():
    from ..commons import *

data_root = "./data/Apollo_Sim_3D_Lane_Release/"
train_ann_path = "data_splits/standard/train.json"
val_ann_path = "data_splits/standard/test.json"

dataset = "ApolloSimDataset"

# virtual camera
vc_config = dict(
    use_virtual_cam=True,
    vc_height=1.7860000133514404,      # same as BEVLaneDet
    vc_pitch=0.04325083977888603,
    vc_im_shape=(1920, 1080)           # todo: use consistent axes order
)

albu_train_transforms = [
    dict(type="MotionBlur", p=0.2),
    dict(type="RandomBrightnessContrast"),
    dict(type="ColorJitter", p=0.1)
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="GenerateLaneTargets",
        x_range=x_range,
        y_range=y_range,
        meter_per_pixel=meter_per_pixel,
        vc_config=vc_config
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
    dict(type="PackLaneData")
    # todo: add normlization in data_preprocessor in model config
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="GenerateLaneTargets",     # todo: this warps the image in vc and also generates gts, the gts part is not required for evaluation
        x_range=x_range,
        y_range=y_range,
        meter_per_pixel=meter_per_pixel,
        vc_config=vc_config
        ),
    dict(type="Resize", scale=(input_shape[1], input_shape[0]), keep_ratio=True),
    dict(type="PackLaneData")
    # dict(type="RandomFlip", prob=0.5),
    
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    shuffle=True,
    dataset=dict(
        type=dataset,
        data_root=data_root,
        ann_file=train_ann_path,
        pipeline=train_pipeline,
        lazy_init=False # todo: dummy for now
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    shuffle=True,
    dataset=dict(
        type=dataset,
        data_root=data_root,
        ann_file=val_ann_path,
        pipeline=val_pipeline,
        lazy_init=False # todo: dummy for now
    )
)

val_evaluator = dict(
    type="DummyEval",
    post_conf=0.9,
    post_emb_margin=6.0,
    post_min_cluster_size=15,
    x_range=x_range,
    meter_per_pixel=meter_per_pixel,
)
