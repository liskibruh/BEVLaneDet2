data_root = "./data/Apollo_Sim_3D_Lane_Release/"
train_ann_path = "data_splits/standard/train.json"
val_ann_path = "data_splits/standard/val.json"

dataset = "ApolloDataset"

input_shape = (576,1024)
output_2d_shape = (144, 256)

x_range = (3, 103)
y_range = (-12, 12)
meter_per_pixel = 0.5

bev_x = int((x_range[1] - x_range[0]) / meter_per_pixel)
bev_y = int((y_range[1] - y_range[0]) / meter_per_pixel)
bev_shape = (bev_x, bev_y)

# virtual camera
vc_config = dict(
    use_virtual_camera=True,
    vc_height=0.04325083977888603,      # same as BEVLaneDet
    vc_pitch=1.7860000133514404
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="GenerateLaneTargets",
        x_range=x_range,
        y_range=y_range,
        meter_per_pixel=meter_per_pixel,
        vc_config=vc_config
        ),
    # dict(type="Resize", scale=input_shape, keep_ratio=True),
    # dict(type="RandomFlip", prob=0.5),
    # dict(type="Pad", size_divisor=32),
    # dict(type="PackDetInputs")
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="Resize", scale=input_shape, keep_ratio=True),
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