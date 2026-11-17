from mmengine.config import read_base

with read_base():
    # from .._base_.datasets.apollosim_synthetic_lanes import *
    from .._base_.datasets.apollosim_synthetic_lanes_vox import *
    # from .._base_.models.bevlanedet_r34 import *
    # from .._base_.models.bevlanedet_r34_vox import *
    from .._base_.models.bevlanedet_r34_vox_upscale import *
    # from .._base_.models.bevlanedet_r34_mod import *
    from .._base_.schedulers.schedule_2x import *
    from .._base_.default_runtime import *

default_scope = 'mmdet'

custom_imports = dict(
    imports=[
        'bevlanedet',
    ],
    allow_failed_imports=False
)

work_dir = './work_dirs_train/bevlanedet_temp'

randomness = dict(
    seed=42, 
    # diff_rank_seed=True,  # ensures different seeds per GPU if doing multi-GPU training
    deterministic=False     # Forces deterministic cuDNN execution algorithms
)

# load_from = r"C:\Users\oo_wa\Downloads\python_scripts\BEVLaneDet2\checkpoints\bevlanedet_r34_vox_illusChg_exp1.pth"

# val_cfg = None
# val_evaluator = None
# val_dataloader = None

# test_cfg = None
test_evaluator = val_evaluator
test_dataloader = val_dataloader

visualizer = dict(
    type='Visualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')],
)

custom_hooks = [
    # dict(
    #     type="mmengine.FeatMapVisHookCustom",
    #     target_layers=[
    #         'backbone.layer3', 'backbone.layer4',
    #         'fv_lane_head.fv_up', 'fv_lane_head.head.neck', 'fv_lane_head.head.ms', 'fv_lane_head.head.me',
    #         'ch_reduce',
    #         'learned_height_compressor',
    #         'spatial_relation_module',
    #         'bev_lane_head.bev_up', 'bev_lane_head.head.neck', 'bev_lane_head.head.ms', 'bev_lane_head.head.moff', 'bev_lane_head.head.mz', 'bev_lane_head.head.me'
    #     ],
    #     # target_layers=['backbone.layer3', 'backbone.layer4' ,'ch_reduce', 'learned_height_compressor', 'spatial_relation_module'],
    #     interval=990,
    #     channel_reduction='squeeze_mean',
    #     alpha=0.5
    # ),
    dict(
        type="mmengine.GradientMonitorHook",
        layer_groups = [
            'backbone.layer1','backbone.layer2','backbone.layer3','backbone.layer4',
            'ch_reduce',
            'view_transformer',
            'learned_height_compressor',
            'downscale',
            'spatial_relation_module',
            'bev_lane_head',
            'fv_lane_head',
        ],
        iter_interval=100,
        epoch_interval=2
    )
]