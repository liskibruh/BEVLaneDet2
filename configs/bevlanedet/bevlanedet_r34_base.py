from mmengine.config import read_base

with read_base():
    # from .._base_.datasets.apollosim_synthetic_lanes import *
    from .._base_.datasets.apollosim_synthetic_lanes_vox import *
    # from .._base_.models.bevlanedet_r34 import *
    from .._base_.models.bevlanedet_r34_vox import *
    from .._base_.schedulers.schedule_2x import *
    from .._base_.default_runtime import *

default_scope = 'mmdet'

custom_imports = dict(
    imports=[
        'bevlanedet',
    ],
    allow_failed_imports=False
)

work_dir = './work_dirs_train/bevlandet_base'

# load_from = "./work_dirs/bevlandet_base/epoch_3.pth"

# val_cfg = None
# val_evaluator = None
# val_dataloader = None

# test_cfg = None
test_evaluator = val_evaluator
test_dataloader = val_dataloader