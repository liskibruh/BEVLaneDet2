from mmengine.config import read_base

with read_base():
    from .._base_.datasets.apollosim_synthetic_lanes import *
    from .._base_.models.bevlanedet_r34 import *
    from .._base_.schedulers.schedule_2x import *
    from .._base_.default_runtime import *


work_dir = './work_dirs/bevlandet_base'