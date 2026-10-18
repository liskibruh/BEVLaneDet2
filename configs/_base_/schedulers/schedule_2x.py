# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim.sgd import SGD

# training schedule for 1x
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=50, val_interval=10)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

train_bs = 4        # must match train_dataloader batch_size (single GPU, so no multiplier)
orig_bs = 8         # <- original implementation's batch size
base_lr = 1e-3 * train_bs / orig_bs   # 5e-4 with these values

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, betas=(0.9, 0.999),
                   eps=1e-8, weight_decay=1e-2, amsgrad=False))

param_scheduler = [
    # short warmup; drop it for a strict reproduction
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=50, eta_min=0,
         begin=0, end=50, by_epoch=True),
]

auto_scale_lr = dict(enable=False, base_batch_size=orig_bs)
