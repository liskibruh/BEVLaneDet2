import os
import sys
import numpy as np
import traceback

import torch

from mmengine.config import Config
from mmdet.registry import TRANSFORMS, MODELS
from mmdet.utils import register_all_modules

from bevlanedet.datasets.apollosim_lanes import ApolloSimDataset
from bevlanedet.datasets.transforms import GenerateLaneTargets

register_all_modules()

CONFIG_PTH = "/data8t_1/owais_tahir/BEVLaneDet2/configs/bevlanedet/bevlanedet_r34_base.py"

def main():
    cfg = Config.fromfile(CONFIG_PTH)
    
    dataset_cfg = cfg.train_dataloader['dataset']
    
    dataset = ApolloSimDataset(
        data_root=dataset_cfg['data_root'],
        ann_file=dataset_cfg['ann_file'],
        pipeline=dataset_cfg['pipeline'],
        lazy_init=dataset_cfg['lazy_init']
    )
    
    model_cfg = cfg.model
    
    model = MODELS.build(model_cfg)
    model.train()
    
    sample = dataset.get_data_info(60)   # 1
    
    pipeline_cfg = dataset_cfg['pipeline']
    
    transforms = []
    for i, transform_cfg in enumerate(pipeline_cfg):
        transform = TRANSFORMS.build(transform_cfg)
        transforms.append(transform)
        
    for i, (transform_cfg, transform) in enumerate(zip(pipeline_cfg, transforms)):
        name = transform_cfg['type']
        
        try:
            sample = transform(sample)
            
            if isinstance(sample, dict):
                print(f"After {name}\n {sample.keys()}")
                for key in ['img', 'img_shape', 'ori_shape', 'image_gt', 'bev_gt', 'bev_y_offset_map', 'bev_z_map']:
                    if key in sample.keys():
                        if not isinstance(sample[key], np.ndarray):
                            print(f"{key}: {sample[key]}")
                        else:
                            print(f"{key}: {sample[key].shape}")
                print(f"="*64)
            
            if sample is None:
                print(f"Error tranform {name} returned None")
            
        except Exception as e:
            print(f"Error transform {name}")
            traceback.print_exc()
            
            break
    
    data = dict(        
        inputs = sample['inputs'].unsqueeze(0),
        data_samples = [sample['data_samples']]
    )
    
    data = model.data_preprocessor(data)
    
    fv_out, bev_out = model._forward(**data)
    
    fv_outs_map = {0: 'ms', 1: 'me'}
    bev_outs_map = {0: 'ms', 1: 'me', 2: 'moff', 3: 'mz'}
    
    print("="*64)
    for i, out in enumerate(fv_out):
        print(f"fv_{fv_outs_map[i]}.shape: {out.shape}, device: {out.device}")
    print("="*64)
    for i, out in enumerate(bev_out):
        print(f"bev_{bev_outs_map[i]}.shape: {out.shape}, device: {out.device}")
    print("="*64)
    print("="*64)

if __name__ == "__main__":
    main()