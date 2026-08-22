import os
import sys
import traceback

from mmengine.config import Config
from mmdet.registry import TRANSFORMS
from bevlanedet.datasets.apollosim_lanes import ApolloSimDataset
from bevlanedet.datasets.transforms import GenerateLaneTargets

CONFIG_PTH = "/data24t_1/owais.tahir/Research/BEVLaneDet2/configs/_base_/dataset/apollosim_synthetic_lanes.py"

def main():
    cfg = Config.fromfile(CONFIG_PTH)
    
    dataset_cfg = cfg.train_dataloader['dataset']
    
    dataset = ApolloSimDataset(
        data_root=dataset_cfg['data_root'],
        ann_file=dataset_cfg['ann_file'],
        pipeline=dataset_cfg['pipeline'],
        lazy_init=dataset_cfg['lazy_init']
    )
    
    sample = dataset.get_data_info(0)
    
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
                # print(f"laneLines (len({len(sample['laneLines'])})): \n\t{sample['laneLines']}")
                # print(f"laneLines_visibility (len({len(sample['laneLines_visibility'])})): \n\t{sample['laneLines_visibility']}")
                print(f"="*64)
            
            if sample is None:
                print(f"Error tranform {name} returned None")
            
        except Exception as e:
            print(f"Error transform {name}")
            traceback.print_exc()
            
            break
    

if __name__ == "__main__":
    main()