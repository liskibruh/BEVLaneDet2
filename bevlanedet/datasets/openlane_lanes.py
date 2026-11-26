import json
import copy
import numpy as np
import os
import random
from torch.utils.data import Dataset
from tqdm import tqdm

from mmengine.dataset import BaseDataset
from mmengine.logging import MMLogger
from mmdet.registry import TRANSFORMS, DATASETS

logger = MMLogger.get_instance(name='mmengine', log_level='INFO')
rng = random.Random(42)

class Compose:
    "Compose multiple transforms sequentially"
    def __init__(self, transforms):
        self.transforms = []
        
        if transforms is None:
            transforms = []
            
        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f"transform should be a callable object, "
                                    f"but got {type(transform)}")
                self.transforms.append(transform)
            
            elif callable(transform):
                self.transforms.append(transform)
            
            else:
                raise TypeError(
                    f"transform must be a callable object or dict, "
                    f"but got {type(transform)}"
                )
    
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            
            if data is None:
                return None
        
        return data
    
    def __repr__(self):
        """Print ``self.transforms`` in sequence.
        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

@DATASETS.register_module()
class OpenLaneDataset(Dataset):
    def __init__(self,
                data_root, 
                ann_dir,
                pipeline,
                lazy_init,
                feat_downscale=4):
        
        self._metainfo = {}
        
        self.data_root = data_root
        self.feat_downscale = feat_downscale
        self.ann_dir = os.path.join(data_root, ann_dir)
        self.fully_initialized = False
        
        if not os.path.exists(self.ann_dir):
            raise FileNotFoundError(f"{self.ann_dir} does not exist")
        
        self.pipeline = Compose(pipeline)    
        self.full_init()
    
    def full_init(self):
        if self.fully_initialized:
            return
        
        self.data_list = self.load_data_list()
        self.data_list = self.filter_data()
        
        self.fully_initialized = True
    
    def __len__(self):
        return len(self.data_list)
    
    @property
    def metainfo(self):
        return copy.deepcopy(self._metainfo)
    
    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])
        
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx
        
        return data_info
    
    def load_data_list(self):
        data_infos = []
        
        segments = os.listdir(self.ann_dir)
        
        for i, segment in enumerate(segments):
            segment_pth = os.path.join(self.ann_dir, segment)
            segments_files = os.listdir(segment_pth)
            # segments_files = rng.shuffle(os.listdir(segment_pth))
            
            # count = 0
            for seg_file in tqdm(segments_files, desc=f"Loading JSONs from segment no. {i+1} [{segment}] ", unit="file", total=len(segments_files)):
                
                # if count >= 25:
                #     break
                
                file_pth = os.path.join(segment_pth, seg_file) if seg_file.endswith('.json') else 'invalid'
                
                if file_pth == 'invalid':
                    raise ValueError(f"Expected a .json annotation file, but got {seg_file}")
                
                with open(file_pth, 'r') as infile:
                    info_dict = json.load(infile)
                    raw_file = info_dict['file_path']
                    info_dict['raw_file'] = info_dict.pop('file_path')
                    info_dict['img_path'] = os.path.join(self.data_root, "images", raw_file)
                    
                    data_infos.append(info_dict)
                
                # count += 1
                    
        logger.info(f"Loaded annotations for total of {len(data_infos)} samples")
         
        return data_infos
    
    def filter_data(self):
        # placeholder method for filtering data
        return self.data_list
    
    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)
    
    def __getitem__(self, idx):
        data = self.prepare_data(idx)
        
        if data is None:
            raise ValueError("data cannot be None, check data pipeline.")
        
        return data

# if __name__ == "__main__":
#     dataset = OpenLaneDataset(
#         data_root= r"C:\Users\oo_wa\Downloads\python_scripts\BEVLaneDet2\data\OpenLane",
#         ann_dir = "lane3d_300/training/",
#         lazy_init = False,
#         pipeline = None
#     )
    
#     print(dataset.__getitem__(2))