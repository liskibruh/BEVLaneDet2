import json
import copy
import sys
from os import path as osp
from torch.utils.data import Dataset

from mmengine.dataset import BaseDataset
from mmdet.registry import TRANSFORMS, DATASETS


# @DATASETS.register_module()
# class ApolloSimDataset(BaseDataset):
    
#     METAINFO = {}
    
#     def load_data_list(self):
#         data_list = []
        
#         ann_file = osp.join(self.data_root, self.ann_file)
        
#         if not osp.exists(ann_file):
#             raise FileNotFoundError(f'{ann_file} does not exist')
        
#         with open(ann_file, 'r') as f:
#             for line in f:
#                 info = json.loads(line)
                
#                 info['img_path'] = osp.join(
#                     self.data_root,
#                     info['raw_file']
#                 )
                
#                 data_list.append(info)
        
#         return data_list
    
#     def filter_data(self):
#         return self.data_list
    
#     def get_data_info(self, idx):
#         data_info = copy.deepcopy(self.data_list[idx])
#         data_info['sample_idx'] = idx
#         return data_info


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
class ApolloSimDataset(Dataset):
    def __init__(self,
                data_root, 
                ann_file,
                pipeline,
                lazy_init):
        
        self._metainfo = {}
        self.data_root = data_root
        self.ann_file = osp.join(data_root, ann_file)
        
        if not osp.exists(self.ann_file):
            raise FileNotFoundError(f"{self.ann_file} does not exist")
        
        self.pipeline = Compose(pipeline)    
        self.full_init()
    
    def full_init(self):
        self.data_list = self.load_data_list()
        self.data_list = self.filter_data()
    
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
        
        with open(self.ann_file, 'r') as infile:
            for line in infile:
                info_dict = json.loads(line)
                raw_file = info_dict['raw_file']
                info_dict["img_path"] = osp.join(self.data_root, raw_file)
                # print(f"info_dict.keys(): {info_dict.keys()}")
                data_infos.append(info_dict)
        
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

if __name__ == "__main__":
    dataset = ApolloSimDataset(
        data_root= "/data24t_1/owais.tahir/Research/BEVLaneDet2/data/Apollo_Sim_3D_Lane_Release",
        ann_file = "data_splits/standard/train.json",
        lazy_init = False,
        pipeline = None
    )
    
    print(dataset.__getitem__(2))