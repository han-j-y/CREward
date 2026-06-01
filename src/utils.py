import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Dict
import glob
import numpy as np



def make_dir(path):
    if os.path.isdir(path)==False:
        os.mkdir(path)


def param_size_BM(model: torch.nn.Module, trainable_only: bool = False) -> Dict[str, float]:
    """
    Return parameter counts in billions (B) and millions (M).
    Also reports parameter memory usage in bytes, MB, and GB.

    Args:
        model: torch.nn.Module
        trainable_only: If True, count only parameters with requires_grad=True.

    Returns:
        {
          "B": float,          # parameter count in billions
          "M": float,          # parameter count in millions
          "params": int,       # raw parameter count
          "bytes": int,        # total parameter memory in bytes (by dtype)
          "MB": float,         # bytes / 1024^2
          "GB": float          # bytes / 1024^3
        }
    """
    if trainable_only:
        params = list(p for p in model.parameters() if p.requires_grad)
    else:
        params = list(model.parameters())

    # Total parameter count
    total_params = sum(p.numel() for p in params)

    # Sum element_size per tensor to handle mixed dtypes
    total_bytes = sum(p.numel() * p.element_size() for p in params)

    return {
        "B": total_params / 1e9,
        "M": total_params / 1e6,
        "params": total_params,
        "bytes": total_bytes,
        "MB": total_bytes / (1024 ** 2),
        "GB": total_bytes / (1024 ** 3),
    }

def dataset_preprocess(obj_list, base_path='./', 
                       ranking_base='gemma27b_results',
                       ratios=[800, 100, 100]):
    
    if ratios == None: # for human evaluation
        pair1_list = []
        pair2_list = []
        ranking_list = torch.tensor([]).view(5,0,4).type(torch.long)
        for obj in obj_list:
            pair1_list += sorted(glob.glob(f'{base_path}/{obj}/*_0.png'))
            pair2_list += sorted(glob.glob(f'{base_path}/{obj}/*_1.png'))
            if ranking_base != None:
                ranking_list = torch.cat((ranking_list,
                                          torch.tensor(np.load(f'{base_path}/{ranking_base}_{obj}.npy'), 
                                                       dtype=torch.long)), dim=1)
    else:
        pair1_list = {'train': [], 'valid': [], 'test': []}
        pair2_list = {'train': [], 'valid': [], 'test' : []}
        ranking_list = {'train': torch.tensor([]).view(0,4).type(torch.long),
                        'valid': torch.tensor([]).view(0,4).type(torch.long), 
                        'test': torch.tensor([]).view(0,4).type(torch.long)}
        for obj in obj_list:
            x1_list = sorted(glob.glob(f'{base_path}/{obj}/*_0.png'))
            x2_list = sorted(glob.glob(f'{base_path}/{obj}/*_1.png'))
            obj_ranking = torch.tensor(np.load(f'{base_path}/{ranking_base}_{obj}_mass.npy'), dtype=torch.long)
            
            curr = 0
            for type_, ratio in zip(['train', 'valid', 'test'], ratios):
                pair1_list[type_] += x1_list[curr:curr+ratio]
                pair2_list[type_] += x2_list[curr:curr+ratio]
                if ranking_base != None:
                    ranking_list[type_] = torch.cat([ranking_list[type_],
                                                 obj_ranking[curr:curr+ratio]], dim=0)
                curr = curr+ratio
                
    return pair1_list, pair2_list, ranking_list
        
class CreRewardDataset(Dataset):
    def __init__(self, pair1_list, pair2_list, ranking_list):
        super(CreRewardDataset, self).__init__()
        self.pair1_list = pair1_list
        self.pair2_list = pair2_list
        self.randking_list = ranking_list
        self.transform = transforms.ToTensor() # PIL image -> torch tensor and normalize


    def __getitem__(self, index):
        pair1_image = self.transform(Image.open(self.pair1_list[index]))
        pair2_image = self.transform(Image.open(self.pair2_list[index]))
        ranking_info = self.randking_list[index]

        return pair1_image, pair2_image, ranking_info

    def __len__(self):
        return len(self.randking_list)