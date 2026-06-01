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
    모델 파라미터 개수를 B(십억), M(백만) 단위로 반환.
    추가로 파라미터 메모리 사용량도 bytes/MB/GB로 함께 제공.

    Args:
        model: torch.nn.Module
        trainable_only: True면 requires_grad=True인 파라미터만 집계

    Returns:
        {
          "B": float,          # 파라미터 개수(십억 단위)
          "M": float,          # 파라미터 개수(백만 단위)
          "params": int,       # 파라미터 개수(정수)
          "bytes": int,        # 파라미터 메모리 총 바이트 수 (dtype 반영)
          "MB": float,         # bytes를 1024^2로 나눈 값
          "GB": float          # bytes를 1024^3로 나눈 값
        }
    """
    if trainable_only:
        params = list(p for p in model.parameters() if p.requires_grad)
    else:
        params = list(model.parameters())

    # 총 파라미터 개수
    total_params = sum(p.numel() for p in params)

    # dtype이 다를 수 있으므로 개별 element_size를 곱해 총 바이트 수 계산
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