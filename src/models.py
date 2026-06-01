import numpy as np
import glob
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoModel, CLIPModel
from PIL import Image
from typing import Dict
from torchvision.transforms.functional import to_tensor
from dreamsim import dreamsim

class CreativityReward(nn.Module):
    def __init__(self, device='cpu', backbone='siglip-gemma3', 
                 dtype=torch.float32, requires_grad_=False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.backbone = backbone

        if backbone == 'siglip-gemma3':
            MODEL_ID = "google/gemma-3-4b-it"
            model = Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,
                                                       device_map=self.device,
                                                       torch_dtype=self.dtype).eval()
            self.vision_backbone = model.vision_tower
            self.vision_projector = model.multi_modal_projector
            self.image_processor = transforms.Compose([
                                        transforms.Resize((896, 896)),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                                             std=[0.5, 0.5, 0.5]),  # [-1,1]
                                    ])    
            self.input_f_size = 2560

        if backbone == 'siglip2':
            MODEL_ID = "google/siglip2-so400m-patch14-384"
            model = AutoModel.from_pretrained(MODEL_ID,
                                              device_map=self.device,
                                              torch_dtype=self.dtype).eval()
            self.vision_backbone = model.vision_model
            self.vision_projector = nn.Identity()
            self.image_processor = transforms.Compose([
                                        transforms.Resize((384, 384)),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                                             std=[0.5, 0.5, 0.5]),  # [-1,1]
                                    ])    
            self.input_f_size = 1152

        elif backbone == 'vgg16':
            model = torchvision.models.vgg16(pretrained=True)
            model.classifier = model.classifier[:-1]
            
            self.vision_backbone = model.to(self.device).eval()
            self.vision_projector = nn.Identity()

            self.vision_backbone.requires_grad_(False)
            self.vision_projector.requires_grad_(False)    
            
            self.image_processor = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                ])

            self.input_f_size = 4096

        elif backbone == 'clip':
            MODEL_ID = "openai/clip-vit-large-patch14"
            model = CLIPModel.from_pretrained(MODEL_ID)
            
            self.vision_backbone = model.vision_model.to(self.device).eval()
            self.vision_projector = nn.Identity()
            self.image_processor = transforms.Compose([
                                                        transforms.Resize((224,224)),
                                                        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                                                std=[0.26862954, 0.26130258, 0.27577711])
                                                    ])
            self.input_f_size = 1024

        elif backbone == 'dreamsim':
            model, _ = dreamsim(pretrained=True, device=self.device)
            self.vision_backbone = model.eval()
            self.vision_projector = nn.Identity()
            self.image_processor = transforms.Compose([
                                                        transforms.Resize((224,224), 
                                                                          interpolation=transforms.InterpolationMode.BICUBIC),
                                                        ])

            self.input_f_size = 1792

        elif backbone == 'dino3':
            MODEL_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
            model = AutoModel.from_pretrained(MODEL_ID, dtype=self.dtype).to(self.device).eval()
            self.vision_backbone = model
            self.vision_projector = nn.Identity()
            self.image_processor = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                 std=[0.229, 0.224, 0.225]),
                        ])
            self.input_f_size = 1280

        #------------------------------------#
        self.vision_backbone.requires_grad_(requires_grad_)
        self.vision_projector.requires_grad_(requires_grad_)
        del model

        #------------------------------------#
        self.reward_head = nn.Sequential(
            nn.Linear(self.input_f_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        ).to(self.device)
            
    def vision_forward(self, x):
        x = self.image_processor(x)
        if self.backbone == 'siglip-gemma3':
            h = self.vision_backbone(pixel_values=x, return_dict=True)['last_hidden_state']
            h = self.vision_projector(h)
            h = h.mean(dim=1) # [2B x # token x D] -> [2B x D]
            
        elif self.backbone == 'siglip2':
            h = self.vision_backbone(pixel_values=x, return_dict=True)['pooler_output'] # [2B x D]            
            
        elif self.backbone =='vgg16':
            h = self.vision_backbone(x)
            
        elif self.backbone =='clip':
            h = self.vision_backbone(x)['pooler_output']

        elif self.backbone == 'dreamsim':
            h = self.vision_backbone.embed(x)

        elif self.backbone == 'dino3':
            h = self.vision_backbone(x).last_hidden_state[:, 0, :]
            
        return h
        
    def vision_forward_(self, x):
        x = self.image_processor(x).to(self.device, dtype=self.dtype)
        h = self.vision_backbone(x)
        h = self.vision_projector(h)
        return h
        
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=0)
        h = self.vision_forward(x) 
        reward = self.reward_head(h) # [2B x 4]
        x1_reward, x2_reward = reward.chunk(2) # 2 x [B x 4]
        return torch.stack([x1_reward, x2_reward], dim=1) # [B x 2 x 4]

    def compute_reward(self, x):
        if not torch.is_tensor(x):
            x = to_tensor(x).unsqueeze(0).to(self.device)
        h = self.vision_forward(x)
        reward = self.reward_head(h) # [B x 4]
        return reward

    def compute_reward_lora(self, x):
        if not torch.is_tensor(x):
            x = to_tensor(x).unsqueeze(0).to(self.device)
        h = self.vision_forward(x)
        reward = self.reward_head(h) # [B x 4]
        return reward, h