import os
import numpy as np
from PIL import Image
from typing import Tuple
import matplotlib.pyplot as plt

import torch
from torch.utils import data

from data import transforms

   
class DIV2K(data.Dataset):
    def __init__(self,
                 name: str,
                 dataroot: str,
                 crop_size: int,
                 mode: str,
                 scale: int,
                 rgb_range: int = 1):
        super(DIV2K, self).__init__()
        self.name = name
        self.dataroot = dataroot
        self.crop_size = crop_size
        self.mode = mode
        self.scale = scale
        self.rgb_range = rgb_range
        self.min_size = scale * crop_size

        self.paths_H = []
        self.paths_L = []

        if mode == "train":
            for x in sorted(os.listdir(os.path.join(dataroot, "DIV2K", "DIV2K_train_HR_sub"))):
                self.paths_H.append(os.path.join(os.path.join(dataroot, "DIV2K", "DIV2K_train_HR_sub"), x))
            self.transforms = transforms.Compose([
                    transforms.RandomCrop(crop_size=crop_size, upscale_factor=scale),
                    transforms.RandomHFlip(),
                    transforms.RandomVFlip(),
                    transforms.RandomRotation(),
                    transforms.ToTensor(rgb_range=rgb_range)
                ])
        elif mode == "valid":
            for x in sorted(os.listdir(os.path.join(dataroot, "DIV2K", "DIV2K_valid_HR"))):
                self.paths_H.append(os.path.join(os.path.join(dataroot, "DIV2K", "DIV2K_valid_HR"), x))  
                
            self.transforms = transforms.Compose([
                transforms.CenterCrop(self.crop_size, self.scale),
                transforms.ToTensor(rgb_range=rgb_range)
            ])
        
        self.degrade = transforms.BicubicDownsample(scale)
        
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.paths_H[idx]
        img = Image.open(img_path).convert("RGB")

        # apply geometric transforms
        img = self.transforms(img)
        
        # apply bicubic degradation
        lr, hr = self.degrade(img)

        return {"lr":lr, "hr":hr}

    def __len__(self):
        return len(self.paths_H)


def div2k(config, mode):
    return DIV2K(dataroot=config.dataroot,
                 name="DIV2K",
                 mode=mode,
                 scale=config.scale,
                 crop_size=config.crop_size,
                 rgb_range=config.rgb_range)
