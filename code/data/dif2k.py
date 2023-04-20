import os
import torch
import torch.utils.data as data

from PIL import Image
from typing import Tuple

import data.transforms as transforms



class DIF2K(data.Dataset):
    def __init__(self,
                 name: str,
                 dataroot: str,
                 crop_size: int,
                 mode: str,
                 scale: int,
                 rgb_range: int = 1):
        super(DIF2K, self).__init__()
        self.name = name
        self.dataroot = dataroot
        self.crop_size = crop_size
        self.mode = mode
        self.scale = scale
        self.rgb_range = rgb_range
        
        # combine DIV2K and Flickr2K
        flickr_path = os.path.join(dataroot, "Flickr2K", "Flickr2K_HR")
        div2k_path = os.path.join(dataroot, "DIV2K", "DIV2K_train_HR")
        self.paths_H = []
        for path in [div2k_path, flickr_path]:
            for x in sorted(os.listdir(path)):
                self.paths_H.append(os.path.join(path, x))

        if mode == "train":
            self.transforms = transforms.Compose([
                    transforms.RandomCrop(crop_size=crop_size, upscale_factor=scale),
                    transforms.RandomHFlip(),
                    transforms.RandomVFlip(),
                    transforms.RandomRotation(),
                    transforms.ToTensor(rgb_range=rgb_range)
                ])
        
            self.degrade = transforms.BicubicDownsample(scale)
        
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths_H[idx]).convert("RGB")

        # apply geometric transforms
        img = self.transforms(img)
        
        # apply bicubic degradation
        lr, hr = self.degrade(img)

        return {"lr":lr, "hr":hr}

    def __len__(self):
        return len(self.paths_H)
    
    
class DIF2K_sub(data.Dataset):
    def __init__(self,
                 name: str,
                 dataroot: str,
                 crop_size: int,
                 mode: str,
                 scale: int,
                 extracted_crop_size: int,
                 rgb_range: int = 1) -> None:
        super().__init__()
        self.name = name
        self.dataroot = dataroot
        self.crop_size = crop_size
        self.mode = mode
        self.scale = scale
        self.rgb_range = rgb_range
        
        if extracted_crop_size == 720:
            sub = "larger_sub"
        elif extracted_crop_size == 512:
            sub = "sub"
        
        path = os.path.join(dataroot, "DIV2K_Flickr2K", sub, "train_HR")
        self.paths_H = []
        for x in sorted(os.listdir(path)):
            self.paths_H.append(os.path.join(path, x))
            
        if mode == "train":
            self.transforms = transforms.Compose([
                    transforms.RandomCrop(crop_size=crop_size, upscale_factor=scale),
                    transforms.RandomHFlip(),
                    transforms.RandomVFlip(),
                    transforms.RandomRotation(),
                    transforms.ToTensor(rgb_range=rgb_range)
                ])
        
            self.degrade = transforms.BicubicDownsample(scale)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths_H[idx]).convert("RGB")

        # apply geometric transforms
        img = self.transforms(img)
        
        # apply bicubic degradation
        lr, hr = self.degrade(img)

        return {"lr":lr, "hr":hr}

    def __len__(self):
        return len(self.paths_H)


def dif2k(config, mode):
    return DIF2K(dataroot=config.dataroot,
                 name="DIF2K",
                 mode=mode,
                 scale=config.scale,
                 crop_size=config.crop_size,
                 rgb_range=config.rgb_range)

    
def dif2k_sub(config, mode):
    return DIF2K_sub(dataroot=config.dataroot,
                     name="DIF2K_sub",
                     mode=mode,
                     scale=config.scale,
                     crop_size=config.crop_size,
                     extracted_crop_size=config.extracted_crop_size,
                     rgb_range=config.rgb_range)