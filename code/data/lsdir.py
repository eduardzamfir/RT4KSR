import os
import torch
import torch.utils.data as data

from PIL import Image
from typing import Tuple

import data.transforms as transforms



class LSDIR(data.Dataset):
    def __init__(self,
                 name: str,
                 dataroot: str,
                 crop_size: int,
                 mode: str,
                 scale: int,
                 rgb_range: int = 1):
        super(LSDIR, self).__init__()
        self.name = name
        self.dataroot = dataroot
        self.crop_size = crop_size
        self.mode = mode
        self.scale = scale
        self.rgb_range = rgb_range

        path = os.path.join(dataroot, "LSDIR")
        self.paths_H = []
        for _dir in sorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path, _dir)):
                for x in sorted(os.listdir(os.path.join(path, _dir))):
                    self.paths_H.append(os.path.join(path, _dir, x))

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
    
    
class LSDIR_sub(data.Dataset):
    def __init__(self,
                 name: str,
                 dataroot: str,
                 crop_size: int,
                 mode: str,
                 scale: int,
                 rgb_range: int = 1) -> None:
        super().__init__()
        self.name = name
        self.dataroot = dataroot
        self.crop_size = crop_size
        self.mode = mode
        self.scale = scale
        self.rgb_range = rgb_range
        
        path = os.path.join(dataroot, "LSDIR", "sub")
        self.paths_H = []
        for _dir in sorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path, _dir)):
                for x in sorted(os.listdir(os.path.join(path, _dir))):
                    self.paths_H.append(os.path.join(path, _dir, x))
            
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


def lsdir(config, mode):
    return LSDIR(dataroot=config.dataroot,
                 name="LSDIR",
                 mode=mode,
                 scale=config.scale,
                 crop_size=config.crop_size,
                 rgb_range=config.rgb_range)

    
def lsdir_sub(config, mode):
    return LSDIR_sub(dataroot=config.dataroot,
                     name="LSDIR_sub",
                     mode=mode,
                     scale=config.scale,
                     crop_size=config.crop_size,
                     rgb_range=config.rgb_range)