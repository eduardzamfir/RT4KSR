import os
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from utils.image import modcrop_tensor
from data import transforms
from data.basedataset import BaseDataset


class Benchmark(BaseDataset):
    def __init__(self,
                 dataroot: str,
                 name: str,
                 mode: str,
                 scale: int,
                 crop_size: int = 64,
                 rgb_range: int = 1) -> None:
        super(Benchmark, self).__init__(dataroot=dataroot,
                                        name=name,
                                        mode=mode,
                                        scale=scale,
                                        crop_size=crop_size,
                                        rgb_range=rgb_range)
        
        self.lr_dir_path = os.path.join(dataroot, "testsets", self.name, self.mode, f"LR_bicubic_x{self.scale}")
        self.hr_dir_path = os.path.join(dataroot, "testsets", self.name, self.mode, "HR")
        
        self.lr_files = [os.path.join(self.lr_dir_path, x) for x in sorted(os.listdir(self.lr_dir_path))]
        self.hr_files = [os.path.join(self.hr_dir_path, x) for x in sorted(os.listdir(self.hr_dir_path))]
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(rgb_range=self.rgb_range)
        ])
        self.degrade = transforms.BicubicDownsample(scale)
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self._get_index(index)
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        lr = Image.open(self.lr_files[idx]).convert("RGB")

        lr = self.transforms(lr)
        hr = self.transforms(hr)
        
        # assert that images are divisable by 2
        c, lr_h, lr_w = lr.shape
        lr_hr, lr_wr = int(lr_h/2), int(lr_w/2)
        lr = lr[:, :lr_hr*2, :lr_wr*2]
        hr = hr[:, :lr.shape[-2] * self.scale, :lr.shape[-1] * self.scale]
        
        assert lr.shape[-1] * self.scale == hr.shape[-1]
        assert lr.shape[-2] * self.scale == hr.shape[-2]

        return {"lr":lr.to(torch.float32), "hr":hr.to(torch.float32)}


def set5(config):
    return Benchmark(config.dataroot, "Set5", mode="val", scale=config.scale, rgb_range=config.rgb_range)


def set14(config):
    return Benchmark(config.dataroot, "Set14", mode="val", scale=config.scale, rgb_range=config.rgb_range)


def b100(config):
    return Benchmark(config.dataroot, "B100", mode="val", scale=config.scale, rgb_range=config.rgb_range)


def urban100(config):
    return Benchmark(config.dataroot, "Urban100", mode="val", scale=config.scale, rgb_range=config.rgb_range)
