import os
import numpy as np
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as plt
import skimage.color as sc

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Args:
        dataroot         (str): Training data set address.
        name             (str): Name of dataset.
        crop _size       (int): High resolution image size.
        upscale_factor   (int): Magnification.
        mode             (str): Data set loading method, the training data set is for data enhancement,
                                and the verification data set is not for data enhancement.
        rot_degree       (int): Rotation angle for image augmentation.
    """

    def __init__(self,
                 dataroot: str,
                 name: str,
                 crop_size: int,
                 mode: str,
                 scale: int,
                 percentage: float = 0.5,
                 rot_degree: int = 180,
                 rgb_range: int = 1) -> None:
        super(BaseDataset, self).__init__()
        self.name = name
        self.mode = mode
        self.crop_size = crop_size
        self.scale = scale
        self.percentage = percentage
        self.rot_degree = rot_degree
        self.rgb_range = rgb_range

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self._get_index(index)
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        lr = Image.open(self.lr_files[idx]).convert("RGB")

        lr, hr = self.transforms(lr, hr)

        return {"lr":lr, "hr":hr}

    def __len__(self) -> int:
        return len(self.hr_files)

    @staticmethod
    def _get_index(idx):
        return idx
