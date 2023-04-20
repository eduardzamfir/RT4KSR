import random
import numpy as np
from PIL import Image
from typing import Tuple, List, Callable
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as F


from utils import image


class Compose:
    def __init__(self,
                 transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self,
                 img: Image.Image):
        for transform in self.transforms:
            img = transform(img)
            
        return img


class ToTensor:
    def __init__(self,
                 rgb_range: int = 1):
        self.rgb_range = rgb_range

    def __call__(self,
                 img: Image.Image):
        if self.rgb_range != 1:
            img = F.pil_to_tensor(img).float()
        else:
            img = F.to_tensor(np.array(img) / 255.0)

        return img


class Normalize:
    def __init__(self,
                 mean: List[float] = (0.4488, 0.4371, 0.4040),
                 std: List[float] = (1.0, 1.0, 1.0),
                 rgb_range: int = 1):
        self.mean = mean
        self.std = std
        self.rgb_range = rgb_range
        self.mean_shift = self.rgb_range * torch.Tensor(self.mean) / torch.Tensor(self.std)
        self.norm = tf.Normalize(mean=mean, std=std)

    def __call__(self,
                 img: torch.Tensor,
                 gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rgb_range != 1:
            return img, gt
        else:
            img = self.norm(img)
            return img, gt


class UnNormalize:
    def __init__(self,
                 mean: List[float] = (0.4488, 0.4371, 0.4040),
                 std: List[float] = (1.0, 1.0, 1.0),
                 rgb_range: int = 1):
        self.mean = mean
        self.std = std
        self.rgb_range = rgb_range
        self.mean_shift = self.rgb_range * torch.Tensor(self.mean) / torch.Tensor(self.std)
        self.inv = tf.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )

    def __call__(self,
                 img: torch.Tensor) -> torch.Tensor:
        if self.rgb_range != 1:
            img += self.mean_shift.view(3, 1, 1)
        else:
            img = self.inv(img)
        return img


class RandomHFlip:
    def __init__(self,
                 percentage: float = 0.5):
        self.percentage = percentage

    def __call__(self,
                 img: Image.Image):
        if random.random() < self.percentage:
            img = F.hflip(img)

        return img


class RandomVFlip:
    def __init__(self,
                 percentage: float = 0.5):
        self.percentage = percentage

    def __call__(self,
                 img: Image.Image):
        if random.random() < self.percentage:
            img = F.vflip(img)

        return img


class CenterCrop:
    def __init__(self,
                 crop_size: int,
                 scale: int):
        self.crop_size = crop_size
        self.scale = scale

    def __call__(self,
                 img: Image.Image):
        img = F.center_crop(img, (self.crop_size * self.scale, self.crop_size * self.scale))

        return img


class RandomRotation:
    def __init__(self,
                 percentage: float = 0.5,
                 angle: List = [90, 180, 270]):
        self.percentage = percentage
        self.angles = angle

    def __call__(self,
                 img: Image.Image):
        if isinstance(self.angles, List):
            angle = random.choice(self.angles)
        else:
            angle = self.angles

        if random.random() < self.percentage:
            img = F.rotate(img, angle, expand=True, fill=0)

        return img


class RandomCrop:
    def __init__(self,
                 crop_size: int,
                 upscale_factor: int):
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.crop = tf.RandomCrop(crop_size)
        
    def __call__(self,
                 img: Image.Image):
        i, j, h, w = self.crop.get_params(img=img, 
                                          output_size=(self.crop_size * self.upscale_factor, self.crop_size * self.upscale_factor))
        img = F.crop(img, i, j, h, w)

        return img


class BicubicDownsample:
    def __init__(self, scale):
        self.scale = scale
        
    def __call__(self, img: torch.Tensor):
        assert isinstance(img, torch.Tensor)
        
        # modcrop image first
        C, H, W = img.shape
        H_r, W_r = H % self.scale, W % self.scale
        img = img[:, :H - H_r, :W - W_r]
        
        # apply resize function as in MATLAB
        lr = image.imresize(img, scale=1/self.scale)
        return lr, img
    
        