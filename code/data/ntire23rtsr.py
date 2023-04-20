import os
import torch
import pathlib
import numpy as np

from utils import image

def uint2tensor3(img, rgb_range):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        
    if rgb_range != 1:
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
    else:
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)



class NTIRE23RTSR(torch.utils.data.Dataset):
    def __init__(self, dataroot, scale, rgb_range):
        self.scale = scale
        self.rgb_range = rgb_range
        self.lr_images = image.get_image_paths(os.path.join(dataroot, "testsets/NTIRE234K", f"LR{scale}"))
        self.hr_images = image.get_image_paths(os.path.join(dataroot, "testsets/NTIRE234K/GT"))

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, index):
        lr = self.lr_images[index]
        hr = self.hr_images[index]
        
        # get image names
        lr_img_name, _ = os.path.splitext(os.path.basename(lr))
        hr_img_name, _ = os.path.splitext(os.path.basename(hr))
        
        # check if sr and hr match
        assert lr_img_name == hr_img_name
        
        # load image
        lr_img = image.imread_uint(lr, n_channels=3)
        hr_img = image.imread_uint(hr, n_channels=3)

        # To tensor
        lr_img = uint2tensor3(lr_img, self.rgb_range)
        hr_img = uint2tensor3(hr_img, self.rgb_range)


        return {"lr":lr_img, "hr":hr_img}
    
    
def ntire23rtsr(config):
    return NTIRE23RTSR(config.dataroot, config.scale, config.rgb_range)