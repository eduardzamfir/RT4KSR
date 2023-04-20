import os
import glob
import torch
import pathlib
import unittest
import numpy as np
import torch.nn.functional as F
import torchmetrics
import kornia.color as color

from collections import OrderedDict
from typing import List, Tuple
from torch.nn.functional import interpolate

import data
import model
from utils import image, utils, metrics, parser
from model.utils import rep_utils


def load_checkpoint(model, device, time_stamp=None):       
    checkpoint = glob.glob(os.path.join("results/checkpoints", time_stamp + ".pth"))
    if isinstance(checkpoint, List):
        checkpoint = checkpoint.pop(0)
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
 

def reparameterize(config, model, device, save_rep_checkpoint=False):
    config.is_train = False
    rep_model = torch.nn.DataParallel(model.__dict__[config.arch + "_simplified"](config)).to(device)
    rep_state_dict = rep_model.state_dict()
    pretrained_state_dict = model.state_dict()
    
    for k, v in rep_state_dict.items():            
        if "rep_conv.weight" in k:
            # merge conv1x1-conv3x3-conv1x1
            k0 = pretrained_state_dict[k.replace("rep", "expand")]
            k1 = pretrained_state_dict[k.replace("rep", "fea")]
            k2 = pretrained_state_dict[k.replace("rep", "reduce")]
            
            bias_str = k.replace("weight", "bias")
            b0 = pretrained_state_dict[bias_str.replace("rep", "expand")]
            b1 = pretrained_state_dict[bias_str.replace("rep", "fea")]
            b2 = pretrained_state_dict[bias_str.replace("rep", "reduce")]
            
            mid_feats, n_feats = k0.shape[:2]

            # first step: remove the middle identity
            for i in range(mid_feats):
                k1[i, i, 1, 1] += 1.0
        
            # second step: merge the first 1x1 convolution and the next 3x3 convolution
            merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3).cuda()
            merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1)

            # third step: merge the remain 1x1 convolution
            merged_k0k1k2 = F.conv2d(input=merged_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
            merged_b0b1b2 = F.conv2d(input=merged_b0b1, weight=k2, bias=b2).view(-1)

            # last step: remove the global identity
            for i in range(n_feats):
                merged_k0k1k2[i, i, 1, 1] += 1.0
            
            # save merged weights and biases in rep state dict
            rep_state_dict[k] = merged_k0k1k2.float()
            rep_state_dict[bias_str] = merged_b0b1b2.float()
            
        elif "rep_conv.bias" in k:
            pass
            
        elif k in pretrained_state_dict.keys():
            rep_state_dict[k] = pretrained_state_dict[k]

    rep_model.load_state_dict(rep_state_dict, strict=True)
    if save_rep_checkpoint:
        torch.save(rep_state_dict, f"rep_model_{config.checkpoint_id}.pth")
        
    return rep_model


def test(config):
    """
    SETUP METRICS
    """
    test_results = OrderedDict()
    test_results["psnr_rgb"] = []
    test_results["psnr_y"] = []
    test_results["ssim_rgb"] = []
    test_results["ssim_y"] = []
    test_results = test_results

    """
    SETUP MODEL
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(
        model.__dict__[config.arch + "_" + config._type](config)
        ).to(device)
    model = load_checkpoint(net, device, config.checkpoint_id)
    
    
    if config.rep:
        rep_model = reparameterize()
        model = rep_model

    model.eval()
        
    for benchmark in config.benchmark:
        test_loader = torch.utils.data.DataLoader(
            data.__dict__[benchmark](config),
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False, 
            drop_last=False
        )
        with torch.no_grad():
            print("Testing...")   
            for i, batch in enumerate(test_loader):
                lr_img = batch["lr"].to(device)
                hr_img = batch["hr"].to(device)
                
                # run method
                if config.bicubic:
                    out = interpolate(lr_img, scale_factor=config.scale, mode="bicubic", align_corners=False).clamp(min=0, max=255)
                else:
                    out = model(lr_img)

                    # compute metrics
                    test_results["psnr_rgb"].append(psnr(out,hr_img))
                    test_results["ssim_rgb"].append(ssim(out, hr_img))
                    
                    # compute metrics on y channel
                    out_y = color.rgb_to_y(out)
                    hr_y = color.rgb_to_y(hr_img)
                    test_results["psnr_y"].append(psnr(out_y, hr_y))
                    test_results["ssim_y"].append(ssim(out_y, hr_y))


            print(f"------> Results of X{config.scale} for benchmark: {benchmark}")
            ave_psnr_rgb = sum(test_results["psnr_rgb"]) / len(test_results["psnr_rgb"])
            print('------> Average PSNR (RGB): {:.6f} dB'.format(ave_psnr_rgb))
            ave_ssim_rgb = sum(test_results["ssim_rgb"]) / len(test_results["ssim_rgb"])
            print('------> Average SSIM (RGB): {:.6f}'.format(ave_ssim_rgb))
            ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
            print('------> Average PSNR (Y): {:.6f} dB'.format(ave_psnr_y))
            ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"]) 
            print('------> Average SSIM (Y): {:.6f}'.format(ave_ssim_y))
            
        

if __name__ == "__main__":
    args = parser.test_parser()
    
    test(args)