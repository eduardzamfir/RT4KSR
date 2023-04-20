import os
import pathlib
import argparse
import torch


def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataroot", type=str, default=os.path.join(pathlib.Path.home(), "datasets/image_restoration"))
    parser.add_argument("--benchmark", type=str, nargs="+", default=["ntire23rtsr"])
    parser.add_argument("--checkpoints-root", type=str, default="code/checkpoints")
    parser.add_argument("--checkpoint-id", type=str, default="rt4ksr_x2")
    
    # model definitions
    parser.add_argument("--bicubic", action="store_true")
    parser.add_argument("--arch", type=str, default="rt4ksr_rep")
    parser.add_argument("--feature-channels", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--act-type", type=str, default="gelu", choices=["relu", "lrelu", "gelu"])
    parser.add_argument("--is-train", action="store_true", help="Switch between training and inference mode for reparameterizable blocks.")
    parser.add_argument("--rep", action="store_true", help="Run inference with reparameterized version.")
    parser.add_argument("--save-rep-checkpoint", action="store_true", help="Save checkpoint of reparameterized model intance.")

    # data
    parser.add_argument("--scale", type=int, default=2, choices=[2,3])
    parser.add_argument("--rgb-range", type=float, default=1.0)
    
    return parser.parse_args()