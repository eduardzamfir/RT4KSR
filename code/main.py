import os
import glob
import wandb
import torch
import pathlib
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import kornia.color as color
import kornia.losses as klosses


from typing import List
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure 

import data
import model
from utils import image, utils, parser, losses




#####################
### TRAINER CLASS ###
#####################
        
class Trainer(object):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device

        # timestamp
        self.time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        # init model
        if config.distributed:
            net = model.__dict__[config.arch + "_" + config.type](config).to(device)
            self.model = DDP(net, device_ids=[device], find_unused_parameters=True)
        else:
            net = model.__dict__[config.arch + "_" + config.type](config).to(device)
            self.model = torch.nn.DataParallel(net)

        # loss function
        if config.charbonnier:
            self.criterion = klosses.CharbonnierLoss(reduction="mean")
        else:
            self.criterion = torch.nn.L1Loss().to(device)
        self.aux_loss = losses.AuxLoss(scale=config.scale, device=self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)

        # scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.lr_milestones, gamma=config.scheduler_gamma)
            
        
    def train_step(self, batch, is_train):
        lr = batch["lr"].to(self.device)
        hr = batch["hr"].to(self.device)
            
        out = self.model(lr) 
        l_val = self.criterion(out, hr)
        
        if self.config.auxloss:
            aux_val = self.aux_loss(out, hr)
        else:
            aux_val = 0
                   
        if is_train:
            self.optimizer.zero_grad()
            loss = l_val + aux_val * self.config.loss_weight
            loss.backward() 
            
            # add gradient clipping against exploding gradients issue
            if self.config.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm, norm_type=2)    
            self.optimizer.step()
        
        return l_val, aux_val
    
    
    def test_step(self, batch):
        lr = batch["lr"].to(self.device)
        hr = batch["hr"].to(self.device)
        
        out = self.model(lr)
        
        hr = hr[:lr.shape[-2] * self.config.scale, :lr.shape[-1] * self.config.scale, ...]

        # compute metrics
        psnr = peak_signal_noise_ratio(out, hr, data_range=self.config.rgb_range)
        ssim = structural_similarity_index_measure(out, hr, data_range=self.config.rgb_range)
        
        # compute metrics on y channel
        out_y = color.rgb_to_y(out)
        hr_y = color.rgb_to_y(hr)
        psnr_y = peak_signal_noise_ratio(out_y, hr_y, data_range=self.config.rgb_range)
        ssim_y = structural_similarity_index_measure(out_y, hr_y, data_range=self.config.rgb_range)
        
        return {"psnr_rgb":psnr,
                "psnr_y":psnr_y,
                "ssim_rgb":ssim,
                "ssim_y":ssim_y}

    
    def load_checkpoint(self, time_stamp=None):
        if time_stamp is None:
            time_stamp = self.time_stamp
            
        # load last checkpoint
        checkpoint = glob.glob(os.path.join(self.config.checkpoints_root, time_stamp + ".pth"))
        if isinstance(checkpoint, List):
            checkpoint = checkpoint.pop(0)
        
        if self.config.distributed:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
        else:
            map_location = self.device
            
        checkpoint = torch.load(checkpoint, map_location=map_location)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        
        return checkpoint
 
    
    def save_checkpoint(self, epoch, current_iter, suffix):
        state = {
            "epoch": epoch + 1,
            "iteration": current_iter + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        
        # checkpoint path for saving
        checkpoint_path = os.path.join(self.config.checkpoints_root, self.time_stamp + f"_{suffix}" + ".pth")
        torch.save(state, checkpoint_path)
 
        
    def init_model_from(self, time_stamp):
        # load checkpoint
        checkpoint = self.load_checkpoint(time_stamp)
   
    

############
### MAIN ###
############

def main(gpu: int, config):
    
    ###############################################################################
    # Init
    ###############################################################################
    
    # create out dir
    pathlib.Path(config.checkpoints_root).mkdir(parents=True, exist_ok=True)
    
    # distributed computing
    if config.distributed:
        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=config.world_size, rank=gpu)
        torch.distributed.barrier()
        print(f"{gpu + 1}/{config.world_size} process initialized.\n")

        
    # set main process and device
    main_process = not config.distributed or (config.distributed and gpu == 0)
    device = gpu if config.distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ###############################################################################
    # Trainer and datasets
    ###############################################################################
    
    # init trainer object
    trainer = Trainer(config, device)
    
    # create train and val dataloaders
    train_loader, val_loader = get_dataset(config)
    
        
    # init from checkpoint
    if config.init_from is not None:
        trainer.init_model_from(config.init_from)
        print(f"Initialized model from: {config.init_from}")
        

    ###############################################################################
    # Training loop
    ###############################################################################
    
    best_vallos = 1
    current_iter = 0
    for epoch in tqdm(range(0, config.max_epochs)):
        if config.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        # training
        ep_loss = []
        trainer.model.train()
        for batch in train_loader:
            current_iter += 1
            loss, aux_val = trainer.train_step(batch, is_train=True)
            ep_loss.append(loss.item())

        trainer.scheduler.step()   
        
        # validation     
        ep_valloss = []
        trainer.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                loss, _ = trainer.train_step(batch, is_train=False)
                ep_valloss.append(loss.item())
                
            # save checkpoint
            if np.mean(ep_valloss) < best_vallos:
                best_vallos = np.mean(ep_valloss)
                trainer.save_checkpoint(current_iter=current_iter, epoch=epoch, suffix="best")


    ###############################################################################
    # Testing
    ###############################################################################
        
    # save checkpoint after last epoch
    if main_process:
        trainer.save_checkpoint(current_iter=current_iter, epoch=epoch, suffix="last")

        # load checkpoint
        _ = trainer.load_checkpoint(time_stamp=trainer.time_stamp + "_last")
        trainer.model.eval()
        
        # run test 
        for benchmark in config.benchmark:
            test_loader = DataLoader(
                data.__dict__[benchmark](config),
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                shuffle=False, 
                drop_last=False
            )
            
            test_psnr = []
            test_psnr_y = []
            test_ssim = []
            test_ssim_y = []
            with torch.no_grad():      
                for batch in test_loader:
                    metrics = trainer.test_step(batch)
                    
                    test_psnr.append(metrics["psnr_rgb"])
                    test_ssim.append(metrics["ssim_rgb"])
                    test_psnr_y.append(metrics["psnr_y"])
                    test_ssim_y.append(metrics["ssim_y"])
                    
                print(f"---> Benchmark {benchmark.capitalize()} Avg PSNR (RGB) | {sum(test_psnr) / len(test_psnr) :.2f} dB")
                print(f"---> Benchmark {benchmark.capitalize()} Avg SSIM (RGB) | {sum(test_ssim) / len(test_ssim) :.4f}")
                print(f"---> Benchmark {benchmark.capitalize()} Avg PSNR (Y) | {sum(test_psnr_y) / len(test_psnr_y) :.2f} dB")
                print(f"---> Benchmark {benchmark.capitalize()} Avg SSIM (Y) | {sum(test_ssim_y) / len(test_ssim_y) :.4f}")


    ###############################################################################
    # Clean up
    ###############################################################################
    
    if config.distributed:  
        dist.destroy_process_group()



###############
### HELPER ###
###############

def get_dataset(config):
    datasets_list = [data.__dict__[config.dataset_name](config, mode="train")]
    
    if config.use_gta:
        datasets_list.append(data.gta(config, mode="train"))
        train_set = ConcatDataset(datasets_list)
    elif config.use_lsdir:
        datasets_list.append(data.lsdir_sub(config, mode="train"))
        train_set = ConcatDataset(datasets_list)
    elif config.use_gta and config.use_lsdir:
        datasets_list.append(data.gta(config, mode="train"))
        datasets_list.append(data.lsdir_sub(config, mode="train"))
        train_set = ConcatDataset(datasets_list)
    else:
        train_set = data.__dict__[config.dataset_name](config, mode="train")
        
    val_set = data.div2k(config, mode="valid")
    
    if config.distributed:
        world_size = config.world_size
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, shuffle=True)
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, shuffle=True)
    else:
        world_size = 1
        train_sampler = None
        val_sampler = None
        
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        shuffle=(True if train_sampler is None else False),
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=val_sampler,
        pin_memory=True,
        shuffle=(True if train_sampler is None else False), 
        drop_last=True
    )
    
    return train_loader, val_loader




if __name__ == "__main__":
    args = parser.train_parser()
    utils.fix_seed(args.seed)
    
    # Force disable distributed
    args.distributed = False if not torch.cuda.is_available() else args.distributed

    # Distributed training with multiple gpus
    if args.distributed:
        args.batch_size = args.batch_size // args.gpus

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        
        args.world_size = args.gpus
        mp.spawn(main, nprocs=args.world_size, args=(args,), join=True)

    # DataParallel with GPUs or CPU
    else:
        main(gpu=0, config=args)