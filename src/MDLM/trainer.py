from tqdm import tqdm
import math
import wandb
import omegaconf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DiffusionTrainer(nn.Module):
    def __init__(self, model, config, tokenizer, max_steps):
        super(DiffusionTrainer, self).__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.warmup_steps = config.optimizer.warmup_steps
        
        self.optimizer = torch.optim.AdamW(
            model.backbone.parameters(), 
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay
        )   
        
        warmup_steps = config.optimizer.warmup_steps
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup from 0 to 1
                return step / warmup_steps
            else:
                # Cosine annealing from 1 to 0.1
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(progress * math.pi))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=lr_lambda
        )
        
        
    def step(self, data_loader: DataLoader, train: bool = True):
        total_loss = 0.0
        mode = "train" if train else "val"
        iters = len(data_loader)
        data_iter = tqdm(enumerate(data_loader),total=iters)
        for i, data in data_iter:    
            seq_token = data['sequence'].to(self.model.device)       
            # vdj_token = data['vdj_label'].to(self.model.device) 
            # anarci_token = data['anarci_label'].to(self.model.device)
            # batch_size = seq_token.size(0) 
            
            if train:
                self.optimizer.zero_grad()
            nlls = self.model.forward_pass_diffusion(seq_token)
            loss = nlls.mean()
            total_loss += loss.item()
            
            post_fix = {
                f'{mode}_nll': loss.item(),
                'lr': self.scheduler.get_last_lr()[0] if self.scheduler is not None else None,
            }

            if train:
                loss.backward()
                if self.config.optimizer.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optimizer.grad_clip)
                self.optimizer.step()     
                self.scheduler.step()
                wandb.log(post_fix)
                            
        return loss / len(data_iter) 
    
    
    def fit(self, train_loader: DataLoader):
        self.model.train()
        train_loss = self.step(train_loader, train=True)
        return train_loss
    

    def validate(self, val_loader: DataLoader):
        self.model.eval()
        with torch.no_grad():
            val_loss = self.step(val_loader, train=False)
        return val_loss