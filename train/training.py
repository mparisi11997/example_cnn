

from dataclasses import dataclass
import os
from typing import Any, Dict

import torch
from torch import nn,optim
from data.processing_data import make_dataloader
from models.cnn import CNN
from utils.seed import set_seed

@dataclass
class TrainConfig:
    data_dir: str
    checkpoint_dir: str
    best_ckpt_name: str
    dataset_name: str 
    mean:float
    std:float
    batch_size: int
    num_workers:  int
    val_split: float
    download: bool
    pin_memory: bool
    in_channels: int
    num_classes: int
    dropout: float
    use_batcnorm:bool
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    scheduler: Dict[str,Any]     

def build_scheduler(optimizer:optim.Optimizer, cfg: Dict[str, Any]):
    name = (cfg.get("name") or "").lower()
    if name == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size")),
            gamma=float(cfg.get("gamma"))
        )
    else:
        return None


def run_training(cfg:TrainConfig):
    set_seed(cfg.seed)
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device{device}")    


    train_loader,val_loader,_= make_dataloader(
        dataset_name=cfg.dataset_name,
        data_dir=cfg.data_dir,
        mean=cfg.mean,
        std=cfg.std,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=cfg.val_split,
        download=cfg.download,
        pin_memory=cfg.pin_memory,
        device=device.type
    )

    model = CNN(
        dropout=cfg.dropout,
        use_batchnorm=cfg.use_batcnorm,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes
    ).to(device)

    criterior= nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters(),
                          lr=cfg.lr,
                          weight_decay=cfg.weight_decay,
                          )
    
    
    scheduler = build_scheduler(optimizer,cfg.scheduler)

    os.makedirs(cfg.checkpoint_dir,exist_ok=True)
    best_path = os.path.join(cfg.checkpoint_dir,cfg.best_ckpt_name)

    for epoch in range(1,cfg.epochs +1):
        train_one_epoch
        evaluate
    
    save_checkpoint()

    return best_path