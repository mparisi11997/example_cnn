import yaml
from data.processing_data import make_dataloader
import torch

PATH_CONFIGS = 'config/config.yaml'

def load_config() -> dict:
    with open(PATH_CONFIGS, 'r') as f:
        return yaml.safe_load(f)    
    
if __name__ == '__main__':
    # cfg = load_config()
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # train_dl,val_dl,test_dl = make_dataloader(
    #     data_dir = cfg["paths"]["data_dir"],
    #     dataset_name = cfg["data"]["dataset_name"],
    #     mean= cfg["data"]["mean"],
    #     std=cfg["data"]["std"],
    #     download=cfg["data"]["download"],
    #     val_split=cfg["data"]["val_split"],
    #     pin_memory=cfg["data"]["pin_memory"],
    #     device=device,
    #     batch_size=cfg["data"]["batch_size"],
    #     num_workers=cfg["data"]["num_workers"]

    # )

