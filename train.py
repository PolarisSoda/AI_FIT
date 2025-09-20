import argparse
import random
import torch
import numpy as np

from omegaconf import OmegaConf, DictConfig, ListConfig
from model_action.model import ExerciseModel

def set_seed(seed: int = 42):
    random.seed(seed)                    # Python 내장 난수
    np.random.seed(seed)                 # NumPy 난수
    torch.manual_seed(seed)              # PyTorch CPU 난수
    torch.cuda.manual_seed(seed)         # 현재 GPU
    torch.cuda.manual_seed_all(seed)     # 모든 GPU

    # CuDNN 관련 (재현성 vs 속도 trade-off)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_with_config(cfg: DictConfig | ListConfig):
    if cfg.model.model == "ExerciseModel":
        model = ExerciseModel(**cfg)
    else:
        raise NotImplementedError("Not implemented yet.")
    
    model.train_with_num_epochs(cfg.model.num_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,required=False,default="cfg/train_agcn.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    set_seed()
    train_with_config(cfg)