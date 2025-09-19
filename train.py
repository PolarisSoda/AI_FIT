import argparse
from omegaconf import OmegaConf, DictConfig, ListConfig
from model_action.model import ExerciseModel

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

    train_with_config(cfg)