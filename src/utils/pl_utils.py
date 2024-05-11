import wandb
from lightning.pytorch.loggers import WandbLogger

from src.utils.builders import build_callbacks


def build_params_for_trainer(args, trainer_cfg, lightning_module, with_wandb=True):
    if 'callbacks' in trainer_cfg:
        trainer_cfg['callbacks'] = [
            build_callbacks(config)
            for config in trainer_cfg['callbacks']
        ]
    wandb_kwargs = trainer_cfg.pop('wandb_logger')
    if not with_wandb:
        wandb_kwargs['mode'] = 'disabled'

    wandb_key_path = wandb_kwargs.pop('key_path')
    if wandb_key_path.exists():
        with open(str(wandb_key_path), 'r') as file:
            key = file.readline()
            wandb.login(key=key)
            trainer_cfg['logger'] = WandbLogger(**wandb_kwargs)
            trainer_cfg['logger'].watch(lightning_module, log=None, log_freq=100, log_graph=True)
            wandb.save(str(args.config))
    return trainer_cfg
