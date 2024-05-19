from argparse import ArgumentParser
from pathlib import Path

import torch
from lightning import Trainer

from src.data.data_module import DataModule
from src.utils.common import seed_everything_deterministic
from src.utils.builders import build_module
from src.utils.other import load_module
from src.utils.pl_utils import build_params_for_trainer


def main(args):
    config = load_module(args.config)

    data_module = DataModule(**config.datamodule_cfg())

    trainer_cfg = config.trainer_cfg(
        ds_size=len(data_module.train_dataset)
    )
    seed_everything_deterministic(config.seed)

    main_module_cfg = config.mainmodule_cfg(
        train_ds_size=len(data_module.train_dataset),
        pos_weight=data_module.pos_weight
    )
    main_module = build_module(main_module_cfg)

    torch.set_float32_matmul_precision('medium')
    trainer = Trainer(**build_params_for_trainer(args, trainer_cfg, main_module, with_wandb=args.wandb))
    trainer.fit(main_module, datamodule=data_module)
    trainer.test(main_module, datamodule=data_module)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--wandb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    assert args.config.exists(), f"Config is not found: {args.config}"

    main(args)
