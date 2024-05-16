from argparse import ArgumentParser
from pathlib import Path

import torch
from lightning import Trainer

from src.data.data_module import DataModule
from src.utils.builders import build_module
from src.utils.other import load_module
from src.utils.pl_utils import build_params_for_trainer


def main():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    args = parser.parse_args()

    assert args.config.exists(), f"Config is not found: {args.config}"

    config = load_module(args.config)

    data_module = DataModule(**config.datamodule_cfg())

    trainer_cfg = config.trainer_cfg()
    # seed_everything_deterministic(configs.seed)

    main_module = build_module(config.mainmodule_cfg())

    torch.set_float32_matmul_precision('medium')
    trainer = Trainer(**build_params_for_trainer(args, trainer_cfg, main_module, with_wandb=False))
    trainer.fit(main_module, datamodule=data_module)


if __name__ == '__main__':
    main()
