from argparse import ArgumentParser
from pathlib import Path

import torch
from lightning import Trainer

from src.data.data_module import DataModule
from src.utils.builders import build_module
from src.utils.other import load_module


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint_path', type=Path, help='Checkpoint path')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    return parser.parse_args()


def main(args):
    config = load_module(args.config)

    data_module = DataModule(**config.datamodule_cfg())

    main_module_cfg = config.mainmodule_cfg(
        train_ds_size=len(data_module.train_dataset),
        pos_weight=data_module.pos_weight
    )
    main_module = build_module(main_module_cfg)
    state_dict = torch.load(str(args.checkpoint_path), map_location='cpu')['state_dict']
    main_module.load_state_dict(state_dict, strict=False)
    main_module.eval()
    main_module.to(args.device)
    main_module.half()

    trainer = Trainer()
    trainer.test(main_module, datamodule=data_module)


if __name__ == '__main__':
    args = parse_args()

    assert args.config.exists(), f"Config is not found: {args.config}"
    assert args.checkpoint_path.exists(), f"Checkpoint is not found: {args.path}"

    main(args)
