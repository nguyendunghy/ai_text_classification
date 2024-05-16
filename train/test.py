from argparse import ArgumentParser
from pathlib import Path

import torch
from lightning import Trainer

from src.data.module import DataModule
from src.utils.builders import build_module
from src.utils.other import load_module


def main():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint_path', type=Path, help='Checkpoint path')
    args = parser.parse_args()

    assert args.config.exists(), f"Config is not found: {args.config}"
    assert args.checkpoint_path.exists(), f"Checkpoint is not found: {args.path}"
    config = load_module(args.config)

    data_module = DataModule(**config.datamodule_cfg())

    main_module = build_module(config.mainmodule_cfg())
    state_dict = torch.load(str(args.checkpoint_path), map_location='cpu')['state_dict']
    main_module.load_state_dict(state_dict, strict=False)
    main_module.eval()
    main_module.cuda().half()

    trainer = Trainer()
    trainer.test(main_module, datamodule=data_module)


if __name__ == '__main__':
    main()
