from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lightning import Trainer

from src.data.module import DataModule
from src.utils.builders import build_module
from src.utils.other import load_module
from src.utils.builders import build_dataset


def main():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint_path', type=Path, help='Checkpoint path')
    args = parser.parse_args()

    assert args.config.exists(), f"Config is not found: {args.config}"
    assert args.checkpoint_path.exists(), f"Checkpoint is not found: {args.checkpoint_path}"
    config = load_module(args.config)

    dataset = build_dataset(dict(
        type='JsonDataset',
        json_file='resources/sample_data_1715832347273793558.json',
    ))
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1)

    main_module = build_module(config.mainmodule_cfg())
    state_dict = torch.load(str(args.checkpoint_path), map_location='cpu')['state_dict']
    main_module.load_state_dict(state_dict, strict=False)
    main_module.eval()
    main_module.cuda()
    main_module.half()

    trainer = Trainer()
    trainer.test(main_module, dataloaders=data_loader)


if __name__ == '__main__':
    main()
