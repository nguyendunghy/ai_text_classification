from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from lightning import Trainer

from src.data.data_module import DataModule
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

    main_module = build_module(config.mainmodule_cfg())
    state_dict = torch.load(str(args.checkpoint_path), map_location='cpu')['state_dict']
    main_module.load_state_dict(state_dict, strict=False)
    main_module.eval()
    main_module.cuda()
    main_module.half()

    # datasets = []
    # for json_file in [
    #     'resources/sample_data_1715835733924547734.json',
    #     'resources/sample_data_1715837851346488770.json',
    #     'resources/sample_data_1715840415677838057.json',
    #     'resources/sample_data_1715842067626690262.json'
    # ]:
    #     dataset = build_dataset(dict(
    #         type='JsonDataset',
    #         json_file=json_file,
    #     ))
    #     datasets.append(dataset)
    # concat_dataset = ConcatDataset(datasets)
    datamodule_cfg = config.datamodule_cfg()
    val_dataset = build_dataset(datamodule_cfg.get('val_dataset_cfg'))
    data_loader = DataLoader(val_dataset, shuffle=False, batch_size=16)

    trainer = Trainer()
    trainer.test(main_module, dataloaders=data_loader)


if __name__ == '__main__':
    main()
