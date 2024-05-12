import json
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from src.utils.builders import build_module
from src.utils.other import load_module


def main():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint_path', type=Path, help='Checkpoint path')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    assert args.config.exists(), f"Config is not found: {args.config}"
    assert Path('resources/checkpoints').exists(), f"Resources is not found: {Path('resources')}"
    args.checkpoint_path = list(Path('resources/checkpoints').iterdir())[0]
    assert args.checkpoint_path.exists(), f"Checkpoint is not found: {args.checkpoint_path}"
    config = load_module(args.config)

    # main_module
    main_module = build_module(config.mainmodule_cfg())
    # load checkpoint
    state_dict = torch.load(str(args.checkpoint_path), map_location='cpu')['state_dict']
    main_module.load_state_dict(state_dict, strict=False)
    main_module.eval()
    main_module.to(args.device).half()

    with open('resources/test_data_2.json', 'r') as f:
        data = json.load(f)
        texts = data['data']

    t1 = time.time()
    dataloader = DataLoader(texts, batch_size=32, shuffle=False, num_workers=18)

    trainer = Trainer()
    preds = trainer.predict(main_module, dataloader)
    print(preds)
    t2 = time.time()
    print(f"Time: {t2 - t1:.2f} s")


if __name__ == '__main__':
    main()
