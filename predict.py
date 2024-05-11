from argparse import ArgumentParser
from pathlib import Path

import torch

from src.utils.builders import build_module
from src.utils.other import load_module


def main():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint_path', type=Path, help='Checkpoint path')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    assert args.config.exists(), f"Config is not found: {args.config}"
    assert args.checkpoint_path.exists(), f"Checkpoint is not found: {args.path}"
    config = load_module(args.config)

    # main_module
    main_module = build_module(config.mainmodule_cfg())
    # load checkpoint
    state_dict = torch.load(str(args.checkpoint_path), map_location='cpu')['state_dict']
    main_module.load_state_dict(state_dict, strict=False)
    main_module.eval()
    main_module.to(args.device).half()

    preds = main_module.forward(['Tell me somethink interesting'], device=args.device)
    print(preds)


if __name__ == '__main__':
    main()
