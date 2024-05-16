import json
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.onnx import export

from lightning import Trainer
from torch.utils.data import DataLoader

from src.utils.builders import build_module
from src.utils.other import load_module


def parse():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint_path', type=Path, help='Checkpoint path')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def load_model(config, checkpoint_path, device):
    config = load_module(config)

    # main_module
    main_module = build_module(config.mainmodule_cfg())
    # load checkpoint
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')['state_dict']
    main_module.load_state_dict(state_dict, strict=False)
    main_module.eval()
    main_module.to(device).half()
    return main_module


def load_test_data():
    with open('resources/test_data_2.json', 'r') as f:
        data = json.load(f)
        texts = data['data']
    return texts


def main(args):
    main_module = load_model(args.config, args.checkpoint_path, args.device)

    main_module.forward = main_module.forward_postprocess

    output_names = ['ai_generated', 'model_name']
    with torch.no_grad():
        export(
            main_module, torch.zeros((1, 512), dtype=torch.int32),
            'model.onnx',
            opset_version=11,
            input_names=['input'],
            output_names=output_names,
            export_params=True,
            dynamic_axes={'input': {0: 'batch_size'},
                          **{name: {0: 'batch_size'} for name in output_names}},
            do_constant_folding=True
        )


if __name__ == '__main__':
    args = parse()

    assert args.config.exists(), f"Config is not found: {args.config}"
    args.checkpoint_path = Path('resources/checkpoints/checkpoint_BinaryAccuracy=0.984_MulticlassAccuracy=0.967.ckpt')
    assert Path('resources/checkpoints').exists(), f"Resources is not found: {Path('resources')}"
    assert args.checkpoint_path.exists(), f"Checkpoint is not found: {args.checkpoint_path}"

    main(args)
