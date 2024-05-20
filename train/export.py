import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.onnx import export

from src.utils.builders import build_module
from src.utils.other import load_module


def parse():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint_path', type=Path, help='Checkpoint path')
    return parser.parse_args()


def load_model(config, checkpoint_path):
    config = load_module(config)

    main_module_cfg = config.mainmodule_cfg(
        train_ds_size=0,
        pos_weight=1
    )
    main_module = build_module(main_module_cfg)
    # load checkpoint
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')['state_dict']
    main_module.load_state_dict(state_dict, strict=False)
    main_module.eval()
    main_module.half()
    return main_module


def main(args):
    main_module = load_model(args.config, args.checkpoint_path)

    main_module.forward = main_module.forward_postprocess

    inputs = dict(
        x=dict(
            input_ids=torch.zeros((1, 512), dtype=torch.int32),
            attention_mask=torch.zeros((1, 512), dtype=torch.int32)
        )
    )
    input_names = ['input_ids', 'attention_mask']
    output_names = ['ai_generated', 'model_name']
    with torch.no_grad():
        export(
            main_module, inputs,
            'model.onnx',
            opset_version=16,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            dynamic_axes={**{name: {0: 'batch_size'} for name in input_names},
                          **{name: {0: 'batch_size'} for name in output_names}},
            do_constant_folding=True
        )


if __name__ == '__main__':
    args = parse()

    assert args.config.exists(), f"Config is not found: {args.config}"
    assert args.checkpoint_path.exists(), f"Checkpoint is not found: {args.checkpoint_path}"

    main(args)
