import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from src.utils.builders import build_module, build_tokenizer
from src.utils.other import load_module


def parse():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint_path', type=Path, help='Checkpoint path')
    parser.add_argument('--batch-size', type=int, default=16)
    return parser.parse_args()


def load_model(config, checkpoint_path: Path):
    main_module_cfg = config.mainmodule_cfg(
        train_ds_size=0,
        pos_weight=1
    )
    main_module = build_module(main_module_cfg)
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')['state_dict']
    main_module.load_state_dict(state_dict, strict=False)
    main_module.eval()
    main_module.cuda()
    main_module.half()
    return main_module


class Predictor:
    def __init__(self, config_path: Path, checkpoint_path: Path, batch_size: int):
        config = load_module(config_path)

        self._tokenizer = build_tokenizer(config.datamodule_cfg()['tokenizer_cfg'])
        self._main_module = load_model(config, checkpoint_path)
        self._batch_size = batch_size

    def __call__(self, texts):
        x = [self._tokenizer(text) for text in texts]
        data_loader = DataLoader(x, batch_size=self._batch_size)

        trainer = Trainer()
        outs = trainer.predict(self._main_module, dataloaders=data_loader)
        ai_output = []
        for out in outs:
            ai_output.extend(out['ai_output'].cpu().numpy().tolist())
        return (np.array(ai_output) > 0.5).tolist()


def load_test_data():
    with open('resources/sample_data/sample_data_1715846152327770032.json', 'r') as f:
        data = json.load(f)
        texts = data['texts']
        labels = data['labels']
    return texts, labels


def main(args):
    predictor = Predictor(args.config, args.checkpoint_path, args.batch_size)

    texts, labels = load_test_data()
    t1 = time.time()
    preds = predictor(texts)
    print(preds)
    print(sum(np.array(labels) == np.array(preds)) / len(labels))
    t2 = time.time()
    print(f"Time: {t2 - t1:.2f} s")


if __name__ == '__main__':
    args = parse()
    assert args.config.exists(), f"Config is not found: {args.config}"
    assert args.checkpoint_path.exists(), f"Checkpoint is not found: {args.checkpoint_path}"
    main(args)
