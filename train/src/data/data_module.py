from typing import Optional, Dict, Any

import lightning as pl
import numpy as np
from torch.utils.data import DataLoader

from src.utils.builders import build_dataset, build_tokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        tokenizer = build_tokenizer(self.hparams.tokenizer_cfg)

        self.train_dataset = build_dataset(tokenizer, self.hparams.train_dataset_cfg)
        self.val_dataset = build_dataset(tokenizer, self.hparams.val_dataset_cfg)
        self.test_dataset = build_dataset(tokenizer, self.hparams.test_dataset_cfg) \
            if self.hparams.test_dataset_cfg is not None else None

        self.stats()

    def stats(self):
        ai_generated_labels, model_name_labels = self.train_dataset.labels
        ai_generated_counts = np.unique(ai_generated_labels, return_counts=True)
        print('ai_generated', ai_generated_counts)
        self.pos_weight = ai_generated_counts[1][0] / ai_generated_counts[1][1]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.hparams.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.hparams.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.hparams.loader_kwargs)
