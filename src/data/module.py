from typing import Optional, Dict, Any

import lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils.builders import build_dataset


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_cfg: Dict[str, int],
                 loader_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._dataset = build_dataset(dataset_cfg)
        self._loader_kwargs = loader_kwargs or dict()

    def setup(self, stage: str) -> None:
        train_part, test_dataset = train_test_split(self._dataset, test_size=0.1, random_state=42)
        train_dataset, val_dataset = train_test_split(train_part, test_size=0.2, random_state=42)

        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self._train_dataset, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, **self._loader_kwargs)
