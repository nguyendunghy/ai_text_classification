from typing import Tuple

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, tokenizer, transforms=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer

        from src.utils.builders import build_transform

        transforms = transforms if transforms is not None else []
        self.transforms = [build_transform(transform) for transform in transforms]

    def apply_transforms(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

    def tokenize(self, text: str):
        return self.tokenizer(text)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx) -> Tuple[str, Tuple[str, float]]:
        raise NotImplementedError
