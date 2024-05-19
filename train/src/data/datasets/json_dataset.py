import json

from src.data.datasets.base import BaseDataset


class JsonDataset(BaseDataset):
    def __init__(self, json_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(json_file, 'r') as f:
            self._df = json.load(f)

    def __len__(self):
        return len(self._df['texts'])

    def __getitem__(self, idx):
        text = self._df['texts'][idx]
        is_ai_generated = self._df['labels'][idx]
        text = self.apply_transforms(text)
        x = self.tokenize(text)
        return x, (0, float(is_ai_generated))
