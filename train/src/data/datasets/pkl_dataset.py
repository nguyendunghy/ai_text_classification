from typing import Tuple

import pandas as pd
from torch.utils.data import Dataset


class PKLDataset(Dataset):
    def __init__(self, csv_file, model_names):
        super().__init__()
        self._model_names = model_names
        self._df = pd.read_pickle(csv_file)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx) -> Tuple[str, Tuple[str, float]]:
        row = self._df.iloc[idx]
        is_ai_generated = row['label']
        prompt = row['prompt']
        text = row['text']
        model_name_idx = self._model_names[row['model_name']]
        if is_ai_generated:
            text = prompt + text
        assert isinstance(text, str), f'Text {text} is not a string'
        return text, (model_name_idx, float(is_ai_generated))
