import pandas as pd

from src.data.datasets.base import BaseDataset


class PKLDataset(BaseDataset):
    def __init__(self, csv_file, model_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_names = model_names
        self._df = pd.read_pickle(csv_file)

    def __len__(self):
        return len(self._df)

    @property
    def labels(self):
        return self._df['label'].values, self._df['model_name'].values

    def __getitem__(self, idx):
        row = self._df.iloc[idx]
        is_ai_generated = row['label']
        # prompt = row['prompt']
        text = row['text']
        model_name_idx = self._model_names[row['model_name']]
        # if is_ai_generated:
        #     text = prompt + text
        assert isinstance(text, str), f'Text {text} is not a string'
        text = self.apply_transforms(text)
        x = self.tokenize(text)
        return x, (model_name_idx, float(is_ai_generated))
