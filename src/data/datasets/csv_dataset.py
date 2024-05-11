import pandas as pd
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        self._df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        row = self._df.iloc[idx]
        return row['text'], float(row['label'])
