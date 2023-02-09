import pandas as pd
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, data_file: str, source: str, target: str, src_col_idx: int, tgt_col_idx: int):

        self.source = source
        self.target = target
        self.data = pd.read_csv(data_file)
        self.tgt_col_idx = tgt_col_idx
        self.src_col_idx = src_col_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source = self.data.iloc[index, self.src_col_idx]
        target = self.data.iloc[index, self.tgt_col_idx]

        return {
            self.source: source,
            self.target: target
        }
