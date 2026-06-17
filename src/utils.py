"""Common functions for handling data and training."""

from dataclasses import dataclass

from torch.utils.data import Dataset
from torchmetrics import Metric


class AlignedSequencesFromCSVDataset(Dataset):
    def __init__(self, path, id_col, seq_cols, sep=','):
        data = []
        with open(path) as file:
            line = file.readline()
            cols = line.rstrip('\n').split(sep)
            col_to_idx = {col: idx for idx, col in enumerate(cols)}
            for line in file:
                fields = line.rstrip('\n').split(sep)
                id = fields[col_to_idx[id_col]]
                seq = [fields[col_to_idx[col]] for col in seq_cols]
                seq = ''.join(seq)
                data.append((id, seq))
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


@dataclass
class TrainingMetrics:
    loss: Metric
    acc: Metric
    recall: Metric
    precision: Metric
