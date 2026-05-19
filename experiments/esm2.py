"""Simple fine-tuning of ESM2 by training new prediction head."""

import tomllib
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class AlignedSequenceFromCSVDataset(Dataset):
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

    def __get_item__(self, idx):
        return self.data[idx]


def collate_esm(samples, esm_batch_converter):
    _, _, tokens = esm_batch_converter(samples)
    X = tokens

    ys = torch.empty(len(samples))
    for idx, (id, _) in enumerate(samples):
        if id.split('|')[1] == 'high-poly':
            ys[idx] = 1.0
        else:
            ys[idx] = 0.0

    return X, ys


cluster_name = 'clusters_85'
train_path = Path(f'data/processed/splits/{cluster_name}/train.csv')
val_path = Path(f'data/processed/splits/{cluster_name}/val.csv')

id_col = 'split_id'
with open('config.toml', 'rb') as file:
    config = tomllib.load(file)
seq_cols = config['imgt_columns']['all_polyreactivity']
batch_size = 64

if __name__ == '__main__':
    train_data = AlignedSequenceFromCSVDataset(train_path, id_col, seq_cols)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
