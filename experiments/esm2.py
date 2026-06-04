"""Simple fine-tuning of ESM2 by training new prediction head."""

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torchmetrics
import tqdm
from torch.utils.data import Dataset, DataLoader

import esm


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


class ESM2Classifier(nn.Module):
    def __init__(self, esm2_trunk):
        super().__init__()
        self.esm2_trunk = esm2_trunk
        self.classifier_head = ClassifierHead(esm2_trunk.embed_dim)

    def forward(self, tokens):
        num_layers = self.esm2_trunk.num_layers
        outputs = self.esm2_trunk(tokens, repr_layers=[num_layers])
        x = outputs['representations'][num_layers]
        mask = (~tokens.eq(self.esm2_trunk.padding_idx)).unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)

        x = self.classifier_head(x)
        return x


class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


def esm_collate_generator(samples, esm_batch_converter):
    unaligned_samples = []
    for id, seq in samples:
        seq = seq.replace('-', '')
        unaligned_samples.append((id, seq))
    _, _, tokens = esm_batch_converter(unaligned_samples)
    tokens = tokens

    labels = torch.empty(len(samples))
    for idx, (id, _) in enumerate(samples):
        if id.split('|')[1] == 'high-poly':
            labels[idx] = 1.0
        else:
            labels[idx] = 0.0

    return tokens, labels


cluster_name = 'clusters_85'
train_path = Path(f'data/processed/splits/{cluster_name}/train.csv')
val_path = Path(f'data/processed/splits/{cluster_name}/val.csv')

id_col = 'split_id'
# fmt: off
seq_cols = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
    '13', '14', '15', '16', '17', '17A', '18', '19', '20', '21', '22',
    '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
    '34', '35', '36', '37', '38', '39', '40', '41', '41A', '42', '42A',
    '42B', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52',
    '53', '54', '55', '56', '57', '58', '59', '60', '60A', '61A', '61',
    '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
    '73', '74', '75', '76', '76A', '77', '77A', '78', '79', '79A',
    '80', '81', '82', '83', '84', '85', '85A', '86', '87', '88', '89',
    '90', '91', '92', '93', '94', '95', '96', '97', '97A', '97B',
    '97C', '97D', '98', '98A', '98B', '99', '99A', '99B', '100',
    '100A', '101', '101A', '101B', '101C', '101D', '101E', '101F',
    '102', '103', '104', '105', '106', '107', '108',
    '109', '110', '111', '111A', '111B', '111C', '111D', '111E',
    '111F', '111G', '111H', '112I', '112H', '112G', '112F', '112E',
    '112D', '112C', '112B', '112A', '112', '113', '114', '115', '116',
    '117', '118', '119', '120', '121', '122', '123', '124', '125',
    '126', '127', '128',
]
# fmt: on

experiment_name = 'esm2_finetune_head'
run_name = 'initial_test'
batch_size = 128
epoch_num = 100
log_interval = 10

if __name__ == '__main__':
    esm2_trunk, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    model = ESM2Classifier(esm2_trunk)
    for param in model.esm2_trunk.parameters():
        param.requires_grad = False

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    loss_fn = torch.nn.BCELoss()

    batch_converter = alphabet.get_batch_converter()
    collate_fn = partial(esm_collate_generator, esm_batch_converter=batch_converter)

    train_data = AlignedSequencesFromCSVDataset(train_path, id_col, seq_cols)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn
    )
    val_data = AlignedSequencesFromCSVDataset(val_path, id_col, seq_cols)
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn
    )

    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    recall_metric = torchmetrics.classification.BinaryRecall().to(device)
    precision_metric = torchmetrics.classification.BinaryPrecision().to(device)

    # Training loop
    step_idx = 0
    logs = []
    for epoch_idx in range(epoch_num):
        # Train epoch
        loss = 0
        num_samples = 0
        model.train()
        with tqdm.trange(
            len(train_dataloader), unit='batch', bar_format='{l_bar}{bar:10}{r_bar}'
        ) as bar:
            bar.set_description(f'epoch {epoch_idx}')
            for input, target in train_dataloader:
                input = input.to(device)
                target = target.to(device)
                output = model(input).squeeze()

                step_loss = loss_fn(output, target)

                optimizer.zero_grad()
                step_loss.backward()
                optimizer.step()

                acc = acc_metric(output, target)
                recall_metric(output, target)
                precision_metric(output, target)

                loss += step_loss * len(input)
                num_samples += len(input)
                if step_idx % log_interval == 0:
                    logs.append(
                        {
                            'epoch_idx': epoch_idx,
                            'step_idx': step_idx,
                            'step_loss': step_loss.item(),
                        }
                    )
                bar.set_postfix(loss=step_loss.item())

                bar.update(1)
                step_idx += 1

        log = {'epoch_num': epoch_idx, 'step_idx': step_idx, 'loss': (loss / num_samples).item()}

        acc = acc_metric.compute().item()
        acc_metric.reset()
        log['acc_train'] = acc

        recall = acc_metric.compute().item()
        recall_metric.reset()
        log['recall_train'] = recall

        precision = precision_metric.compute().item()
        precision_metric.reset()
        log['precision_train'] = precision

        # Validation
        outputs = []
        model.eval()
        with torch.no_grad():
            for input, target in val_dataloader:
                input = input.to(device)
                target = target.to(device)
                output = model(input).squeeze()

                acc_metric(output, target)
                recall_metric(output, target)
                precision_metric(output, target)

        acc = acc_metric.compute().item()
        acc_metric.reset()
        log['acc_val'] = acc

        recall = acc_metric.compute().item()
        recall_metric.reset()
        log['recall_val'] = recall

        precision = precision_metric.compute().item()
        precision_metric.reset()
        log['precision_val'] = precision

        logs.append(log)
