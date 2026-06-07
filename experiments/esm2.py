"""Simple fine-tuning of ESM2 by training new prediction head."""

import tomllib
from functools import partial
from pathlib import Path

import mlflow
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


def train():
    step_idx = 0
    for epoch_idx in range(epoch_num):
        # Train epoch
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

                loss_metric(step_loss, len(input))
                acc_metric(output, target)
                recall_metric(output, target)
                precision_metric(output, target)

                if step_idx % log_interval == 0:
                    mlflow.log_metric('step_loss', step_loss.item(), step=step_idx)
                bar.set_postfix(loss=step_loss.item())

                bar.update(1)
                step_idx += 1

        loss = loss_metric.compute().item()
        loss_metric.reset()
        mlflow.log_metric('loss_train', loss, step=step_idx)

        acc = acc_metric.compute().item()
        acc_metric.reset()
        mlflow.log_metric('acc_train', acc, step=step_idx)

        recall = recall_metric.compute().item()
        recall_metric.reset()
        mlflow.log_metric('recall_train', recall, step=step_idx)

        precision = precision_metric.compute().item()
        precision_metric.reset()
        mlflow.log_metric('precision_train', precision, step=step_idx)

        # Validation
        outputs = []
        model.eval()
        with torch.no_grad():
            for input, target in val_dataloader:
                input = input.to(device)
                target = target.to(device)
                output = model(input).squeeze()

                step_loss = loss_fn(output, target)

                loss_metric(step_loss, len(input))
                acc_metric(output, target)
                recall_metric(output, target)
                precision_metric(output, target)

        loss = loss_metric.compute().item()
        loss_metric.reset()
        mlflow.log_metric('loss_val', loss, step=step_idx)

        acc = acc_metric.compute().item()
        acc_metric.reset()
        mlflow.log_metric('acc_val', acc, step=step_idx)

        recall = recall_metric.compute().item()
        recall_metric.reset()
        mlflow.log_metric('recall_val', recall, step=step_idx)

        precision = precision_metric.compute().item()
        precision_metric.reset()
        mlflow.log_metric('precision_val', precision, step=step_idx)


cluster_name = 'clusters_85'
train_path = Path(f'data/processed/splits/{cluster_name}/train.csv')
val_path = Path(f'data/processed/splits/{cluster_name}/val.csv')

id_col = 'split_id'
with open('config.toml', 'rb') as file:
    config = tomllib.load(file)
seq_cols = config['constants']['imgt_columns']['all_polyreactivity']

experiment_name = 'esm2_finetune_head'
run_name = 'initial_test'
esm2_trunk_name = 'esm2_t33_650M_UR50D'
batch_size = 128
epoch_num = 150
lr = 0.001
log_interval = 10

if __name__ == '__main__':
    esm2_trunk, alphabet = esm.pretrained.load_model_and_alphabet(esm2_trunk_name)

    model = ESM2Classifier(esm2_trunk)
    for param in model.esm2_trunk.parameters():
        param.requires_grad = False

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    loss_metric = torchmetrics.aggregation.MeanMetric().to(device)
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    recall_metric = torchmetrics.classification.BinaryRecall().to(device)
    precision_metric = torchmetrics.classification.BinaryPrecision().to(device)

    mlflow.set_experiment(experiment_name)
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(60)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag('cluster_name', cluster_name)
        mlflow.log_params(
            {
                'esm2_trunk_name': esm2_trunk_name,
                'batch_size': batch_size,
                'epoch_num': epoch_num,
                'lr': lr,
            }
        )

        train()

        mlflow.pytorch.log_model(
            pytorch_model=model.classifier_head, name=f'{experiment_name}-final'
        )
