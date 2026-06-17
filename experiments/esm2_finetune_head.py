"""Simple fine-tuning of ESM2 by training new prediction head."""

import tomllib
from functools import partial
from pathlib import Path
from argparse import ArgumentParser

import lightning as L
import mlflow
import torch
import torch.nn as nn
import torchmetrics

from torch.utils.data import DataLoader

import esm

from src.esm2 import esm_collate_generator, train
from src.utils import AlignedSequencesFromCSVDataset, TrainingMetrics


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
batch_size = 256
epoch_num = 150
lr = 0.001
log_interval = 10
system_metrics_interval = 60

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--devices', type=int, default=1)
    args = parser.parse_args()

    esm2_trunk, alphabet = esm.pretrained.load_model_and_alphabet(esm2_trunk_name)

    model = ESM2Classifier(esm2_trunk)
    for param in model.esm2_trunk.parameters():
        param.requires_grad = False

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

    fabric = L.Fabric(accelerator='gpu', num_nodes=args.num_nodes, devices=args.devices)
    fabric.launch()
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    loss_metric = torchmetrics.aggregation.MeanMetric().to(fabric.device)
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(fabric.device)
    recall_metric = torchmetrics.classification.BinaryRecall().to(fabric.device)
    precision_metric = torchmetrics.classification.BinaryPrecision().to(fabric.device)
    metrics = TrainingMetrics(
        loss=loss_metric,
        acc=acc_metric,
        recall=recall_metric,
        precision=precision_metric,
    )

    if fabric.global_rank == 0:
        mlflow.set_experiment(experiment_name)
        mlflow.enable_system_metrics_logging()
        mlflow.set_system_metrics_sampling_interval(system_metrics_interval)
        mlflow.start_run(run_name=run_name)
        mlflow.set_tag('cluster_name', cluster_name)
        mlflow.log_params(
            {
                'esm2_trunk_name': esm2_trunk_name,
                'batch_size': batch_size,
                'world_size': fabric.world_size,
                'epoch_num': epoch_num,
                'lr': lr,
            }
        )

    train(
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer,
        fabric,
        epoch_num,
        metrics,
        log_interval,
    )

    if fabric.global_rank == 0:
        mlflow.pytorch.log_model(
            pytorch_model=model.classifier_head, name=f'{experiment_name}-final'
        )
        mlflow.end_run()
