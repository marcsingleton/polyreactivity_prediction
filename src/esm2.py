"""Functions for handling ESM2-based models."""

import mlflow
import torch
import tqdm


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


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    fabric,
    epoch_num,
    metrics,
    log_interval,
):
    step_idx = 0
    for epoch_idx in range(epoch_num):
        # Train step
        model.train()
        if fabric.global_rank == 0:
            bar = tqdm.trange(
                len(train_dataloader),
                desc=f'epoch {epoch_idx}',
                unit='batch',
                bar_format='{l_bar}{bar:10}{r_bar}',
            )
        for input, target in train_dataloader:
            output = model(input).squeeze()

            step_loss = loss_fn(output, target)

            optimizer.zero_grad()
            fabric.backward(step_loss)
            optimizer.step()

            metrics.loss(step_loss, len(input))
            metrics.acc(output, target)
            metrics.recall(output, target)
            metrics.precision(output, target)

            if step_idx % log_interval == 0 and fabric.global_rank == 0:
                mlflow.log_metric('step_loss', step_loss.item(), step=step_idx)
            if fabric.global_rank == 0:
                bar.set_postfix(loss=step_loss.item())
                bar.update(1)

            step_idx += 1
        if fabric.global_rank == 0:
            bar.close()

        # Train metrics
        loss = metrics.loss.compute().item()
        metrics.loss.reset()

        acc = metrics.acc.compute().item()
        metrics.acc.reset()

        recall = metrics.recall.compute().item()
        metrics.recall.reset()

        precision = metrics.precision.compute().item()
        metrics.precision.reset()

        if fabric.global_rank == 0:
            mlflow.log_metric('loss_train', loss, step=step_idx)
            mlflow.log_metric('acc_train', acc, step=step_idx)
            mlflow.log_metric('recall_train', recall, step=step_idx)
            mlflow.log_metric('precision_train', precision, step=step_idx)

        # Val batches
        model.eval()
        with torch.no_grad():
            for input, target in val_dataloader:
                output = model(input).squeeze()

                step_loss = loss_fn(output, target)

                metrics.loss(step_loss, len(input))
                metrics.acc(output, target)
                metrics.recall(output, target)
                metrics.precision(output, target)

        # Val metrics
        loss = metrics.loss.compute().item()
        metrics.loss.reset()

        acc = metrics.acc.compute().item()
        metrics.acc.reset()

        recall = metrics.recall.compute().item()
        metrics.recall.reset()

        precision = metrics.precision.compute().item()
        metrics.precision.reset()

        if fabric.global_rank == 0:
            mlflow.log_metric('loss_val', loss, step=step_idx)
            mlflow.log_metric('acc_val', acc, step=step_idx)
            mlflow.log_metric('recall_val', recall, step=step_idx)
            mlflow.log_metric('precision_val', precision, step=step_idx)
