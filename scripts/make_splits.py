"""Split data into train, test, and validation sets."""

import random
import tomllib
from itertools import groupby
from pathlib import Path

import pandas as pd


def group_split(groups, test_ratio, val_ratio, seed=42):
    random.seed(seed)

    # Shuffle to avoid bias when sizes are equal
    random.shuffle(groups)

    # Sort largest first
    group = sorted(groups, key=lambda x: -len(x[1]))

    # Make splits
    total = sum(len(group) for _, group in groups)
    split_sets = [[], [], []]
    split_counts = [0, 0, 0]
    targets = [1 - test_ratio - val_ratio, test_ratio, val_ratio]
    for _, group in groups:
        # Assign to whichever split is most below its target
        deltas = [targets[i] - split_counts[i] / total for i in range(3)]
        split_idx = max(range(3), key=lambda x: deltas[x])
        split_sets[split_idx].extend(group)
        split_counts[split_idx] += len(group)

    return split_sets


high_poly_path = Path('data/raw/high_polyreactivity_high_throughput.csv')
low_poly_path = Path('data/raw/low_polyreactivity_high_throughput.csv')
cluster_path = Path('data/processed/clustered/clusters_85.tsv')
output_dir = Path('data/processed/splits')

seed = 42
test_ratio = 0.1
val_ratio = 0.1

if __name__ == '__main__':
    # Make splits
    with open(cluster_path) as file:
        records = map(lambda x: x.rstrip('\n').split('\t'), file.readlines())

    groups = []
    for key, group in groupby(records, lambda x: x[0]):
        groups.append((key, [row[1] for row in group]))

    train_split, test_split, val_split = group_split(groups, 0.1, 0.1, 42)

    # Load and format outputs
    with open('config.toml', 'rb') as file:
        config = tomllib.load(file)

    high_cols = config['constants']['imgt_columns']['high_polyreactivity']
    low_cols = config['constants']['imgt_columns']['low_polyreactivity']
    all_cols = config['constants']['imgt_columns']['all_polyreactivity']
    assert set(all_cols) == (set(high_cols) | set(low_cols))

    high_poly = pd.read_csv(high_poly_path, usecols=['Id'] + high_cols)
    low_poly = pd.read_csv(low_poly_path, usecols=['Id'] + low_cols)

    high_poly['split_id'] = high_poly['Id'] + '|high-poly'
    low_poly['split_id'] = low_poly['Id'] + '|low-poly'

    all_poly = pd.concat([high_poly, low_poly]).drop('Id', axis=1).set_index('split_id').fillna('-')
    all_poly = all_poly[all_cols]

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    train_poly = all_poly.loc[train_split]
    train_poly.to_csv(output_dir / 'train.csv')

    test_poly = all_poly.loc[test_split]
    test_poly.to_csv(output_dir / 'test.csv')

    val_poly = all_poly.loc[val_split]
    val_poly.to_csv(output_dir / 'val.csv')
