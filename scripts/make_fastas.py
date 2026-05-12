"""Make FASTAs of sequences for clustering."""

import tomllib
from pathlib import Path

import pandas as pd


def write_fasta(path, records, max_width=80):
    with open(path, 'w') as file:
        for header, seq in records:
            seqlines = '\n'.join(seq[i : i + max_width] for i in range(0, len(seq), max_width))
            file.write(f'>{header}\n{seqlines}\n')


high_poly_path = Path('data/raw/high_polyreactivity_high_throughput.csv')
low_poly_path = Path('data/raw/low_polyreactivity_high_throughput.csv')
output_dir = Path('data/processed/fastas')

if __name__ == '__main__':
    with open('config.toml', 'rb') as file:
        config = tomllib.load(file)

    output_dir.mkdir(parents=True, exist_ok=True)

    high_cols = config['constants']['imgt_columns']['high_polyreactivity']
    low_cols = config['constants']['imgt_columns']['low_polyreactivity']

    high_poly = pd.read_csv(high_poly_path, usecols=['Id'] + high_cols)
    low_poly = pd.read_csv(low_poly_path, usecols=['Id'] + low_cols)

    # high, aligned
    records = []
    seqs = set()
    for row in high_poly.itertuples(index=False):
        header = f'{row[0]}|high-poly'
        seq = ''.join(row[1:])
        if seq in seqs:
            continue
        records.append((header, seq))
        seqs.add(seq)
    write_fasta(output_dir / 'high.afa', records)

    # high, unaligned
    records = []
    for row in high_poly.itertuples(index=False):
        header = f'{row[0]}|high-poly'
        seq = ''.join(row[1:]).replace('-', '')
        if seq in seqs:
            continue
        records.append((header, seq))
        seqs.add(seq)
    write_fasta(output_dir / 'high.fa', records)

    # low, aligned
    records = []
    for row in low_poly.itertuples(index=False):
        header = f'{row[0]}|low-poly'
        seq = ''.join(row[1:])
        if seq in seqs:
            continue
        records.append((header, seq))
        seqs.add(seq)
    write_fasta(output_dir / 'low.afa', records)

    # low, unaligned
    records = []
    for row in low_poly.itertuples(index=False):
        header = f'{row[0]}|low-poly'
        seq = ''.join(row[1:]).replace('-', '')
        if seq in seqs:
            continue
        records.append((header, seq))
        seqs.add(seq)
    write_fasta(output_dir / 'low.fa', records)
