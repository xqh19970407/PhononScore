#!/usr/bin/env python
"""Freeze checkpoint-specific moments from gradient-training predictions only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phononscore.scoring import COMPONENTS, DEFAULT_CONFIG, load_scoring_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='The released stage manifest containing record_id, id, and role.')
    parser.add_argument('--model', choices=['phononscore_pretrain', 'phononscore_dft'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--base-config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    train = [r for r in manifest if r['role'] == 'gradient_train']
    with args.predictions.open(newline='') as stream:
        reader = csv.DictReader(stream)
        key = 'record_id' if 'record_id' in reader.fieldnames else ('cif_name' if 'cif_name' in reader.fieldnames else 'id')
        predictions = list(reader)
    def identity(row, *, manifest=False):
        if key == 'record_id':
            return row['record_id']
        name = Path(row['id'] if manifest else row[key]).name
        return name if name.endswith('.cif') else name + '.cif'
    ids = {identity(r, manifest=True) for r in train}
    if len(ids) != len(train):
        raise ValueError('Training identities are not unique')
    rows = [r for r in predictions if identity(r) in ids]
    if len(rows) != len(ids) or {identity(r) for r in rows} != ids:
        raise ValueError('Predictions must contain exactly one row for every gradient-training record')
    if key != 'record_id':
        by_id = {identity(r, manifest=True): r for r in train}
        for row in rows:
            label = row.get('min_frequency', row.get('target'))
            if label is not None and not math.isclose(float(label), float(by_id[identity(row)]['min_frequency']), rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError('Local IDs/labels do not match the supplied manifest; join historical results by record_id first.')
    components = {}
    for component in COMPONENTS:
        values = [float(r[component]) for r in rows]
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f'Non-finite {component}')
        mean = math.fsum(values) / len(values)
        variance = math.fsum((x - mean) ** 2 for x in values) / len(values)
        if variance <= 0:
            raise ValueError(f'Cannot standardize a constant component: {component}')
        components[component] = {'mean': mean, 'variance': variance, 'std': math.sqrt(variance)}
    config = load_scoring_config(args.base_config)
    config['models'][args.model] = {
        'training_stage': 'pretrain' if args.model == 'phononscore_pretrain' else 'dft_posttrain',
        'n_train': len(rows), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'components': components,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + '\n')
    print(f'Frozen {len(rows)} training records for {args.model}: {args.out}')


if __name__ == '__main__':
    main()
