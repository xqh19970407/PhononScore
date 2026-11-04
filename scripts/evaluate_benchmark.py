#!/usr/bin/env python
"""Evaluate frozen scores against a portable external-test manifest."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def local_identity(value):
    name = Path(value).name
    return name if name.endswith('.cif') else name + '.cif'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--k', default='10,50,100')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    with args.predictions.open(newline='') as stream:
        predictions = list(csv.DictReader(stream))
    by_record = all(r.get('record_id') for r in predictions)
    def identity(row):
        return row['record_id'] if by_record else local_identity(row.get('cif_name') or row['id'])
    by_id = {identity(r): r for r in predictions}
    if len(by_id) != len(predictions) or {identity(r) for r in manifest} != set(by_id):
        raise ValueError('Prediction IDs must match the manifest one-to-one')
    groups = defaultdict(list)
    for record in manifest:
        score = float(by_id[identity(record)]['standardized_score'])
        if not math.isfinite(score):
            raise ValueError('Scores must be finite')
        groups[record['model_key']].append((score, record['id'], float(record['min_frequency']) > -0.1))
    cutoffs = [int(x) for x in args.k.split(',')]
    if not cutoffs or min(cutoffs) <= 0 or max(cutoffs) > min(map(len, groups.values())):
        raise ValueError('Each cutoff must be positive and no larger than each source pool')
    result = {'n': len(manifest), 'threshold_THz': -0.1, 'stability_rule': 'min_frequency > threshold', 'by_source': {}}
    for name, rows in groups.items():
        ranked = sorted(rows, key=lambda r: (-r[0], r[1]))
        baseline = sum(r[2] for r in rows) / len(rows)
        metrics = {'n': len(rows), 'pool_stable_fraction': baseline}
        for k in cutoffs:
            rate = sum(r[2] for r in ranked[:k]) / k
            metrics[f'top{k}_stable_fraction'] = rate
            metrics[f'top{k}_enrichment'] = rate / baseline if baseline else None
        result['by_source'][name] = metrics
    result['macro_average'] = {f'top{k}_stable_fraction': sum(m[f'top{k}_stable_fraction'] for m in result['by_source'].values()) / len(groups) for k in cutoffs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['macro_average'], indent=2))


if __name__ == '__main__':
    main()
