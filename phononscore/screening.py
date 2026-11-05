"""Validate and evaluate the frozen candidate pools used in the screening figures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from phononscore.scoring import add_fixed_scores, load_scoring_config

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / 'data/screening_pools_20261105'
METHODS = ('phononscore', 'phononscore_dft', 'alignn')
COMPONENTS = ('prediction', 'threshold_score', 'pair_geometry_score')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(data_dir: Path = DEFAULT_DATA) -> dict:
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    for filename, expected in manifest['files_sha256'].items():
        path = data_dir / filename
        if path.parent != data_dir or sha256(path) != expected:
            raise ValueError(f'Released artifact checksum mismatch: {filename}')
    if sha256(REPO / manifest['scoring_config']) != manifest['scoring_config_sha256']:
        raise ValueError('Fixed scoring configuration differs from the figure release')
    return manifest


def load_pool(data_dir: Path, spec: dict) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / spec['csv'], keep_default_na=False)
    return validate_pool(frame, spec)


def validate_pool(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Use explicit pool_index for deterministic ordering and score ties."""
    frame = frame.sort_values('pool_index', kind='stable').reset_index(drop=True).copy()
    if len(frame) != spec['n'] or not np.array_equal(frame.pool_index, np.arange(1, len(frame)+1)):
        raise ValueError('Pool size or pool_index differs from the released pool')
    for column in ['record_id', 'id', 'pool_cif_path']:
        if not frame[column].is_unique:
            raise ValueError(f'Duplicate {column} in candidate pool')
    if not frame.pool_id.eq(spec['pool_id']).all():
        raise ValueError('Pool identity mismatch')
    if not frame.stable.isin([0, 1]).all() or not np.array_equal(frame.stable.astype(bool), frame.min_frequency > -.1):
        raise ValueError('Stability labels must use min_frequency > -0.1 THz')
    if int(frame.stable.sum()) != spec['stable_count']:
        raise ValueError('Stable count differs from the released pool')
    if not np.isfinite(frame[list(METHODS)+['min_frequency']].to_numpy(dtype=float)).all():
        raise ValueError('Scores and minimum frequencies must be finite')
    if not frame.label_source.eq(spec['reference_label']).all() or not frame.primary_k.eq(spec['primary_k']).all():
        raise ValueError('Reference labels or primary screening budget differ')
    if spec['pool_id'].startswith('MP20'):
        grouped = frame.groupby('model_key').agg(n=('id', 'size'), positives=('stable', 'sum'))
        if len(grouped) != 8 or not grouped.n.eq(400).all() or not grouped.positives.eq(20).all():
            raise ValueError('MP20 must contain 400 candidates and 20 stable structures per source')
    config = load_scoring_config()
    for method, model_name in [('phononscore', 'phononscore_pretrain'), ('phononscore_dft', 'phononscore_dft')]:
        rows = [{c: float(row[f'{method}_raw_{c}']) for c in COMPONENTS} for row in frame.to_dict('records')]
        add_fixed_scores(rows, model_name=model_name, config=config)
        if not np.allclose(frame[method], [r['standardized_score'] for r in rows], atol=3e-12, rtol=0):
            raise ValueError(f'{method} scores differ from the frozen raw-head combination')
        for component in COMPONENTS:
            if not np.allclose(frame[f'{method}_z_{component}'], [r[component+'_z'] for r in rows], atol=3e-12, rtol=0):
                raise ValueError(f'{method} standardized {component} differs from the fixed reference')
    for method in METHODS:
        order = descending_order(frame, method)
        ranks = np.empty(len(frame), dtype=int)
        ranks[order] = np.arange(1, len(frame)+1)
        chosen = ranks <= spec['primary_k']
        if not np.array_equal(frame[f'{method}_rank'], ranks):
            raise ValueError(f'{method} recorded ranks do not match scores')
        if not np.array_equal(frame[f'{method}_selected_top1pct'].astype(bool), chosen):
            raise ValueError(f'{method} selected IDs do not match the screening budget')
        if not np.array_equal(frame[f'{method}_stable_hit_top1pct'].astype(bool), chosen & frame.stable.astype(bool)):
            raise ValueError(f'{method} recorded hit IDs do not match reference stability labels')
    return frame


def descending_order(frame: pd.DataFrame, method: str) -> np.ndarray:
    return np.lexsort((frame.pool_index.to_numpy(), -frame[method].to_numpy()))


def evaluate_pool(frame: pd.DataFrame, spec: dict):
    metrics, topk, curves, selections, retrieval = [], [], {}, [], []
    y = frame.stable.to_numpy(dtype=bool)
    n, npos = len(y), int(y.sum())
    for method in METHODS:
        score = frame[method].to_numpy()
        order = descending_order(frame, method)
        hits = np.r_[0, np.cumsum(y[order])]
        fpr, tpr, thresholds = roc_curve(y, score, drop_intermediate=False)
        precision, recall, pr_thresholds = precision_recall_curve(y, score)
        curves[method] = {'fpr': fpr, 'tpr': tpr, 'roc_thresholds': thresholds,
                          'precision': precision, 'recall': recall, 'pr_thresholds': pr_thresholds,
                          'screened_fraction': np.arange(n+1)/n, 'retrieval_recall': hits/npos}
        for k in spec['ks']:
            h = int(hits[k])
            topk.append({'pool_id': spec['pool_id'], 'method': method, 'k': k, 'hits': h,
                         'precision': h/k, 'recall': h/npos, 'enrichment': (h/k)/(npos/n),
                         'random_expected_hits': k*npos/n})
        k = spec['primary_k']
        metrics.append({'pool_id': spec['pool_id'], 'method': method, 'n': n, 'stable_count': npos,
                        'stable_fraction': npos/n, 'primary_k': k,
                        'ap': float(average_precision_score(y, score)), 'roc_auc': float(roc_auc_score(y, score)),
                        'tpr_at_fpr0p01': float(np.interp(.01, fpr, tpr)),
                        'hits': int(hits[k]), 'precision': float(hits[k]/k),
                        'recall': float(hits[k]/npos), 'enrichment': float((hits[k]/k)/(npos/n))})
        for i in order[:k]:
            row = frame.iloc[i]
            selections.append({key: row[key] for key in ['pool_id', 'pool_index', 'record_id', 'id', 'material_id', 'formula_reduced', 'label_source', 'min_frequency', 'stable', 'pool_cif_path']} |
                              {'method': method, 'rank': int(row[f'{method}_rank']), 'score': float(score[i]), 'primary_k': k})
        for k in range(n+1):
            retrieval.append({'pool_id': spec['pool_id'], 'method': method, 'k': k,
                              'screened_fraction': k/n, 'hits': int(hits[k]), 'recall': hits[k]/npos,
                              'precision': hits[k]/k if k else '',
                              'enrichment': (hits[k]/k)/(npos/n) if k else ''})
    return pd.DataFrame(metrics), pd.DataFrame(topk), curves, pd.DataFrame(selections), pd.DataFrame(retrieval)


def check_reference(metrics: pd.DataFrame, data_dir: Path):
    expected = pd.read_csv(data_dir / 'expected_metrics.csv').set_index(['pool_id', 'method'])
    actual = metrics.set_index(['pool_id', 'method'])
    reference = expected.loc[actual.index]
    columns = ['n', 'stable_count', 'stable_fraction', 'primary_k', 'ap', 'roc_auc', 'tpr_at_fpr0p01', 'hits', 'precision', 'recall', 'enrichment']
    if not np.allclose(actual[columns], reference[columns], atol=1e-12, rtol=0):
        raise ValueError('Recomputed metrics differ from the recorded figure values')


def verify_and_extract_cifs(data_dir: Path, frames: list[pd.DataFrame], destination: Path | None = None):
    """Check each CIF against its CSV identity; optionally extract these exact members."""
    with ZipFile(data_dir / 'cifs.zip') as archive:
        for frame in frames:
            for row in frame.itertuples(index=False):
                member = Path(row.pool_cif_path)
                if member.is_absolute() or '..' in member.parts:
                    raise ValueError('Invalid relative CIF member path')
                contents = archive.read(member.as_posix())
                if hashlib.sha256(contents).hexdigest() != row.cif_sha256:
                    raise ValueError(f'CIF checksum differs: {row.pool_cif_path}')
                if destination is not None:
                    target = destination / member
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(contents)
