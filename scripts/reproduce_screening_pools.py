#!/usr/bin/env python
"""Recompute fixed-pool screening metrics, selected IDs, curves and figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from phononscore.screening import (DEFAULT_DATA, METHODS, check_reference, evaluate_pool,
                                  load_manifest, load_pool, verify_and_extract_cifs)

COLORS = {'phononscore': '#98DCFF', 'phononscore_dft': '#2F80ED', 'alignn': '#D98A36'}
LABELS = {'phononscore': 'PhononScore', 'phononscore_dft': 'PhononScore-DFT', 'alignn': 'ALIGNN'}


def render(spec, metrics, curves, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                        'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
                        'xtick.major.size': 3, 'ytick.major.size': 3,
                        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 450})
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.65))
    point = metrics.set_index('method')
    prior = spec['stable_count']/spec['n']
    for ax in axes[0]:
        ax.plot([0, 1], [0, 1], color='#A8A8A8', linestyle=(0, (4, 2)), linewidth=1.1, label='Random')
    axes[1, 0].axhline(prior, color='#A8A8A8', linestyle=(0, (4, 2)), linewidth=1.1, label=f'Random ({prior:.2f})')
    limit = round(.1*spec['n'])
    axes[1, 1].plot(np.arange(limit+1)*100/spec['n'], np.arange(limit+1)/spec['n'],
                    color='#A8A8A8', linestyle=(0, (4, 2)), linewidth=1.1, label='Random')
    for method in reversed(METHODS):
        v = curves[method]
        for ax in axes[0]:
            line, = ax.plot(v['fpr'], v['tpr'], color=COLORS[method], linewidth=1.6,
                            label=f'{LABELS[method]} ({point.loc[method, "roc_auc"]:.3f})')
            if method == 'phononscore':
                line.set_path_effects([pe.Stroke(linewidth=2.1, foreground='#6B7280', alpha=.35), pe.Normal()])
        line, = axes[1, 0].step(v['recall'], v['precision'], where='post', color=COLORS[method], linewidth=1.6,
                                label=f'{LABELS[method]} ({point.loc[method, "ap"]:.3f})')
        if method == 'phononscore':
            line.set_path_effects([pe.Stroke(linewidth=2.1, foreground='#6B7280', alpha=.35), pe.Normal()])
        line, = axes[1, 1].step(v['screened_fraction'][:limit+1]*100, v['retrieval_recall'][:limit+1], where='post',
                                color=COLORS[method], linewidth=1.6, label=LABELS[method])
        if method == 'phononscore':
            line.set_path_effects([pe.Stroke(linewidth=2.1, foreground='#6B7280', alpha=.35), pe.Normal()])
    for col, ax in enumerate(axes[0]):
        ax.set(xlim=(0, .05 if col else 1), xlabel='False-positive rate', ylabel='True-positive rate')
        ax.set_xticks([0, .01, .02, .03, .04, .05] if col else [0, .2, .4, .6, .8, 1])
        ax.legend(title='AUROC', title_fontsize=7, frameon=False, loc='upper left' if col else 'lower right', handlelength=1.4)
    axes[1, 0].set(xlim=(0, 1), xlabel='Recall', ylabel='Precision')
    axes[1, 0].set_xticks([0, .2, .4, .6, .8, 1])
    axes[1, 0].legend(title='AP', title_fontsize=7, frameon=False, loc='upper right', handlelength=1.4)
    axes[1, 1].axvline(1, linestyle=':', color='#3A3A3A', linewidth=.8)
    axes[1, 1].set(xlim=(0, 10), xlabel='Candidates screened (%)', ylabel='Recall')
    axes[1, 1].legend(frameon=False, loc='upper left', handlelength=1.4)
    for ax, letter in zip(axes.flat, 'ABCD'):
        ax.set_ylim(0, 1.025)
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='both', color='#E6E6E6', linewidth=.55)
        ax.set_axisbelow(True)
        ax.text(-.17, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight='bold')
    domain = spec['pool_id'].split('_')[0]
    fig.suptitle(f'{domain}: {spec["n"]} candidates, 5% stable', fontsize=9, y=1.00)
    fig.subplots_adjust(wspace=.35, hspace=.35, top=.93)
    for suffix in ['pdf', 'svg', 'png']:
        path = out / f'{spec["figure_stem"]}.{suffix}'
        fig.savefig(path, bbox_inches='tight')
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--out', type=Path, default=REPO / 'outputs/screening_pools_20261105')
    parser.add_argument('--pool', choices=['all', 'MP20_05pct', 'DFT_05pct'], default='all')
    parser.add_argument('--extract-cifs', type=Path, help='Also extract the pool CIFs beneath this directory')
    parser.add_argument('--metrics-only', action='store_true', help='Skip PDF/SVG/PNG generation')
    args = parser.parse_args()
    manifest = load_manifest(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    frames, metrics, topks, chosen, retrievals, rocs, prs = [], [], [], [], [], [], []
    for spec in manifest['pools']:
        if args.pool != 'all' and args.pool != spec['pool_id']:
            continue
        frame = load_pool(args.data, spec)
        metric, topk, curves, selections, retrieval = evaluate_pool(frame, spec)
        check_reference(metric, args.data)
        frames.append(frame)
        metrics.append(metric)
        topks.append(topk)
        chosen.append(selections)
        retrievals.append(retrieval)
        for method, values in curves.items():
            for fpr, tpr, threshold in zip(values['fpr'], values['tpr'], values['roc_thresholds']):
                rocs.append({'pool_id': spec['pool_id'], 'method': method, 'fpr': fpr, 'tpr': tpr, 'score_threshold': threshold})
            for i, (recall, precision) in enumerate(zip(values['recall'], values['precision'])):
                prs.append({'pool_id': spec['pool_id'], 'method': method, 'recall': recall, 'precision': precision,
                            'score_threshold': values['pr_thresholds'][i] if i < len(values['pr_thresholds']) else ''})
        if not args.metrics_only:
            render(spec, metric, curves, args.out)
    verify_and_extract_cifs(args.data, frames, args.extract_cifs)
    for name, values in [('primary_metrics', metrics), ('topk_metrics', topks), ('selected_candidates', chosen), ('retrieval_curves', retrievals)]:
        pd.concat(values, ignore_index=True).to_csv(args.out / f'{name}.csv', index=False)
    pd.DataFrame(rocs).to_csv(args.out / 'roc_curves.csv', index=False)
    pd.DataFrame(prs).to_csv(args.out / 'pr_curves.csv', index=False)
    verification = {'pool_ids': [str(f.pool_id.iloc[0]) for f in frames],
                    'candidate_records': sum(map(len, frames)), 'cif_checksums_verified': True,
                    'raw_heads_reproduce_fixed_scores': True, 'ranks_and_selected_ids_verified': True,
                    'metrics_match_recorded_values': True, 'resampling': False,
                    'normalization_refit': False}
    (args.out / 'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(pd.concat(metrics).to_string(index=False))
    print(f'Wrote verified metrics, selected IDs and curves to {args.out}')


if __name__ == '__main__':
    main()
