# Fixed screening pools and recorded scores

These files reproduce the two four-panel screening figures using the exact
candidate IDs and scores associated with the reported results.

| Pool CSV | Candidates | Stable targets | Primary screening budget | Reference labels |
| --- | ---: | ---: | ---: | --- |
| `mp20_05pct.csv` | 3,200 | 160 | Top-32 | MatterSim |
| `dft_05pct.csv` | 500 | 25 | Top-5 | DFT-PBE |

MP20 contains 400 structures from each of eight generation sources, including
20 stable structures per source. This figure ranks the combined pool globally.
All candidates belong to the released external test sets. Stability is defined
as `min_frequency > -0.1 THz`.

## Files

- `mp20_05pct.csv`, `dft_05pct.csv`: every candidate, its CIF identity, label,
  three model scores, ranks, and Top-1% selection/hit flags. The two PhononScore
  models also include their three raw outputs and fixed standardized components.
- `stable_targets.csv`: the 185 reference-stable structures present in the pools.
- `selected_candidates.csv`: the actual Top-32/Top-5 lists for each of the three
  models (111 model-specific selections). `stable` distinguishes retrieved
  reference-stable structures from unstable selections.
- `expected_metrics.csv`: AP, ROC-AUC, low-FPR recall, hits, precision, recall
  and enrichment for each pool/model combination.
- `cifs.zip`: all 3,700 CIF files, with separate directories for the two pools.
- `cifs.zip.sha256`: archive checksum. Each candidate CSV also records the
  SHA-256 of its individual CIF.
- `manifest.json`: frozen evaluation settings, file hashes, model identifiers
  and screening budgets.
- `pool_construction.json`: sampling seeds, draw indices and the procedure
  used to choose the displayed pools. These are fixed illustrative subsets;
  the recorded metrics describe these pools rather than an average over draws.

## Important columns

`record_id` is the global identity and `id` is the original local CIF ID. IDs
such as `cifs/000001.cif` can occur in both datasets, so retain `pool_id` when
combining the CSVs. `archive_path` maps a record to the full public dataset ZIP;
`pool_cif_path` maps it to this directory's smaller `cifs.zip`. `material_id`
provides the MP identifier where available.

`phononscore`, `phononscore_dft`, and `alignn` are the ranking scores (higher
first). `METHOD_rank` starts at one. `METHOD_selected_top1pct` marks the
model-selected candidates; `METHOD_stable_hit_top1pct` additionally requires
the reference stability label. These flags are different from `stable`, which
labels every reference-stable target, including those outside the top ranks.
`pool_index` preserves the original order and resolves exact score ties.

For each PhononScore model, `METHOD_raw_COMPONENT` and `METHOD_z_COMPONENT`
contain the raw and standardized regression, threshold and geometry outputs.
The score uses the checkpoint-specific training statistics in
`configs/fixed_scoring.json` and `z(regression) + 0.5*z(threshold) + 0.2*z(geometry)`.
ALIGNN uses the cached single-task baseline from the paper, with no additional
DFT fine-tuning.

From the repository root:

```bash
pip install -r requirements-screening.txt
python scripts/reproduce_screening_pools.py --extract-cifs outputs/screening_pool_cifs
```

See [Screening reproduction](../../docs/SCREENING_REPRODUCTION.md) for output
files, figure names and inference commands.
