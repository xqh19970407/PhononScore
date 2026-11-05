# Reproduce the fixed-pool screening figures

The exact two 5% stable candidate pools, labels, model scores and selected
material IDs are released in
[`data/screening_pools_20261105`](../data/screening_pools_20261105).
No resampling, training or normalization fitting is required to reproduce
the metrics and figures. Pool construction is recorded in
[`pool_construction.json`](../data/screening_pools_20261105/pool_construction.json).

## Recompute metrics and draw both figures

The analysis requires Python 3.10+ and only CPU dependencies. PyTorch, DGL
and the model checkpoints are needed only when rerunning CIF inference.

```bash
pip install -r requirements-screening.txt
python scripts/reproduce_screening_pools.py \
  --out outputs/screening_pools_20261105 \
  --extract-cifs outputs/screening_pool_cifs
```

The script verifies the released files and CIF contents, recomputes both
PhononScore variants from raw outputs using the frozen training statistics,
checks all ranks and selected IDs, and compares recomputed metrics with
`expected_metrics.csv`. It writes:

| Output | Contents |
| --- | --- |
| `MP20_05pct_ROC_ZOOM_PR_RECALL.pdf/png/svg` | MP20: ROC, low-FPR ROC, PR and early recall |
| `DFT_05pct_ROC_ZOOM_PR_RECALL.pdf/png/svg` | DFT-PBE: the same four panels |
| `primary_metrics.csv` | AP, ROC-AUC and Top-1% screening metrics |
| `topk_metrics.csv` | Hits, precision, recall and EF at additional budgets |
| `selected_candidates.csv` | Exact material IDs selected by each model |
| `roc_curves.csv`, `pr_curves.csv` | Every curve point and score threshold |
| `retrieval_curves.csv` | Hits and recall for every screening budget |
| `verification.json` | Results of the identity, score and metric checks |

Both figures have (A) full ROC curves, (B) FPR up to 0.05, (C) precision--recall
curves and (D) recall versus the percentage screened, with Top-1% marked.
The reference figures are also available in
[`figures/screening_pools_20261105`](../figures/screening_pools_20261105).

To reproduce only the DFT pool, or to skip plotting:

```bash
python scripts/reproduce_screening_pools.py --pool DFT_05pct --out outputs/dft_screening
python scripts/reproduce_screening_pools.py --metrics-only
```

## Reference numbers

| Pool | Method | Top-1% hits | Precision | Recall | EF | AP | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MP20, 160/3,200 stable | PhononScore | 24/32 | 0.750 | 0.150 | 15.000 | 0.368 | 0.850 |
| MP20, 160/3,200 stable | PhononScore-DFT | 16/32 | 0.500 | 0.100 | 10.000 | 0.198 | 0.759 |
| MP20, 160/3,200 stable | ALIGNN | 3/32 | 0.09375 | 0.01875 | 1.875 | 0.186 | 0.842 |
| DFT, 25/500 stable | PhononScore | 2/5 | 0.400 | 0.080 | 8.000 | 0.147 | 0.716 |
| DFT, 25/500 stable | PhononScore-DFT | 4/5 | 0.800 | 0.160 | 16.000 | 0.383 | 0.876 |
| DFT, 25/500 stable | ALIGNN | 0/5 | 0.000 | 0.000 | 0.000 | 0.112 | 0.714 |

AP is non-interpolated average precision (`average_precision_score`), not the
trapezoidal area of the PR plot. Recall@K is the number of stable hits divided
by the number of stable targets in that pool. EF@K is Precision@K divided by
the stable fraction of the pool (0.05). The candidate CSVs and Top-K files
retain the actual selected structures, including false positives.

These fixed-pool figures use MP20 global ranking across 3,200 candidates and
DFT global ranking across 500 candidates. They differ from the main benchmark's
Top-100 per generator and from the repeated 200-candidate hard-screening pools;
their denominators and reported metrics should be kept separate.

## CIF identities and inference

The smaller `cifs.zip` is an ordinary Git file and contains only the 3,700
structures needed here. After the extraction command above:

```text
outputs/screening_pool_cifs/MP20_05pct/cifs/*.cif
outputs/screening_pool_cifs/DFT_05pct/cifs/*.cif
```

The same CIF bytes are present in the complete public dataset under each
row's `archive_path`. `cif_sha256` verifies the one-to-one correspondence.

With the full model environment installed, rerun either released model on a
pool using the standard inference entry point, for example:

```bash
python -m phononscore.inference.score_cifs \
  --input outputs/screening_pool_cifs/DFT_05pct/cifs \
  --out outputs/dft_pool_phononscore_dft.csv \
  --model phononscore_dft --device cpu --batch-size 32
```

Use `--model phononscore_pretrain` for PhononScore. Run the same commands on
the MP20 directory to score that pool. Match the inference `cif_name` to the
basename of the pool CSV's `id`; compare `standardized_score` to the pool's
corresponding `phononscore` or `phononscore_dft` column. The exact output can
vary slightly with floating-point arithmetic, hardware and library versions.
The recorded raw outputs and scores reproduce the displayed metrics directly.

Run the data/score/identity regression checks with:

```bash
python -m unittest discover -s tests -v
```
