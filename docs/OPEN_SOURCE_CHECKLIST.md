# Release contents and verification

This update adds the paper trainer, fixed-ID LMDB builder, portable data assignments,
checkpoint-specific training normalization, and single-CIF fixed scoring.

- Models and original pretrained/DFT checkpoints are retained.
- `phononscore/training/train.py` exposes the original multi-task objectives.
- `scripts/train_model.sh` reproduces the two training stages with portable paths.
- `data/splits` distinguishes model validation, score-weight validation and external tests.
- `configs/fixed_scoring.json` records means, population variances/stds and checkpoint hashes.
- Single-CIF and multi-CIF inputs use the same frozen score definition.
- `tests/test_scoring.py` checks recorded paper predictions and pool invariance.
- The split workbook is included under `data/`; the full CIF ZIP is prepared
  separately for archival upload, with its URL pending the author's supplied link.

The inference revision changes standardized scores from the initial public version,
which normalized within the query pool with alpha=0.25, beta=2.0. Current defaults
use fixed training moments with alpha=0.2, beta=0.5. The raw checkpoints and heads
are unchanged. See docs/SCORING_FORMULA.md and docs/DATA_SPLITS.md.
