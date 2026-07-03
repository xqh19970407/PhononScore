# Open Source Checklist

## Completed Locally

- Created a standalone release directory: `PhononScore-release`.
- Bundled the current ALIGNN source under `third_party/alignn-main`.
- Added PhononScore model code under `phononscore/models`.
- Added a user-facing CIF scoring entry point:

```bash
python -m phononscore.inference.score_cifs
```

- Added two checkpoints:

```text
checkpoints/phononscore_pretrain/best_model.pt
checkpoints/phononscore_dft/best_model.pt
```

- Added checkpoint config files:

```text
phonon_score_geom_mdn_config.json
```

- Added example CIF files under `examples/cifs`.
- Added README, model card, scoring formula documentation, citation metadata, and MIT license.
- Exported the full working conda dependency stack from `my_alignn` to `environment.yml`.
- Renamed the public conda environment in `environment.yml` to `phononscore`.
- Preserved the raw local environment export at `docs/environment-my_alignn-raw.yml` for provenance.

## Verified Commands

CPU smoke test:

```bash
PYTHON_BIN=/home/xqhan/miniconda3/envs/my_alignn/bin/python bash scripts/smoke_test.sh
```

Direct PhononScore-DFT scoring:

```bash
/home/xqhan/miniconda3/envs/my_alignn/bin/python -m phononscore.inference.score_cifs \
  --input examples/cifs \
  --out examples/example_scores.csv \
  --model phononscore_dft \
  --device cpu
```

Direct PhononScore pretrained scoring:

```bash
/home/xqhan/miniconda3/envs/my_alignn/bin/python -m phononscore.inference.score_cifs \
  --input examples/cifs \
  --out examples/example_scores_pretrain.csv \
  --model phononscore_pretrain \
  --device cpu
```

Single-CIF scoring:

```bash
/home/xqhan/miniconda3/envs/my_alignn/bin/python -m phononscore.inference.score_cifs \
  --input examples/cifs/NdSe2_example.cif \
  --out examples/single_cif_score.csv \
  --model phononscore_dft \
  --device cpu
```

Expected behavior for a single CIF: `standardized_score` is `NA`, because z-score scoring requires a candidate pool.

## GitHub Repository

Target remote:

```text
https://github.com/xqh19970407/PhononScore.git
```

Suggested commands after local review:

```bash
cd /home/xqhan/InvDesFlow3.0/CrystalFomer-torch/PhononScore-release
git init
git branch -M main
git add .
git commit -m "Initial PhononScore release"
git remote add origin https://github.com/xqh19970407/PhononScore.git
git push -u origin main
```

The two model checkpoints are about 17 MB each, so they can be pushed directly to GitHub. If larger checkpoints are added later, use Git LFS or GitHub Releases.

## Before Public Announcement

- Recreate `phononscore` from `environment.yml` on a clean machine and rerun the CPU smoke test.
- Confirm the third-party ALIGNN license compatibility.
- Decide whether to keep bundled ALIGNN source or switch to a documented installation dependency.
- Add a formal paper citation once the PhononScore manuscript/preprint is available.
- Optionally add a small GitHub Actions smoke test that runs only import checks, not full DGL inference.
- Consider uploading checkpoints to a GitHub Release or Hugging Face if the repository grows.
