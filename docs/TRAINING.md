# Training and reproduction

`phononscore/training/train.py` is the actual geometry-MDN PhononScore trainer used
for the released checkpoints. Public-package import paths replace development
imports; the objectives, optimizer, model transfer and checkpoint selection follow
the original implementation.

## Build caches and train

Unpack `PhononScore_data_splits_20261104.zip`. Its root directory contains the
prepared training pools, raw labels and exact ordered split IDs. Then run:

```bash
bash scripts/train_model.sh pretrain /path/phononscore_data_20261104 outputs/pretrain
PRETRAIN_CKPT=outputs/pretrain/best_model.pt \
  bash scripts/train_model.sh dft_posttrain /path/phononscore_data_20261104 outputs/dft
```

Set `DEVICE=cpu` for CPU execution, `CUDA_VISIBLE_DEVICES` for the GPU, or
`PYTHON_BIN` for a Python executable. The defaults reproduce the original 100 epochs,
batch size 96, AdamW learning rate 0.0003 and weight decay 0.00001.
For a short installation check, `EPOCHS=1` changes only the number of epochs.
This is a smoke run and does not reproduce the paper checkpoint quality.

The builder reads fixed IDs using `--split_ids`. It preserves both membership and
record order when writing train/val/test LMDBs. It refuses to overwrite an existing
cache. Graph settings and the original ALIGNN architecture are in
`configs/training_pretrain.json` and `configs/training_dft_posttrain.json`.

The DFT stage warm-starts compatible weights from the provided PhononScore
checkpoint through the model's existing `load_alignn_atomwise_state` method.
By default the shell wrapper can use the bundled pretraining checkpoint; set
`PRETRAIN_CKPT` to a newly trained checkpoint to run both stages from scratch.

## Objectives and selection

The targets remain raw phonon minima in `id_prop.csv`. Only the regression loss
clips its targets to [-5, 0] THz. Threshold labels and geometry weights use the raw
minimum frequencies. The loss weights are:

| Term | Weight |
| --- | ---: |
| Smooth L1 regression | 1 |
| Class-balanced multi-threshold BCE | 0.3 |
| Threshold-aware score ranking | 0.3 |
| Ordinal monotonicity | 0.02 |
| Stable-weighted geometry-MDN likelihood | 0.05 |
| Geometry-score ranking | 0.05 |

Thresholds are -0.001, -0.01, -0.1 and -1.0 THz. Pair ranking samples up to 4096 pairs;
MDN uses hidden width 128, 10 Gaussians and distance scale 8.0 Angstrom. The geometry
stability weighting uses threshold -0.1 THz and temperature 0.2.
The internal training score uses beta=0.6 and alpha=0.1. These differ from the final
standardized inference weights beta=0.5 and alpha=0.2.

The best checkpoint minimizes total loss on internal checkpoint validation.
Its final internal `test_metrics.json` uses the historical mirrored validation
cache; it is not the external test metric. Use the released external manifests
for paper screening evaluation. No external test labels enter checkpoint or
fusion-weight selection in the revised evaluation procedure.

Training writes `best_model.pt`, `current_model.pt`, the full run configuration,
loss history and internal validation-alias predictions. Hardware/library changes
can affect numerical reproducibility even with the fixed random seed.

## Freeze moments for a retrained checkpoint

Run inference on that stage's training-pool CIFs with the raw model heads, then
freeze only records whose manifest role is `gradient_train`:

```bash
python scripts/freeze_training_statistics.py \
  --predictions raw_training_predictions.csv \
  --manifest /path/phononscore_data_20261104/pretrain/manifest.json \
  --model phononscore_pretrain --checkpoint outputs/pretrain/best_model.pt \
  --out outputs/pretrain/fixed_scoring.json
```

The CSV must contain the three raw components, and either `record_id` or a local
CIF identity from the same prepared stage directory as the supplied manifest.
Historical evaluation directories can reuse the same local filenames for different
structures; join those outputs to global `record_id` before fitting statistics.
When local IDs and raw labels are available, the script also checks their agreement.
Raw inference can be obtained with the Python API
`load_model`, `build_graph_record`, and `score_records` from
`phononscore.inference.score_cifs`; these functions do not standardize outputs.
The final CLI checks that the reference moments belong to the supplied checkpoint.
Pass the resulting reference file with `--scoring-config` when scoring a custom model.
Never fit means/variances on validation, test or query outputs. If weights are
reselected, use score validation and keep external test labels separate.

## External evaluation

```bash
python scripts/evaluate_benchmark.py \
  --manifest /path/phononscore_data_20261104/test_mp20/manifest.json \
  --predictions outputs/mp20_scores.csv --k 10,50,100 --out outputs/mp20_metrics.json
```

Metrics are computed per generator and averaged across generators for MP20.
DFT uses its single balanced pool. Stability is defined as minimum frequency > -0.1 THz.
