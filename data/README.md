# Paper data assignments

Download [PhononScore_split_ids_20261104.xlsx](PhononScore_split_ids_20261104.xlsx)
for all current training, checkpoint-validation, score-validation and external-test IDs.
It includes 150,610 distinct records, identified by global record_id and subset-local CIF ID.

Frozen internal ID lists, benchmark IDs, counts and the formula-overlap audit are
in [splits](splits). See [Data splits](../docs/DATA_SPLITS.md) for their distinct roles.
Means, population variances and standard deviations are in the workbook and in
[fixed_scoring.json](../configs/fixed_scoring.json); JSON retains full numerical precision.

The portable CIF archive is named `PhononScore_data_splits_20261104.zip`.
[Download the complete ZIP](https://media.githubusercontent.com/media/xqh19970407/PhononScore/main/data/PhononScore_data_splits_20261104.zip) (140.9 MiB; Git LFS). It contains the same workbook,
all five dataset manifests, raw labels, frozen IDs and all 150,610 referenced CIFs.

## CIFs and minimum-frequency labels

CIF files contain the crystal structures. Their reference minimum phonon frequencies
are supplied as separate labels in THz, with one label per CIF:

- `<subset>/manifest.json`: `id` / `cif_path` identify the CIF; `min_frequency` is its label.
- `<subset>/id_prop.csv`: first column is the local CIF path, second is the minimum frequency.
- The Excel workbook and `record_assignments.csv` contain the same mapping with global `record_id` values.

Resolve each CIF path relative to its subset directory. Reused local filenames in
different subsets identify different structures; use `record_id` when joining subsets.
Pretraining and MP20 labels come from MatterSim; DFT post-training and DFT-test labels
come from DFT-PBE phonon calculations. Original frequencies are retained; regression
target clipping is applied by the training loss rather than to the archived labels.

## Fixed screening pools and scores

The two 5% stable candidate pools used for the ROC/PR/recall figures are recorded in
[screening_pools_20261105](screening_pools_20261105): 3,200 MP20 structures and
500 DFT-PBE structures. Candidate CSVs include CIF IDs and paths, labels, raw
model outputs, fixed scores, ranks and Top-1% screening flags. Separate CSVs list
all reference-stable targets and the actual materials selected by each model.
A smaller `cifs.zip` includes all 3,700 required structures.

See [Screening reproduction](../docs/SCREENING_REPRODUCTION.md) to recompute the
metrics and regenerate both figures directly from these frozen CSVs.

## Download verification

SHA256 of the complete ZIP:

```text
848f9a916bb75fff4e2911045e6d82c55c6765fd003758b2884602a58f0350ee
```

Use the direct dataset link above or `git lfs pull` to obtain the binary ZIP.
The Git history stores an LFS pointer for this file.
