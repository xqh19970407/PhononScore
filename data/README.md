# Paper data assignments

Download [PhononScore_split_ids_20261104.xlsx](PhononScore_split_ids_20261104.xlsx)
for all current training, checkpoint-validation, score-validation and external-test IDs.
It includes 150,610 distinct records, identified by global record_id and subset-local CIF ID.

Frozen internal ID lists, benchmark IDs, counts and the formula-overlap audit are
in [splits](splits). See [Data splits](../docs/DATA_SPLITS.md) for their distinct roles.
Means, population variances and standard deviations are in the workbook and in
[fixed_scoring.json](../configs/fixed_scoring.json); JSON retains full numerical precision.

The portable CIF archive is named `PhononScore_data_splits_20261104.zip`.
Its archival download URL is pending the authors' link. It contains the same workbook,
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
