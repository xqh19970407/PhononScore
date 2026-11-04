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
