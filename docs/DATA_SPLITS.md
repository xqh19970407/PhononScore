# Frozen data assignments

The updated data package contains 150,610 unique `record_id` values and CIF files.
Alex20 is not included.

| Subset | Records | Role |
| --- | ---: | --- |
| pretrain | 133,389 | MatterSim training pool |
| dft_posttrain | 8,221 | DFT-PBE training pool |
| validation_mp20 | 1,600 | Score-weight validation: 200/source |
| test_mp20 | 6,400 | External benchmark: 800/source |
| test_dft_pbe | 1,000 | External DFT-PBE test: 500 stable / 500 unstable |

The validation split for fusion weights was fixed by reduced-formula groups with
seed 20261103. The selected weights are alpha=0.2, beta=0.5, using mean Top25/200
stable rate (81.0%) on MP20 validation. The initial 78-point grid was refined with
this additional alpha=0.2 point; it was not part of the initial grid.
The same weights are used for DFT-PBE without DFT test tuning.

The actual checkpoint training partitions are preserved exactly:

| Stage | gradient_train | checkpoint_validation | unused_rounding |
| --- | ---: | ---: | ---: |
| pretrain | 126,719 | 6,669 | 1 |
| dft_posttrain | 7,809 | 411 | 1 |

`checkpoint_validation` selects training checkpoints. `validation_mp20` selects
inference-score weights. They are separate datasets and must not be merged.
The original builder mirrors `id_val` to `id_test`; this internal trainer output
is an alias of checkpoint validation, not an independent external test.
The workbook flags that alias explicitly. One unused record per training pool
comes from independently rounding down the original 95% and 5% counts.

Reduced formulas in either external test or score-validation set do not occur in
either complete training pool, and score-validation formulas do not occur in
external tests. The original internal train/checkpoint-validation split is random
at the structure level and can contain repeated formulas across those two roles.
`data/splits/split_audit.json` reports the actual intersections.
The revised split preserves the previously evaluated pool and checkpoints; it does
not turn historical model-development data into a prospectively unseen benchmark.

## Workbook and manifests

`PhononScore_split_ids_20261104.xlsx` contains README, Summary, Pretraining,
DFT_posttraining, MP20_weight_validation, MP20_test, DFT_test, and Fixed_statistics.
`record_id` is globally unique. Local `id` values such as `cifs/000000.cif` are unique
only within their subset; joining unrelated subsets by that local ID is invalid.
Each dataset manifest includes both identities, formula, source, reference frequency,
computed stability label, split role and relative CIF path.

In the full ZIP, each subset contains `manifest.json`, `id_prop.csv` and `cifs/`.
Use CIF paths relative to that subset directory. The LMDB builder takes the exact
ordered ID lists in `splits/*_internal_ids.json`; it does not resample them.
The smaller metadata ZIP contains the workbook, manifests and assignments but
intentionally omits CIF contents. The full dataset includes all referenced CIFs.

The data-archive URL will be filled in after the authors upload the prepared ZIP.
No new download URL is invented here. Existing source data:
https://zenodo.org/records/21157982.
