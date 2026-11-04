# Fixed scoring formula

For each raw component, z(x) = (x - mu_train) / sigma_train.
The population variance uses ddof=0 and sigma_train = sqrt(variance_train).
Divide by the standard deviation, not by the variance.

The final score is z(prediction) + 0.5*z(threshold_score) + 0.2*z(pair_geometry_score).
The checkpoint internal raw score remains prediction + 0.6*threshold_score + 0.1*pair_geometry_score.

## Frozen reference moments

| Model | Actual training records | Component | Mean | Variance | Standard deviation |
| --- | ---: | --- | ---: | ---: | ---: |
| phononscore_pretrain | 126,719 | prediction | -0.904731979918483 | 1.83500784703738 | 1.35462461480566 |
| phononscore_pretrain | 126,719 | threshold_score | 0.471143072525874 | 0.145623628229141 | 0.381606640703672 |
| phononscore_pretrain | 126,719 | pair_geometry_score | -0.542678372444163 | 1.34155785761599 | 1.15825638682288 |
| phononscore_dft | 7,809 | prediction | -0.205946621588344 | 0.42256757523871 | 0.650051978874543 |
| phononscore_dft | 7,809 | threshold_score | 0.715006522277293 | 0.0940505425726221 | 0.306676609105784 |
| phononscore_dft | 7,809 | pair_geometry_score | -0.338174456197299 | 0.20873352952907 | 0.456873647225433 |

Full precision and checkpoint hashes are in `configs/fixed_scoring.json`.
Only actual gradient-training predictions enter these moments. Internal checkpoint
validation, score validation, external tests and user queries do not contribute.
Each model uses its own frozen training-output moments.

## Interpretation

The same structure has the same score when submitted alone, in another candidate
pool, or in a different inference batch, up to floating-point forward-pass noise.
Scores can therefore be compared across pools for the same checkpoint/reference.
They are ranking values, not stability probabilities; zero is not a universal
stability threshold. Scores from different checkpoints are not jointly calibrated.

Rank-percentile plots in the paper are visualization transforms only and do not
modify this fixed scoring definition.

## Custom checkpoints

The CLI verifies the checkpoint SHA256 against its reference. For a retrained
checkpoint, use its raw gradient-training predictions with
`scripts/freeze_training_statistics.py` and pass the matching file using
`--scoring-config`. Keep any new fusion-weight selection on validation data.
See docs/TRAINING.md for the full workflow.
