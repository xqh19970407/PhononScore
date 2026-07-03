# Scoring Formula

For each CIF candidate, PhononScore outputs three score components:

```text
prediction
threshold_score
pair_geometry_score
```

The checkpoint also outputs a raw internal score:

```text
final_score_train = prediction + 0.6 * threshold_score + 0.1 * pair_geometry_score
```

For candidate-pool reranking, this release uses the standardized score used in the paper figures:

```text
score = z(prediction) + 2.0 * z(threshold_score) + 0.25 * z(pair_geometry_score)
```

where z-score normalization is computed within the candidate pool:

```text
z(x) = (x - mean(x)) / std(x)
```

The standardized score is meaningful for a candidate pool, not for a single isolated CIF.
