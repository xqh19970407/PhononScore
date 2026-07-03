# Checkpoints

This directory contains two PhononScore checkpoints with the periodic geometry-likelihood branch:

```text
phononscore_pretrain/best_model.pt
phononscore_dft/best_model.pt
```

Each checkpoint directory also contains:

```text
phonon_score_geom_mdn_config.json
```

The config file stores the thresholds, score weights, MDN settings, and training metadata needed to reconstruct
the model architecture.
