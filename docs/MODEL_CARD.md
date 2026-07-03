# Model Card

## Models

### PhononScore

- Checkpoint: `checkpoints/phononscore_pretrain/best_model.pt`
- Training labels: MatterSim phonon labels
- Training sources: generated crystal structures and MP40 real-material structures
- Main use: generated-candidate reranking

### PhononScore-DFT

- Checkpoint: `checkpoints/phononscore_dft/best_model.pt`
- Initialization: PhononScore pretrained checkpoint
- Post-training labels: DFT-PBE phonon labels
- Main use: DFT-PBE calibrated dynamical-stability ranking

## Intended Use

PhononScore ranks CIF candidates by likely dynamical stability. It is designed for high-throughput screening and
Top-K enrichment, not for replacing explicit phonon calculations.

## Runtime Environment

The release environment uses the full dependency stack exported from the `my_alignn` conda environment used during
development, training, and local smoke testing. The public environment name is `phononscore`:

```bash
conda env create -f environment.yml
conda activate phononscore
```

The exported file is intentionally full rather than minimal, because the current inference path depends on the
same ALIGNN, DGL, PyTorch, JARVIS, and CIF-parsing stack used to produce the reported results. A raw export with
the original local environment name and prefix is kept in `docs/environment-my_alignn-raw.yml` for provenance.

## Limitations

- The score is a ranking score, not a certified minimum phonon frequency.
- Standardized scores require a candidate pool.
- Materials outside the training distribution should be validated with explicit phonon calculations.
- Structures that fail CIF parsing or ALIGNN graph construction cannot be scored.
