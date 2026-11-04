"""Safety guards for clean seed20260605 phonon-scoring experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


LEGACY_MARKERS = (
    "seed20260602",
    "holdout_seed20260602",
    "mp20_8models_holdout_split_seed20260602",
    "alignn_min_phonon_clip5_holdout_seed20260602",
    "min_phonon_clip5_holdout_seed20260602",
)


def guard_against_legacy_seed20260602(paths: Iterable[str | Path]) -> None:
    """Reject paths that would reuse the old leakage-risk seed20260602 split/cache."""
    bad = []
    for path in paths:
        text = str(path)
        if any(marker in text for marker in LEGACY_MARKERS):
            bad.append(text)
    if bad:
        joined = "\n  - ".join(bad)
        raise ValueError(
            "Refusing to use legacy seed20260602 / holdout paths for clean seed20260605 training:\n"
            f"  - {joined}"
        )
