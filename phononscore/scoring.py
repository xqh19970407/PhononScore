"""Fixed training-reference normalization, independent of the query pool."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "fixed_scoring.json"
COMPONENTS = ("prediction", "threshold_score", "pair_geometry_score")


def load_scoring_config(path: Path | str | None = None) -> dict[str, Any]:
    payload = json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if payload.get("normalization") != "fixed_gradient_training_outputs":
        raise ValueError("Expected fixed gradient-training-output normalization.")
    for model in payload["models"].values():
        for key in COMPONENTS:
            component = model["components"][key]
            mean, std = float(component["mean"]), float(component["std"])
            if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
                raise ValueError(f"Invalid fixed statistics for {key}")
            if "variance" in component and not math.isclose(
                float(component["variance"]), std * std, rel_tol=1e-10, abs_tol=1e-14
            ):
                raise ValueError(f"Variance and standard deviation disagree for {key}")
    return payload


def verify_checkpoint(checkpoint: Path, model_name: str, config: dict[str, Any]) -> str:
    """Prevent using a new checkpoint with another checkpoint's reference moments."""
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected = config["models"][model_name]["checkpoint_sha256"]
    if digest != expected:
        raise ValueError(
            "Checkpoint does not match the fixed scoring reference. "
            "For a retrained model, freeze its training-output statistics and "
            "supply the matching file with --scoring-config."
        )
    return digest


def add_fixed_scores(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    config: dict[str, Any] | None = None,
    alpha: float | None = None,
    beta: float | None = None,
) -> None:
    config = config if config is not None else load_scoring_config()
    alpha = float(config["alpha"] if alpha is None else alpha)
    beta = float(config["beta"] if beta is None else beta)
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("Scoring weights must be finite.")
    reference = config["models"][model_name]["components"]
    for row in rows:
        normalized = []
        for key in COMPONENTS:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {key} for {row.get('id', '<unknown>')}")
            component = reference[key]
            z = (value - float(component["mean"])) / float(component["std"])
            row[f"{key}_z"] = z
            normalized.append(z)
        row["standardized_score"] = normalized[0] + beta * normalized[1] + alpha * normalized[2]
