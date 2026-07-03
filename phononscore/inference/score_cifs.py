#!/usr/bin/env python
"""Score CIF files with PhononScore or PhononScore-DFT checkpoints.

The script is designed for candidate-pool reranking. For a pool of CIF files,
it writes the raw model outputs and, when at least two structures are present,
a standardized score:

    standardized_score =
        z(prediction) + beta_eval * z(threshold_score)
        + alpha_eval * z(pair_geometry_score)

The default paper-style weights are alpha_eval=0.25 and beta_eval=2.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def add_local_paths(repo_root: Path) -> None:
    """Make the bundled ALIGNN source importable without installation."""
    paths = [
        repo_root,
        repo_root / "third_party" / "alignn-main",
    ]
    for path in paths:
        text = str(path.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)


def collect_cif_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".cif":
            raise ValueError(f"Expected a .cif file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        paths = sorted(input_path.rglob("*.cif"))
        if not paths:
            raise ValueError(f"No .cif files found in directory: {input_path}")
        return paths
    raise FileNotFoundError(input_path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_thresholds(values: Any) -> tuple[float, ...]:
    if values is None:
        return (-0.001, -0.01, -0.1, -1.0)
    if isinstance(values, str):
        return tuple(float(x.strip()) for x in values.split(",") if x.strip())
    return tuple(float(x) for x in values)


def zscore(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = var ** 0.5
    if std <= 0:
        std = 1.0
    return [(x - mean) / std for x in values]


def build_graph_from_cif(cif_path: Path, config: Any):
    from alignn.graphs import Graph
    from jarvis.core.atoms import Atoms
    import torch

    atoms = Atoms.from_cif(str(cif_path), use_cif2cell=False)
    graph, line_graph = Graph.atom_dgl_multigraph(
        atoms,
        cutoff=float(config.cutoff),
        max_neighbors=config.max_neighbors,
        atom_features=config.atom_features,
        compute_line_graph=True,
        use_canonize=config.use_canonize,
        cutoff_extra=config.cutoff_extra,
        neighbor_strategy=config.neighbor_strategy,
        dtype=config.dtype,
    )
    lattice = torch.tensor(atoms.lattice_mat).type(torch.get_default_dtype())
    formula = getattr(atoms.composition, "reduced_formula", "")
    return atoms, graph, line_graph, lattice, formula


def build_graph_record(cif_path: Path, config: Any) -> dict[str, Any]:
    atoms, graph, line_graph, lattice, formula = build_graph_from_cif(cif_path, config)
    resolved_cif_path = cif_path.resolve()
    return {
        "id": cif_path.stem,
        "cif_path": str(cif_path),
        "cif_name": cif_path.name,
        "cif_stem": cif_path.stem,
        "resolved_cif_path": str(resolved_cif_path),
        "resolved_cif_name": resolved_cif_path.name,
        "resolved_cif_stem": resolved_cif_path.stem,
        "formula": formula,
        "n_atoms": int(atoms.num_atoms),
        "graph": graph,
        "line_graph": line_graph,
        "lattice": lattice,
    }


def load_model(*, checkpoint: Path, model_config_json: Path, alignn_config_json: Path, device: str):
    import torch
    from alignn.config import TrainingConfig
    from alignn.models.alignn_atomwise import ALIGNNAtomWiseConfig

    from phononscore.models import ALIGNNPhononScoreGeomMDN

    alignn_config = TrainingConfig(**load_json(alignn_config_json))
    model_payload = (
        alignn_config.model.model_dump()
        if hasattr(alignn_config.model, "model_dump")
        else alignn_config.model.dict()
    )
    atomwise_config = ALIGNNAtomWiseConfig(**model_payload)
    geom_config = load_json(model_config_json)
    model = ALIGNNPhononScoreGeomMDN(
        atomwise_config,
        thresholds=parse_thresholds(geom_config.get("thresholds")),
        beta=float(geom_config.get("beta", 0.6)),
        alpha=float(geom_config.get("alpha", 0.1)),
        mdn_hidden_dim=int(geom_config.get("mdn_hidden_dim", 128)),
        mdn_gaussians=int(geom_config.get("mdn_gaussians", 10)),
        mdn_distance_max=float(geom_config.get("mdn_distance_max", 8.0)),
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    target_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(target_device)
    model.eval()
    return model, alignn_config, target_device


def score_records(model, records: list[dict[str, Any]], device) -> list[dict[str, Any]]:
    import dgl
    import torch

    graph = dgl.batch([record["graph"] for record in records]).to(device)
    line_graph = dgl.batch([record["line_graph"] for record in records]).to(device)
    lattice = torch.stack([record["lattice"] for record in records], dim=0).to(device)
    with torch.no_grad():
        out = model([graph, line_graph, lattice])
    min_freq = out.min_freq.detach().cpu().view(-1).tolist()
    threshold_score = out.threshold_score.detach().cpu().view(-1).tolist()
    pair_geometry_score = out.pair_geometry_score.detach().cpu().view(-1).tolist()
    final_score = out.final_score.detach().cpu().view(-1).tolist()
    rows = []
    for i, record in enumerate(records):
        rows.append(
            {
                "id": record["id"],
                "cif_path": record["cif_path"],
                "cif_name": record["cif_name"],
                "cif_stem": record["cif_stem"],
                "resolved_cif_path": record["resolved_cif_path"],
                "resolved_cif_name": record["resolved_cif_name"],
                "resolved_cif_stem": record["resolved_cif_stem"],
                "formula": record["formula"],
                "n_atoms": record["n_atoms"],
                "prediction": float(min_freq[i]),
                "threshold_score": float(threshold_score[i]),
                "pair_geometry_score": float(pair_geometry_score[i]),
                "final_score": float(final_score[i]),
            }
        )
    return rows


def iter_chunks(items: list[Any], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def add_standardized_scores(rows: list[dict[str, Any]], *, alpha_eval: float, beta_eval: float) -> None:
    pred_z = zscore([float(row["prediction"]) for row in rows])
    thr_z = zscore([float(row["threshold_score"]) for row in rows])
    geom_z = zscore([float(row["pair_geometry_score"]) for row in rows])
    if pred_z is None or thr_z is None or geom_z is None:
        for row in rows:
            row["standardized_score"] = "NA"
        return
    for row, zp, zt, zg in zip(rows, pred_z, thr_z, geom_z):
        row["standardized_score"] = float(zp + beta_eval * zt + alpha_eval * zg)


def write_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "cif_path",
        "cif_name",
        "cif_stem",
        "resolved_cif_path",
        "resolved_cif_name",
        "resolved_cif_stem",
        "formula",
        "n_atoms",
        "prediction",
        "threshold_score",
        "pair_geometry_score",
        "final_score",
        "standardized_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    model_name: str,
    device: Any,
    requested_device: str,
    batch_size: int,
    alpha_eval: float,
    beta_eval: float,
    input_path: Path,
    csv_path: Path,
    total_cifs: int,
    build_seconds: float,
    inference_seconds: float,
    total_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    throughput = len(rows) / total_seconds if total_seconds > 0 else 0.0
    inference_throughput = len(rows) / inference_seconds if inference_seconds > 0 else 0.0
    top_rows = rows[: min(10, len(rows))]
    with path.open("w", encoding="utf-8") as f:
        f.write("PhononScore inference summary\n")
        f.write("=============================\n\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"CSV output: {csv_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Requested device: {requested_device}\n")
        f.write(f"Actual device: {device}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Total CIF files discovered: {total_cifs}\n")
        f.write(f"Successfully scored: {len(rows)}\n")
        f.write(f"Failed: {len(failures)}\n\n")
        f.write("Timing\n")
        f.write("------\n")
        f.write(f"Graph construction time: {build_seconds:.3f} s\n")
        f.write(f"Model inference time: {inference_seconds:.3f} s\n")
        f.write(f"Total wall time: {total_seconds:.3f} s\n")
        f.write(f"End-to-end throughput: {throughput:.3f} CIF/s\n")
        f.write(f"Model inference throughput: {inference_throughput:.3f} CIF/s\n\n")
        f.write("Standardized reranking score\n")
        f.write("----------------------------\n")
        f.write("standardized_score = z(prediction)\n")
        f.write(f"                   + {beta_eval:g} * z(threshold_score)\n")
        f.write(f"                   + {alpha_eval:g} * z(pair_geometry_score)\n\n")
        f.write("Top ranked structures\n")
        f.write("---------------------\n")
        if top_rows:
            f.write(
                "rank,id,cif_name,resolved_cif_name,formula,standardized_score,"
                "prediction,threshold_score,pair_geometry_score\n"
            )
            for rank, row in enumerate(top_rows, start=1):
                f.write(
                    f"{rank},{row['id']},{row['cif_name']},{row['resolved_cif_name']},"
                    f"{row['formula']},{row['standardized_score']},"
                    f"{row['prediction']},{row['threshold_score']},{row['pair_geometry_score']}\n"
                )
        else:
            f.write("No structures were successfully scored.\n")
        if failures:
            f.write("\nFailures\n")
            f.write("--------\n")
            f.write("id,cif_path,error\n")
            for item in failures:
                f.write(f"{item['id']},{item['cif_path']},{item['error']}\n")


def default_paths(repo_root: Path, model_name: str) -> tuple[Path, Path]:
    model_dir = repo_root / "checkpoints" / model_name
    return model_dir / "best_model.pt", model_dir / "phonon_score_geom_mdn_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A CIF file or a directory containing CIF files.")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--model",
        choices=("phononscore_pretrain", "phononscore_dft"),
        default="phononscore_dft",
        help="Bundled model to use when --checkpoint is not given.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to best_model.pt.")
    parser.add_argument("--model-config", type=Path, default=None, help="Path to phonon_score_geom_mdn_config.json.")
    parser.add_argument(
        "--alignn-config",
        type=Path,
        default=None,
        help="Path to config_min_phonon.json. Defaults to configs/config_min_phonon.json.",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of CIF graphs per GPU/CPU forward pass.")
    parser.add_argument("--alpha-eval", type=float, default=0.25, help="Weight for z(pair_geometry_score).")
    parser.add_argument("--beta-eval", type=float, default=2.0, help="Weight for z(threshold_score).")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Output TXT summary path. Defaults to the CSV path with .txt suffix.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately if any CIF fails to parse or score.",
    )
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/phononscore-mplconfig")
    repo_root = Path(__file__).resolve().parents[2]
    add_local_paths(repo_root)

    checkpoint, model_config = default_paths(repo_root, args.model)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    if args.model_config is not None:
        model_config = args.model_config
    alignn_config = args.alignn_config or (repo_root / "configs" / "config_min_phonon.json")

    model, config, device = load_model(
        checkpoint=checkpoint,
        model_config_json=model_config,
        alignn_config_json=alignn_config,
        device=args.device,
    )
    cif_paths = collect_cif_paths(args.input)
    failures: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    build_started = time.perf_counter()
    for cif_path in cif_paths:
        try:
            records.append(build_graph_record(cif_path, config))
        except Exception as exc:
            if args.strict:
                raise
            failures.append({"id": cif_path.stem, "cif_path": str(cif_path), "error": repr(exc)})
    build_seconds = time.perf_counter() - build_started

    rows: list[dict[str, Any]] = []
    inference_started = time.perf_counter()
    for batch in iter_chunks(records, args.batch_size):
        try:
            rows.extend(score_records(model, batch, device))
        except Exception as exc:
            if args.strict:
                raise
            if len(batch) == 1:
                record = batch[0]
                failures.append({"id": record["id"], "cif_path": record["cif_path"], "error": repr(exc)})
                continue
            for record in batch:
                try:
                    rows.extend(score_records(model, [record], device))
                except Exception as item_exc:
                    failures.append(
                        {"id": record["id"], "cif_path": record["cif_path"], "error": repr(item_exc)}
                    )
    inference_seconds = time.perf_counter() - inference_started
    if not rows:
        raise RuntimeError("No CIF files were successfully scored.")
    add_standardized_scores(rows, alpha_eval=args.alpha_eval, beta_eval=args.beta_eval)
    rows.sort(
        key=lambda row: float("-inf") if row["standardized_score"] == "NA" else float(row["standardized_score"]),
        reverse=True,
    )
    write_scores(args.out, rows)
    summary_out = args.summary_out or args.out.with_suffix(".txt")
    total_seconds = time.perf_counter() - started
    write_summary(
        summary_out,
        rows=rows,
        failures=failures,
        model_name=args.model,
        device=device,
        requested_device=args.device,
        batch_size=args.batch_size,
        alpha_eval=args.alpha_eval,
        beta_eval=args.beta_eval,
        input_path=args.input,
        csv_path=args.out,
        total_cifs=len(cif_paths),
        build_seconds=build_seconds,
        inference_seconds=inference_seconds,
        total_seconds=total_seconds,
    )
    print(f"Scored {len(rows)} CIF files with {args.model} on {device}.")
    print(f"Failed {len(failures)} CIF files.")
    print(f"Wrote {args.out}")
    print(f"Wrote {summary_out}")
    if len(rows) < 2:
        print("Note: standardized_score is NA for a single CIF; use a candidate pool for z-score reranking.")


if __name__ == "__main__":
    main()
