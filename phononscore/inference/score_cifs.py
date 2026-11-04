#!/usr/bin/env python
"""Score CIF files with PhononScore or PhononScore-DFT checkpoints.

The script writes raw model outputs and a standardized score for every CIF,
using the frozen training-output moments of the selected checkpoint:

    standardized_score =
        z(prediction) + beta_eval * z(threshold_score)
        + alpha_eval * z(pair_geometry_score)

The default paper weights are alpha_eval=0.2 and beta_eval=0.5. A structure's
score is independent of the other structures submitted with it.
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


def add_standardized_scores(
    rows: list[dict[str, Any]], *, model_name: str,
    alpha_eval: float, beta_eval: float, scoring_config: dict[str, Any],
) -> None:
    from phononscore.scoring import add_fixed_scores

    add_fixed_scores(rows, model_name=model_name, config=scoring_config,
                     alpha=alpha_eval, beta=beta_eval)


def build_progress_message(
    *,
    stage: str,
    processed: int,
    total: int,
    success_count: int,
    failure_count: int,
    elapsed_seconds: float,
    recent_ok: list[str] | None = None,
    recent_fail: list[str] | None = None,
) -> str:
    attempted = success_count + failure_count
    success_rate = (100.0 * success_count / attempted) if attempted > 0 else 0.0
    parts = [
        f"[{stage}] {processed}/{total}",
        f"ok={success_count}",
        f"failed={failure_count}",
        f"success={success_rate:.1f}%",
        f"elapsed={elapsed_seconds:.1f}s",
    ]
    if recent_ok:
        parts.append(f"recent_ok={','.join(recent_ok)}")
    if recent_fail:
        parts.append(f"recent_fail={','.join(recent_fail)}")
    return " | ".join(parts)


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
        "prediction_z",
        "threshold_score_z",
        "pair_geometry_score_z",
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
    scoring_config: dict[str, Any],
    checkpoint_sha256: str,
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
        reference = scoring_config["models"][model_name]
        f.write("Normalization: fixed gradient-training-output moments (ddof=0)\n")
        f.write(f"Reference training records: {reference['n_train']}\n")
        f.write(f"Checkpoint SHA256: {checkpoint_sha256}\n")
        for key, stats in reference["components"].items():
            f.write(f"{key}: mean={stats['mean']:.17g}, std={stats['std']:.17g}, "
                    f"variance={stats['variance']:.17g}\n")
        f.write("\n")
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
    parser.add_argument("--alpha-eval", type=float, default=None, help="Geometry weight; defaults to the fixed config (0.2).")
    parser.add_argument("--beta-eval", type=float, default=None, help="Threshold weight; defaults to the fixed config (0.5).")
    parser.add_argument("--scoring-config", type=Path, default=None,
                        help="Fixed training-reference statistics and weights; defaults to configs/fixed_scoring.json.")
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Emit a flushed progress log line every N CIFs during graph building and scoring.",
    )
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/phononscore-mplconfig")
    repo_root = Path(__file__).resolve().parents[2]
    add_local_paths(repo_root)

    checkpoint, model_config = default_paths(repo_root, args.model)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    if args.model_config is not None:
        model_config = args.model_config
    alignn_config = args.alignn_config or (repo_root / "configs" / "config_min_phonon.json")

    from phononscore.scoring import load_scoring_config, verify_checkpoint

    scoring_config = load_scoring_config(args.scoring_config)
    args.alpha_eval = float(scoring_config["alpha"] if args.alpha_eval is None else args.alpha_eval)
    args.beta_eval = float(scoring_config["beta"] if args.beta_eval is None else args.beta_eval)
    checkpoint_sha256 = verify_checkpoint(checkpoint, args.model, scoring_config)

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
    recent_build_ok: list[str] = []
    recent_build_fail: list[str] = []
    for index, cif_path in enumerate(cif_paths, start=1):
        try:
            records.append(build_graph_record(cif_path, config))
            recent_build_ok.append(cif_path.name)
            recent_build_ok = recent_build_ok[-3:]
        except Exception as exc:
            if args.strict:
                raise
            failures.append({"id": cif_path.stem, "cif_path": str(cif_path), "error": repr(exc)})
            recent_build_fail.append(cif_path.name)
            recent_build_fail = recent_build_fail[-3:]
        if index % args.progress_every == 0 or index == len(cif_paths):
            print(
                build_progress_message(
                    stage="graph-build",
                    processed=index,
                    total=len(cif_paths),
                    success_count=len(records),
                    failure_count=len(failures),
                    elapsed_seconds=time.perf_counter() - build_started,
                    recent_ok=recent_build_ok,
                    recent_fail=recent_build_fail,
                ),
                flush=True,
            )
    build_seconds = time.perf_counter() - build_started

    rows: list[dict[str, Any]] = []
    inference_started = time.perf_counter()
    score_failures_before = len(failures)
    scored_records = 0
    recent_score_ok: list[str] = []
    recent_score_fail: list[str] = []
    for batch in iter_chunks(records, args.batch_size):
        try:
            batch_rows = score_records(model, batch, device)
            rows.extend(batch_rows)
            scored_records += len(batch)
            recent_score_ok.extend(row["id"] for row in batch_rows[:3])
            recent_score_ok = recent_score_ok[-3:]
        except Exception as exc:
            if args.strict:
                raise
            if len(batch) == 1:
                record = batch[0]
                failures.append({"id": record["id"], "cif_path": record["cif_path"], "error": repr(exc)})
                scored_records += 1
                recent_score_fail.append(record["id"])
                recent_score_fail = recent_score_fail[-3:]
                if scored_records % args.progress_every == 0 or scored_records == len(records):
                    print(
                        build_progress_message(
                            stage="scoring",
                            processed=scored_records,
                            total=len(records),
                            success_count=len(rows),
                            failure_count=len(failures) - score_failures_before,
                            elapsed_seconds=time.perf_counter() - inference_started,
                            recent_ok=recent_score_ok,
                            recent_fail=recent_score_fail,
                        ),
                        flush=True,
                    )
                continue
            for record in batch:
                try:
                    single_rows = score_records(model, [record], device)
                    rows.extend(single_rows)
                    recent_score_ok.extend(row["id"] for row in single_rows)
                    recent_score_ok = recent_score_ok[-3:]
                except Exception as item_exc:
                    failures.append(
                        {"id": record["id"], "cif_path": record["cif_path"], "error": repr(item_exc)}
                    )
                    recent_score_fail.append(record["id"])
                    recent_score_fail = recent_score_fail[-3:]
                finally:
                    scored_records += 1
        if scored_records % args.progress_every == 0 or scored_records == len(records):
            print(
                build_progress_message(
                    stage="scoring",
                    processed=scored_records,
                    total=len(records),
                    success_count=len(rows),
                    failure_count=len(failures) - score_failures_before,
                    elapsed_seconds=time.perf_counter() - inference_started,
                    recent_ok=recent_score_ok,
                    recent_fail=recent_score_fail,
                ),
                flush=True,
            )
    inference_seconds = time.perf_counter() - inference_started
    if not rows:
        raise RuntimeError("No CIF files were successfully scored.")
    add_standardized_scores(rows, model_name=args.model, alpha_eval=args.alpha_eval,
                            beta_eval=args.beta_eval, scoring_config=scoring_config)
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
        scoring_config=scoring_config,
        checkpoint_sha256=checkpoint_sha256,
    )
    print(f"Scored {len(rows)} CIF files with {args.model} on {device}.")
    print(f"Failed {len(failures)} CIF files.")
    print(f"Wrote {args.out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
