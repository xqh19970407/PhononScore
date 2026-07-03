#!/usr/bin/env python
"""Benchmark PhononScore inference on up to 1000 CIF files.

The benchmark runs the public CIF-folder inference entry point repeatedly with
different batch sizes on a single selected GPU. It stops at the first CUDA OOM
by default and writes both per-batch outputs and a compact benchmark summary.

Example:

    python scripts/benchmark_1000_cifs.py \
      --input path/to/cif_directory \
      --gpu 0 \
      --batch-sizes 8,16,32,64,128

For CPU-only smoke testing of this benchmark script:

    python scripts/benchmark_1000_cifs.py \
      --input examples/cifs \
      --batch-sizes 1,2 \
      --max-cifs 4 \
      --allow-cpu
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def collect_cifs(input_dir: Path, *, max_cifs: int) -> list[Path]:
    if input_dir.is_file():
        if input_dir.suffix.lower() != ".cif":
            raise ValueError(f"Expected a .cif file or CIF directory, got: {input_dir}")
        return [input_dir]
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    paths = sorted(input_dir.rglob("*.cif"))
    if not paths:
        raise ValueError(f"No .cif files found in {input_dir}")
    return paths[:max_cifs]


def parse_batch_sizes(text: str) -> list[int]:
    sizes = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("Batch sizes must be positive integers.")
        sizes.append(value)
    if not sizes:
        raise ValueError("At least one batch size is required.")
    return sizes


def make_subset_dir(cif_paths: list[Path], subset_dir: Path) -> None:
    subset_dir.mkdir(parents=True, exist_ok=False)
    for index, path in enumerate(cif_paths):
        target = subset_dir / f"{index:06d}_{path.name}"
        try:
            target.symlink_to(path.resolve())
        except OSError:
            shutil.copy2(path, target)


def build_env(repo_root: Path, *, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp/phononscore-mplconfig")
    pythonpath = [
        str(repo_root),
        str(repo_root / "third_party" / "alignn-main"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(x for x in pythonpath if x)
    return env


def cuda_available(python_bin: str, env: dict[str, str]) -> bool:
    code = "import torch; print(int(torch.cuda.is_available()))"
    proc = subprocess.run(
        [python_bin, "-c", code],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode == 0 and proc.stdout.strip().endswith("1")


def contains_oom(text: str) -> bool:
    lowered = text.lower()
    needles = [
        "cuda out of memory",
        "outofmemoryerror",
        "out of memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
    ]
    return any(item in lowered for item in needles)


def parse_summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    patterns = {
        "actual_device": r"^Actual device:\s*(.+)$",
        "scored": r"^Successfully scored:\s*(.+)$",
        "failed": r"^Failed:\s*(.+)$",
        "graph_seconds": r"^Graph construction time:\s*(.+?)\s*s$",
        "inference_seconds": r"^Model inference time:\s*(.+?)\s*s$",
        "total_seconds": r"^Total wall time:\s*(.+?)\s*s$",
        "throughput": r"^End-to-end throughput:\s*(.+?)\s*CIF/s$",
        "inference_throughput": r"^Model inference throughput:\s*(.+?)\s*CIF/s$",
    }
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        for key, pattern in patterns.items():
            match = re.match(pattern, line)
            if match:
                result[key] = match.group(1).strip()
    return result


def run_batch(
    *,
    repo_root: Path,
    python_bin: str,
    env: dict[str, str],
    subset_dir: Path,
    run_dir: Path,
    model: str,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    batch_dir = run_dir / f"batch_{batch_size:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    csv_path = batch_dir / "scores.csv"
    txt_path = batch_dir / "scores.txt"
    log_path = batch_dir / "run.log"
    cmd = [
        python_bin,
        "-m",
        "phononscore.inference.score_cifs",
        "--input",
        str(subset_dir),
        "--out",
        str(csv_path),
        "--summary-out",
        str(txt_path),
        "--model",
        model,
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--strict",
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(proc.stdout, encoding="utf-8")
    oom = contains_oom(proc.stdout)
    parsed = parse_summary(txt_path)
    status = "ok" if proc.returncode == 0 else ("oom" if oom else "failed")
    row: dict[str, Any] = {
        "batch_size": batch_size,
        "status": status,
        "returncode": proc.returncode,
        "elapsed_seconds": f"{elapsed:.3f}",
        "actual_device": parsed.get("actual_device", ""),
        "scored": parsed.get("scored", ""),
        "failed": parsed.get("failed", ""),
        "graph_seconds": parsed.get("graph_seconds", ""),
        "inference_seconds": parsed.get("inference_seconds", ""),
        "total_seconds": parsed.get("total_seconds", ""),
        "throughput_cif_per_s": parsed.get("throughput", ""),
        "inference_throughput_cif_per_s": parsed.get("inference_throughput", ""),
        "csv_path": str(csv_path),
        "summary_path": str(txt_path),
        "log_path": str(log_path),
    }
    return row


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "batch_size",
        "status",
        "returncode",
        "elapsed_seconds",
        "actual_device",
        "scored",
        "failed",
        "graph_seconds",
        "inference_seconds",
        "total_seconds",
        "throughput_cif_per_s",
        "inference_throughput_cif_per_s",
        "csv_path",
        "summary_path",
        "log_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_success(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok_rows = [row for row in rows if row["status"] == "ok" and row.get("throughput_cif_per_s")]
    if not ok_rows:
        return None
    return max(ok_rows, key=lambda row: float(row["throughput_cif_per_s"]))


def write_summary(path: Path, *, rows: list[dict[str, Any]], args: argparse.Namespace, run_dir: Path) -> None:
    best = best_success(rows)
    with path.open("w", encoding="utf-8") as f:
        f.write("PhononScore 1000-CIF benchmark summary\n")
        f.write("======================================\n\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Run directory: {run_dir}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"GPU id via CUDA_VISIBLE_DEVICES: {args.gpu}\n")
        f.write(f"Batch sizes: {args.batch_sizes}\n")
        f.write(f"Max CIFs: {args.max_cifs}\n\n")
        if best is None:
            f.write("No successful batch-size run was recorded.\n")
        else:
            f.write("Fastest successful run\n")
            f.write("----------------------\n")
            f.write(f"Batch size: {best['batch_size']}\n")
            f.write(f"Actual device: {best['actual_device']}\n")
            f.write(f"Total wall time: {best['total_seconds']} s\n")
            f.write(f"End-to-end throughput: {best['throughput_cif_per_s']} CIF/s\n")
            f.write(f"Model inference throughput: {best['inference_throughput_cif_per_s']} CIF/s\n")
            f.write(f"CSV: {best['csv_path']}\n")
            f.write(f"TXT: {best['summary_path']}\n\n")
        f.write("All runs\n")
        f.write("--------\n")
        f.write("batch_size,status,total_seconds,throughput_cif_per_s,actual_device\n")
        for row in rows:
            f.write(
                f"{row['batch_size']},{row['status']},{row['total_seconds']},"
                f"{row['throughput_cif_per_s']},{row['actual_device']}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CIF directory or one CIF file.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/benchmark_1000_cifs"))
    parser.add_argument("--model", choices=("phononscore_dft", "phononscore_pretrain"), default="phononscore_dft")
    parser.add_argument("--gpu", default="0", help="Single GPU id exposed via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--max-cifs", type=int, default=1000)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback for smoke testing.")
    parser.add_argument("--no-stop-on-oom", action="store_true", help="Continue testing larger batches after OOM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_cifs <= 0:
        raise ValueError("--max-cifs must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    env = build_env(repo_root, gpu=args.gpu)
    if not args.allow_cpu and not cuda_available(args.python_bin, env):
        raise RuntimeError(
            "CUDA is not available for the selected GPU. Use --allow-cpu only for script smoke testing."
        )

    cif_paths = collect_cifs(args.input, max_cifs=args.max_cifs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.out_dir / f"{args.model}_{timestamp}").resolve()
    subset_dir = run_dir / "subset_cifs"
    run_dir.mkdir(parents=True, exist_ok=False)
    make_subset_dir(cif_paths, subset_dir)

    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        row = run_batch(
            repo_root=repo_root,
            python_bin=args.python_bin,
            env=env,
            subset_dir=subset_dir,
            run_dir=run_dir,
            model=args.model,
            batch_size=batch_size,
            device="cuda",
        )
        rows.append(row)
        write_results_csv(run_dir / "benchmark_results.csv", rows)
        write_summary(run_dir / "benchmark_summary.txt", rows=rows, args=args, run_dir=run_dir)
        print(
            f"batch={batch_size} status={row['status']} "
            f"total={row['total_seconds']}s throughput={row['throughput_cif_per_s']} CIF/s"
        )
        if row["status"] == "oom" and not args.no_stop_on_oom:
            print("Stopping at first CUDA OOM.")
            break
        if row["status"] == "failed":
            print(f"Stopping after non-OOM failure. See {row['log_path']}")
            break

    print(f"Benchmark results: {run_dir / 'benchmark_results.csv'}")
    print(f"Benchmark summary: {run_dir / 'benchmark_summary.txt'}")


if __name__ == "__main__":
    main()
