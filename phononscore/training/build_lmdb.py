#!/usr/bin/env python
"""Build ALIGNN train/val/test LMDB caches from a prepared user_data directory."""

from __future__ import annotations

import argparse
import csv
import json
import pickle as pk
import random
import sys
from pathlib import Path
from typing import Sequence


def _ensure_path(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def read_id_prop(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            rows.append((row[0], float(row[1])))
    return rows


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    mirror_test_to_val: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    ids = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    id_train = ids[:n_train]
    id_val = ids[n_train : n_train + n_val]
    if mirror_test_to_val:
        return id_train, id_val, list(id_val)
    id_test = ids[n_train + n_val :]
    return id_train, id_val, id_test


def lmdb_path(prefix: Path, split: str) -> Path:
    return Path(f"{prefix}{split}_data")


def build_one_lmdb(
    *,
    rows: Sequence[tuple[str, float]],
    indices: Sequence[int],
    dataset_dir: Path,
    out_path: Path,
    config,
    map_size: int,
) -> list[str]:
    from alignn.graphs import Graph
    from jarvis.core.atoms import Atoms
    import lmdb
    import torch
    from tqdm import tqdm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing LMDB: {out_path}")
    ids = []
    env = lmdb.open(str(out_path), map_size=int(map_size))
    with env.begin(write=True) as txn:
        for local_idx, source_idx in tqdm(enumerate(indices), total=len(indices), desc=out_path.name):
            rel_cif, target = rows[source_idx]
            atoms = Atoms.from_cif(str(dataset_dir / rel_cif))
            graph = Graph.atom_dgl_multigraph(
                atoms,
                cutoff=float(config.cutoff),
                max_neighbors=config.max_neighbors,
                atom_features=config.atom_features,
                compute_line_graph=bool(config.compute_line_graph),
                use_canonize=config.use_canonize,
                cutoff_extra=config.cutoff_extra,
                neighbor_strategy=config.neighbor_strategy,
                dtype=config.dtype,
            )
            if config.compute_line_graph:
                graph, line_graph = graph
            lattice = torch.tensor(atoms.lattice_mat).type(torch.get_default_dtype())
            label = torch.tensor(target).type(torch.get_default_dtype())
            if config.compute_line_graph:
                serialized = pk.dumps((graph, line_graph, lattice, label))
            else:
                serialized = pk.dumps((graph, lattice, label))
            txn.put(f"{local_idx}".encode(), serialized)
            ids.append(rel_cif)
    env.close()
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignn_src", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lmdb_prefix", type=Path, required=True)
    parser.add_argument("--ids_json", type=Path, required=True)
    parser.add_argument("--split_ids", type=Path, default=None,
                        help="Frozen ID lists to reproduce the released training partition exactly.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--mirror_test_to_val", action="store_true")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--map_size", type=int, default=1_000_000_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_path(args.alignn_src)
    _ensure_path(Path(__file__).resolve().parents[2])

    from alignn.config import TrainingConfig
    from phononscore.training.safety import guard_against_legacy_seed20260602

    guard_against_legacy_seed20260602([args.dataset_dir, args.config, args.lmdb_prefix, args.ids_json])

    config = TrainingConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    rows = read_id_prop(args.dataset_dir / "id_prop.csv")
    if args.split_ids is not None:
        frozen = json.loads(args.split_ids.read_text(encoding="utf-8"))
        by_id = {name: i for i, (name, _) in enumerate(rows)}
        if len(by_id) != len(rows):
            raise ValueError("Duplicate IDs in id_prop.csv")
        if set(frozen["id_train"]) & set(frozen["id_val"]):
            raise ValueError("Gradient-training and checkpoint-validation IDs overlap")
        if set(frozen["id_train"]) & set(frozen["id_test"]):
            raise ValueError("Gradient-training and internal evaluation IDs overlap")
        for key in ("id_train", "id_val", "id_test"):
            if len(frozen[key]) != len(set(frozen[key])):
                raise ValueError(f"Duplicate IDs in {key}")
        id_train, id_val, id_test = (
            [by_id[name] for name in frozen[key]]
            for key in ("id_train", "id_val", "id_test")
        )
    else:
        id_train, id_val, id_test = split_indices(
            len(rows), args.train_ratio, args.val_ratio, args.seed,
            mirror_test_to_val=args.mirror_test_to_val,
        )
    ids_data = {
        "id_train": build_one_lmdb(
            rows=rows,
            indices=id_train,
            dataset_dir=args.dataset_dir,
            out_path=lmdb_path(args.lmdb_prefix, "train"),
            config=config,
            map_size=args.map_size,
        ),
        "id_val": build_one_lmdb(
            rows=rows,
            indices=id_val,
            dataset_dir=args.dataset_dir,
            out_path=lmdb_path(args.lmdb_prefix, "val"),
            config=config,
            map_size=args.map_size,
        ),
        "id_test": build_one_lmdb(
            rows=rows,
            indices=id_test,
            dataset_dir=args.dataset_dir,
            out_path=lmdb_path(args.lmdb_prefix, "test"),
            config=config,
            map_size=args.map_size,
        ),
    }
    args.ids_json.parent.mkdir(parents=True, exist_ok=True)
    args.ids_json.write_text(json.dumps(ids_data, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "lmdb_prefix": str(args.lmdb_prefix),
        "ids_json": str(args.ids_json),
        "n_total": len(rows),
        "n_train": len(ids_data["id_train"]),
        "n_val": len(ids_data["id_val"]),
        "n_test": len(ids_data["id_test"]),
        "seed": args.seed,
        "split_source": "frozen_ids" if args.split_ids is not None else "random_ratios",
        "frozen_split_ids": str(args.split_ids) if args.split_ids is not None else None,
        "train_ratio": args.train_ratio if args.split_ids is None else None,
        "val_ratio": args.val_ratio if args.split_ids is None else None,
        "mirror_test_to_val": ids_data["id_test"] == ids_data["id_val"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
