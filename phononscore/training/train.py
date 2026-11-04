"""Train PhononScore with an RTMScore-like stable pair-geometry MDN branch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _ensure_path(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def parse_thresholds(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignn_src", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lmdb_prefix", type=Path, required=True)
    parser.add_argument("--ids_json", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--restart_encoder", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--thresholds", default="-0.001,-0.01,-0.1,-1.0")
    parser.add_argument("--reg_target_mode", choices=("raw", "clip5"), default="clip5")
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lambda_cls", type=float, default=0.3)
    parser.add_argument("--lambda_rank", type=float, default=0.3)
    parser.add_argument("--lambda_ordinal", type=float, default=0.02)
    parser.add_argument("--lambda_mdn", type=float, default=0.05)
    parser.add_argument("--lambda_geom_rank", type=float, default=0.05)
    parser.add_argument("--cls_loss", choices=("bce", "balanced_bce", "focal", "balanced_focal"), default="balanced_bce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--rank_loss", choices=("pairwise", "threshold_aware"), default="threshold_aware")
    parser.add_argument("--min_label_gap", type=float, default=0.05)
    parser.add_argument("--max_rank_pairs", type=int, default=4096)
    parser.add_argument("--mdn_hidden_dim", type=int, default=128)
    parser.add_argument("--mdn_gaussians", type=int, default=10)
    parser.add_argument("--mdn_distance_max", type=float, default=8.0)
    parser.add_argument("--stable_threshold", type=float, default=-0.1)
    parser.add_argument("--stable_tau", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--save_every", type=int, default=100)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_lmdb(prefix: Path, split: str) -> Path:
    return Path(f"{prefix}{split}_data")


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_line_graph if dataset.line_graph else dataset.collate,
        num_workers=num_workers,
        pin_memory=False,
    )


def to_device_batch(batch, device: torch.device):
    graph, line_graph, lattice, target = batch
    return graph.to(device), line_graph.to(device), lattice.to(device), target.to(device).view(-1)


def transform_regression_target(target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return target
    if mode == "clip5":
        return target.clamp(min=-5.0, max=0.0)
    raise ValueError(f"Unknown regression target mode: {mode}")


def epoch_pass(
    *,
    model,
    loader,
    optimizer,
    device: torch.device,
    thresholds: Sequence[float],
    lambda_cls: float,
    lambda_rank: float,
    lambda_ordinal: float,
    lambda_mdn: float,
    lambda_geom_rank: float,
    cls_loss_mode: str,
    focal_gamma: float,
    rank_loss_mode: str,
    min_label_gap: float,
    max_rank_pairs: int | None,
    reg_target_mode: str,
    stable_threshold: float,
    stable_tau: float,
    train: bool,
):
    from phononscore.models.alignn_phonon_score_geom_mdn import stable_sample_weight, weighted_mdn_nll
    from phononscore.models.phonon_ranker_losses import (
        classification_loss,
        make_threshold_labels,
        mdn_negative_log_likelihood,
        ordinal_monotonic_loss,
        pairwise_ranking_loss,
        threshold_aware_ranking_loss,
    )

    mse = nn.SmoothL1Loss()
    model.train(train)
    total = 0
    sums = {
        "loss": 0.0,
        "reg": 0.0,
        "cls": 0.0,
        "rank": 0.0,
        "ordinal": 0.0,
        "geom_mdn": 0.0,
        "geom_rank": 0.0,
    }
    preds: list[float] = []
    scores: list[float] = []
    threshold_scores: list[float] = []
    geometry_scores: list[float] = []
    targets_all: list[float] = []
    with torch.set_grad_enabled(train):
        for batch in loader:
            graph, line_graph, lattice, target = to_device_batch(batch, device)
            out = model([graph, line_graph, lattice])
            reg_target = transform_regression_target(target, reg_target_mode)
            reg_loss = mse(out.min_freq, reg_target)
            cls_labels = make_threshold_labels(target, thresholds)
            cls_loss = classification_loss(
                out.threshold_logits,
                cls_labels,
                mode=cls_loss_mode,
                focal_gamma=focal_gamma,
            )
            if rank_loss_mode == "threshold_aware":
                rank_loss = threshold_aware_ranking_loss(
                    out.final_score,
                    target,
                    thresholds=thresholds,
                    max_pairs=max_rank_pairs,
                )
            else:
                rank_loss = pairwise_ranking_loss(
                    out.final_score,
                    target,
                    min_label_gap=min_label_gap,
                    max_pairs=max_rank_pairs,
                )
            ordinal_loss = ordinal_monotonic_loss(out.threshold_logits)
            edge_nll = mdn_negative_log_likelihood(out.mdn_pi, out.mdn_sigma, out.mdn_mu, out.edge_distance)
            sample_weight = stable_sample_weight(target, threshold=stable_threshold, tau=stable_tau)
            geom_mdn_loss = weighted_mdn_nll(edge_nll, edge_batch=out.edge_batch, sample_weight=sample_weight)
            geom_rank_loss = threshold_aware_ranking_loss(
                out.pair_geometry_score,
                target,
                thresholds=thresholds,
                max_pairs=max_rank_pairs,
            )
            loss = (
                reg_loss
                + lambda_cls * cls_loss
                + lambda_rank * rank_loss
                + lambda_ordinal * ordinal_loss
                + lambda_mdn * geom_mdn_loss
                + lambda_geom_rank * geom_rank_loss
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            n = int(target.numel())
            total += n
            sums["loss"] += float(loss.detach().cpu()) * n
            sums["reg"] += float(reg_loss.detach().cpu()) * n
            sums["cls"] += float(cls_loss.detach().cpu()) * n
            sums["rank"] += float(rank_loss.detach().cpu()) * n
            sums["ordinal"] += float(ordinal_loss.detach().cpu()) * n
            sums["geom_mdn"] += float(geom_mdn_loss.detach().cpu()) * n
            sums["geom_rank"] += float(geom_rank_loss.detach().cpu()) * n
            preds.extend(out.min_freq.detach().cpu().view(-1).tolist())
            scores.extend(out.final_score.detach().cpu().view(-1).tolist())
            threshold_scores.extend(out.threshold_score.detach().cpu().view(-1).tolist())
            geometry_scores.extend(out.pair_geometry_score.detach().cpu().view(-1).tolist())
            targets_all.extend(target.detach().cpu().view(-1).tolist())
    metrics = {k: v / max(total, 1) for k, v in sums.items()}
    metrics["mae"] = float(np.mean(np.abs(np.asarray(preds) - np.asarray(targets_all)))) if total else math.nan
    if reg_target_mode == "clip5" and total:
        clip_targets = np.clip(np.asarray(targets_all), -5.0, 0.0)
        metrics["mae_reg_target"] = float(np.mean(np.abs(np.asarray(preds) - clip_targets)))
    metrics["mean_threshold_score"] = float(np.mean(threshold_scores)) if threshold_scores else math.nan
    metrics["mean_pair_geometry_score"] = float(np.mean(geometry_scores)) if geometry_scores else math.nan
    metrics["n"] = total
    return metrics, targets_all, preds, threshold_scores, geometry_scores, scores


def save_predictions(
    path: Path,
    ids: Sequence[str],
    targets: Sequence[float],
    preds: Sequence[float],
    threshold_scores: Sequence[float],
    geometry_scores: Sequence[float],
    scores: Sequence[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "target", "prediction", "threshold_score", "pair_geometry_score", "final_score"])
        for jid, target, pred, threshold_score, geometry_score, score in zip(
            ids,
            targets,
            preds,
            threshold_scores,
            geometry_scores,
            scores,
        ):
            writer.writerow(
                [
                    jid,
                    f"{float(target):.12g}",
                    f"{float(pred):.12g}",
                    f"{float(threshold_score):.12g}",
                    f"{float(geometry_score):.12g}",
                    f"{float(score):.12g}",
                ]
            )


def main() -> None:
    args = parse_args()
    _ensure_path(args.alignn_src)
    _ensure_path(Path(__file__).resolve().parents[2])
    from phononscore.training.safety import guard_against_legacy_seed20260602

    guard_against_legacy_seed20260602(
        [args.config, args.lmdb_prefix, args.ids_json, args.out_dir, args.restart_encoder or ""]
    )
    set_seed(args.seed)

    from alignn.config import TrainingConfig
    from alignn.lmdb_dataset import TorchLMDBDataset
    from alignn.models.alignn_atomwise import ALIGNNAtomWiseConfig

    from phononscore.models.alignn_phonon_score_geom_mdn import ALIGNNPhononScoreGeomMDN

    args.out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)
    config = TrainingConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    model_config = ALIGNNAtomWiseConfig(**config.model.dict())
    model = ALIGNNPhononScoreGeomMDN(
        model_config,
        thresholds=thresholds,
        beta=args.beta,
        alpha=args.alpha,
        mdn_hidden_dim=args.mdn_hidden_dim,
        mdn_gaussians=args.mdn_gaussians,
        mdn_distance_max=args.mdn_distance_max,
    )
    if args.restart_encoder is not None:
        state = torch.load(args.restart_encoder, map_location="cpu")
        model.load_alignn_atomwise_state(state)
        print(f"[restart] loaded compatible ALIGNN encoder/head weights from {args.restart_encoder}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    ids_data = json.loads(args.ids_json.read_text(encoding="utf-8"))
    train_dataset = TorchLMDBDataset(str(split_lmdb(args.lmdb_prefix, "train")), line_graph=True, ids=ids_data["id_train"])
    val_dataset = TorchLMDBDataset(str(split_lmdb(args.lmdb_prefix, "val")), line_graph=True, ids=ids_data["id_val"])
    test_dataset = TorchLMDBDataset(str(split_lmdb(args.lmdb_prefix, "test")), line_graph=True, ids=ids_data["id_test"])
    train_loader = build_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = build_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False)
    test_loader = build_loader(test_dataset, args.batch_size, args.num_workers, shuffle=False)

    run_config = jsonable(vars(args).copy())
    run_config["thresholds"] = thresholds
    run_config["device"] = str(device)
    (args.out_dir / "phonon_score_geom_mdn_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_val = float("inf")
    history: list[dict[str, object]] = []
    print(f"[start] train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} device={device}")
    for epoch in range(1, args.epochs + 1):
        tic = time.time()
        train_metrics, _, _, _, _, _ = epoch_pass(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            thresholds=thresholds,
            lambda_cls=args.lambda_cls,
            lambda_rank=args.lambda_rank,
            lambda_ordinal=args.lambda_ordinal,
            lambda_mdn=args.lambda_mdn,
            lambda_geom_rank=args.lambda_geom_rank,
            cls_loss_mode=args.cls_loss,
            focal_gamma=args.focal_gamma,
            rank_loss_mode=args.rank_loss,
            min_label_gap=args.min_label_gap,
            max_rank_pairs=args.max_rank_pairs,
            reg_target_mode=args.reg_target_mode,
            stable_threshold=args.stable_threshold,
            stable_tau=args.stable_tau,
            train=True,
        )
        with torch.no_grad():
            val_metrics, _, _, _, _, _ = epoch_pass(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                thresholds=thresholds,
                lambda_cls=args.lambda_cls,
                lambda_rank=args.lambda_rank,
                lambda_ordinal=args.lambda_ordinal,
                lambda_mdn=args.lambda_mdn,
                lambda_geom_rank=args.lambda_geom_rank,
                cls_loss_mode=args.cls_loss,
                focal_gamma=args.focal_gamma,
                rank_loss_mode=args.rank_loss,
                min_label_gap=args.min_label_gap,
                max_rank_pairs=args.max_rank_pairs,
                reg_target_mode=args.reg_target_mode,
                stable_threshold=args.stable_threshold,
                stable_tau=args.stable_tau,
                train=False,
            )
        elapsed = time.time() - tic
        row = {"epoch": epoch, "time_s": elapsed, "train": train_metrics, "val": val_metrics}
        history.append(row)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"[epoch {epoch:04d}] "
            f"train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} "
            f"val_mae={val_metrics['mae']:.5f} "
            f"val_reg={val_metrics['reg']:.5f} val_cls={val_metrics['cls']:.5f} "
            f"val_rank={val_metrics['rank']:.5f} val_geom_mdn={val_metrics['geom_mdn']:.5f} "
            f"val_geom_rank={val_metrics['geom_rank']:.5f} "
            f"val_geom_score={val_metrics['mean_pair_geometry_score']:.5f} time={elapsed:.1f}s",
            flush=True,
        )
        torch.save(model.state_dict(), args.out_dir / "current_model.pt")
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(model.state_dict(), args.out_dir / f"epoch_{epoch:06d}.pt")

    model.load_state_dict(torch.load(args.out_dir / "best_model.pt", map_location=device))
    with torch.no_grad():
        test_metrics, targets, preds, threshold_scores, geometry_scores, scores = epoch_pass(
            model=model,
            loader=test_loader,
            optimizer=optimizer,
            device=device,
            thresholds=thresholds,
            lambda_cls=args.lambda_cls,
            lambda_rank=args.lambda_rank,
            lambda_ordinal=args.lambda_ordinal,
            lambda_mdn=args.lambda_mdn,
            lambda_geom_rank=args.lambda_geom_rank,
            cls_loss_mode=args.cls_loss,
            focal_gamma=args.focal_gamma,
            rank_loss_mode=args.rank_loss,
            min_label_gap=args.min_label_gap,
            max_rank_pairs=args.max_rank_pairs,
            reg_target_mode=args.reg_target_mode,
            stable_threshold=args.stable_threshold,
            stable_tau=args.stable_tau,
            train=False,
        )
    save_predictions(
        args.out_dir / "prediction_results_test_set.csv",
        ids_data["id_test"],
        targets,
        preds,
        threshold_scores,
        geometry_scores,
        scores,
    )
    (args.out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"[done] best_val_loss={best_val:.6f} test_loss={test_metrics['loss']:.6f} test_mae={test_metrics['mae']:.6f}")
    train_dataset.close()
    val_dataset.close()
    test_dataset.close()


if __name__ == "__main__":
    main()
