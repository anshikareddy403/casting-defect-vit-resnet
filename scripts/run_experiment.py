#!/usr/bin/env python3
"""Run one (model, train_fraction) experiment and append results to CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    load_all_samples,
    make_loaders,
    resolve_data_root,
    split_train_val,
    subset_train_stratified,
)
from src.evaluate import evaluate_model
from src.metrics import format_metrics_row
from src.train_resnet import train_resnet
from src.train_vit import train_vit
from src.utils import ensure_dir, load_config, resolve_device, save_json, set_seed
from src.visualize import plot_confusion_matrix, plot_training_history


def append_csv(row: dict, csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/eval one casting defect experiment.")
    parser.add_argument("--model", choices=["vit", "resnet"], required=True)
    parser.add_argument("--fraction", type=float, required=True, help="Train subset fraction (0.25, 0.5, 1.0)")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = resolve_device(cfg.get("device", "auto"))
    data_root = resolve_data_root(cfg)
    layout = cfg["data"].get("layout", "auto")

    train_pool, test_samples, resolved_layout = load_all_samples(data_root, layout=layout)
    train_samples, val_samples = split_train_val(
        train_pool,
        val_fraction=cfg["data"]["val_fraction"],
        seed=seed,
        group_augmentations=cfg["data"].get("group_augmentations", True),
    )
    train_subset = subset_train_stratified(
        train_samples,
        fraction=args.fraction,
        seed=seed,
        group_augmentations=cfg["data"].get("group_augmentations", True),
    )

    train_loader, val_loader, test_loader = make_loaders(
        train_subset,
        val_samples,
        test_samples,
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 2),
    )

    run_name = f"{args.model}_frac{args.fraction}_seed{seed}"
    ckpt_dir = ensure_dir(ROOT / cfg["paths"]["checkpoints"] / run_name)
    ckpt_path = ckpt_dir / "best.pt"

    if args.model == "resnet":
        model, train_meta, history = train_resnet(
            train_loader, val_loader, device, cfg, checkpoint_path=ckpt_path
        )
        is_vit = False
    else:
        model, train_meta, history = train_vit(
            train_loader, val_loader, device, cfg, checkpoint_path=ckpt_path
        )
        is_vit = True

    eval_out = evaluate_model(model, test_loader, device, is_vit=is_vit)
    row = format_metrics_row(
        model=args.model,
        train_fraction=args.fraction,
        metrics=eval_out["metrics"],
        train_time_s=train_meta["train_time_s"],
        infer_latency_ms=eval_out["infer_latency_ms"],
        model_size_mb=eval_out["model_size_mb"],
        seed=seed,
    )
    row["dataset_layout"] = resolved_layout
    row["n_train"] = len(train_subset)
    row["n_val"] = len(val_samples)
    row["n_test"] = len(test_samples)

    csv_path = ROOT / cfg["paths"]["comparison_csv"]
    append_csv(row, csv_path)

    log_dir = ensure_dir(ROOT / cfg["paths"]["logs"] / run_name)
    save_json({"history": history, "train_meta": train_meta, "eval": eval_out["metrics"]}, log_dir / "metrics.json")

    if not args.no_viz:
        fig_dir = ensure_dir(ROOT / cfg["paths"]["figures"] / run_name)
        plot_training_history(history, fig_dir / "training_curves.png", title=run_name)
        plot_confusion_matrix(
            eval_out["metrics"]["confusion_matrix"],
            fig_dir / "confusion_matrix.png",
            title=run_name,
        )

    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
