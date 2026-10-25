"""Classification metrics and confusion-matrix helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASS_NAMES = ("acceptable", "defective")
DEFECT_LABEL = 1


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "defect_precision": float(
            precision_score(y_true, y_pred, pos_label=DEFECT_LABEL, zero_division=0)
        ),
        "defect_recall": float(
            recall_score(y_true, y_pred, pos_label=DEFECT_LABEL, zero_division=0)
        ),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=list(CLASS_NAMES), zero_division=0
        ),
    }


def format_metrics_row(
    model: str,
    train_fraction: float,
    metrics: dict[str, Any],
    train_time_s: float,
    infer_latency_ms: float,
    model_size_mb: float,
    seed: int,
) -> dict[str, Any]:
    cm = metrics["confusion_matrix"]
    return {
        "model": model,
        "train_fraction": train_fraction,
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "defect_precision": metrics["defect_precision"],
        "defect_recall": metrics["defect_recall"],
        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1],
        "train_time_s": train_time_s,
        "infer_latency_ms_mean": infer_latency_ms,
        "model_size_mb": model_size_mb,
    }


CSV_FIELDNAMES = [
    "model",
    "train_fraction",
    "seed",
    "accuracy",
    "macro_f1",
    "defect_precision",
    "defect_recall",
    "tn",
    "fp",
    "fn",
    "tp",
    "train_time_s",
    "infer_latency_ms_mean",
    "model_size_mb",
    "dataset_layout",
    "n_train",
    "n_val",
    "n_test",
]


def append_metrics_csv(row: dict[str, Any], csv_path: Path, **extra: Any) -> Path:
    """Append one experiment row to comparison_table.csv (creates file if missing)."""
    from src.utils import ensure_dir

    full = {**row, **extra}
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: full.get(k) for k in CSV_FIELDNAMES})
    return csv_path
