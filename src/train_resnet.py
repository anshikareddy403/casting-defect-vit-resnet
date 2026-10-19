"""Fine-tune ResNet-18 for binary casting defect classification."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from src.metrics import compute_metrics


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.train(train)
    losses: list[float] = []
    all_y: list[int] = []
    all_pred: list[int] = []
    for images, labels in tqdm(loader, leave=False, desc="train" if train else "eval"):
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train and optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        all_y.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
    return float(np.mean(losses)), np.array(all_y), np.array(all_pred)


def train_resnet(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    checkpoint_path: Path | None = None,
) -> tuple[nn.Module, dict[str, Any], list[dict[str, float]]]:
    model_cfg = cfg["models"]["resnet"]
    train_cfg = cfg["training"]
    model = build_resnet18(pretrained=model_cfg.get("pretrained", True)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"],
    )

    best_metric = -float("inf")
    best_state: dict[str, Any] | None = None
    patience = train_cfg["early_stopping_patience"]
    stale = 0
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for epoch in range(model_cfg["epochs"]):
        train_loss, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, y_val, pred_val = _run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        val_metrics = compute_metrics(y_val, pred_val)
        val_f1 = val_metrics["macro_f1"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_f1,
            }
        )
        if val_f1 > best_metric:
            best_metric = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break

    train_time_s = time.perf_counter() - t0
    if best_state:
        model.load_state_dict(best_state)
    meta = {"train_time_s": train_time_s, "best_val_macro_f1": best_metric, "epochs_run": len(history)}
    return model, meta, history
