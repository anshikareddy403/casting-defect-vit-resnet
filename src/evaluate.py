"""Test-set evaluation, latency, and model size."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from src.metrics import compute_metrics
from src.train_resnet import build_resnet18


def model_size_mb(model: nn.Module | ViTForImageClassification) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024**2)


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_vit: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[torch.Tensor]]:
    model.eval()
    all_y: list[int] = []
    all_pred: list[int] = []
    all_images: list[torch.Tensor] = []
    for images, labels in tqdm(loader, desc="test"):
        images = images.to(device)
        if is_vit:
            logits = model(pixel_values=images).logits
        else:
            logits = model(images)
        preds = logits.argmax(dim=1)
        all_y.extend(labels.tolist())
        all_pred.extend(preds.cpu().tolist())
        all_images.append(images.cpu())
    return np.array(all_y), np.array(all_pred), all_images


@torch.no_grad()
def measure_latency_ms(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_vit: bool = False,
    max_batches: int = 20,
) -> float:
    model.eval()
    times: list[float] = []
    n = 0
    for images, _ in loader:
        if n >= max_batches:
            break
        images = images.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if is_vit:
            _ = model(pixel_values=images).logits
        else:
            _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000 / images.size(0)
        times.append(elapsed)
        n += 1
    return float(np.mean(times)) if times else 0.0


def load_resnet_checkpoint(path: Path, device: torch.device) -> nn.Module:
    model = build_resnet18(pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


def load_vit_checkpoint(path: Path, device: torch.device) -> ViTForImageClassification:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_id = payload.get("model_id", "google/vit-base-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained(model_id, num_labels=2)
    model.load_state_dict(payload["state_dict"])
    return model.to(device)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    is_vit: bool = False,
) -> dict[str, Any]:
    y_true, y_pred, _ = predict_loader(model, test_loader, device, is_vit=is_vit)
    metrics = compute_metrics(y_true, y_pred)
    latency = measure_latency_ms(model, test_loader, device, is_vit=is_vit)
    size_mb = model_size_mb(model)
    return {
        "metrics": metrics,
        "infer_latency_ms": latency,
        "model_size_mb": size_mb,
        "y_true": y_true,
        "y_pred": y_pred,
    }
