"""Single-image inference for demo / Streamlit."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data import build_transforms
from src.evaluate import load_resnet_checkpoint, load_vit_checkpoint
from src.metrics import CLASS_NAMES
from src.utils import load_config, project_root, resolve_device


def list_checkpoints(cfg: dict | None = None) -> list[Path]:
    cfg = cfg or load_config()
    root = project_root() / cfg["paths"]["checkpoints"]
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*/best.pt") if p.is_file())


def infer_model_type_from_run_name(run_name: str) -> str:
    name = run_name.lower()
    if "vit" in name:
        return "vit"
    if "resnet" in name:
        return "resnet"
    raise ValueError(f"Cannot infer model type from checkpoint folder: {run_name}")


def detect_checkpoint_type(checkpoint_path: Path) -> str:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return "vit"
    return "resnet"


def ensure_checkpoint_from_url(url: str, cache_name: str) -> Path:
    cache_dir = project_root() / "results" / "streamlit_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / cache_name
    if not dest.is_file():
        urlretrieve(url, dest)  # noqa: S310 — demo URLs from Hugging Face / instructor config
    return dest


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    run_name = checkpoint_path.parent.name
    try:
        model_type = infer_model_type_from_run_name(run_name)
    except ValueError:
        model_type = detect_checkpoint_type(checkpoint_path)
    if model_type == "resnet":
        return load_resnet_checkpoint(checkpoint_path, device), "resnet"
    return load_vit_checkpoint(checkpoint_path, device), "vit"


def preprocess_pil(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB")
    tfm = build_transforms(image_size, train=False)
    return tfm(image)


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    is_vit: bool,
) -> dict[str, float | int | str]:
    model.eval()
    batch = image_tensor.unsqueeze(0).to(device)
    if is_vit:
        logits = model(pixel_values=batch).logits
    else:
        logits = model(batch)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    return {
        "label": CLASS_NAMES[pred_idx],
        "class_index": pred_idx,
        "prob_acceptable": float(probs[0]),
        "prob_defective": float(probs[1]),
    }


def pil_to_display_array(image: Image.Image, size: int = 224) -> np.ndarray:
    return np.array(image.convert("RGB").resize((size, size)))
