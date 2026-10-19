"""Dataset discovery, leakage-safe splits, and PyTorch dataloaders."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils import project_root

CLASS_TO_IDX = {"ok_front": 0, "acceptable": 0, "def_front": 1, "defective": 1}
IDX_TO_CLASS = {0: "acceptable", 1: "defective"}
FOLDER_ALIASES = {
    "ok_front": 0,
    "def_front": 1,
}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    group_id: str


def _group_id_from_filename(path: Path) -> str:
    """Map augmented Kaggle names (e.g. cast_ok_12_3.jpeg) to a source image group."""
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "cast":
        return "_".join(parts[:-1])
    return stem


def _collect_from_class_dirs(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for folder_name, label in FOLDER_ALIASES.items():
        class_dir = root / folder_name
        if not class_dir.is_dir():
            continue
        for p in sorted(class_dir.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                samples.append(Sample(path=p, label=label, group_id=_group_id_from_filename(p)))
    return samples


def detect_layout(data_root: Path) -> str:
    """Return layout key: original_flat | kaggle_split | unknown."""
    data_root = Path(data_root)
    if (data_root / "def_front").is_dir() and (data_root / "ok_front").is_dir():
        return "original_flat"
    train = data_root / "casting_data" / "train"
    if not train.is_dir():
        train = data_root / "train"
    test = data_root / "casting_data" / "test"
    if not test.is_dir():
        test = data_root / "test"
    if train.is_dir() and test.is_dir():
        return "kaggle_split"
    return "unknown"


def load_all_samples(data_root: Path, layout: str = "auto") -> tuple[list[Sample], list[Sample], str]:
    data_root = Path(data_root)
    if layout == "auto":
        layout = detect_layout(data_root)

    if layout == "original_flat":
        all_samples = _collect_from_class_dirs(data_root)
        if not all_samples:
            raise FileNotFoundError(f"No images under {data_root} (expected ok_front/ def_front/)")
        groups = np.array([s.group_id for s in all_samples])
        labels = np.array([s.label for s in all_samples])
        idx = np.arange(len(all_samples))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, stratify=labels, random_state=42
        )
        train_pool = [all_samples[i] for i in train_idx]
        test_samples = [all_samples[i] for i in test_idx]
        return train_pool, test_samples, layout

    if layout == "kaggle_split":
        train_root = data_root / "casting_data" / "train"
        test_root = data_root / "casting_data" / "test"
        if not train_root.is_dir():
            train_root = data_root / "train"
            test_root = data_root / "test"
        train_pool = _collect_from_class_dirs(train_root)
        test_samples = _collect_from_class_dirs(test_root)
        if not train_pool or not test_samples:
            raise FileNotFoundError(f"Missing train/test under {data_root}")
        return train_pool, test_samples, layout

    raise ValueError(
        f"Unrecognized dataset layout under {data_root}. "
        "See README for expected casting_data/train|test or flat ok_front/def_front."
    )


def split_train_val(
    train_pool: list[Sample],
    val_fraction: float,
    seed: int,
    group_augmentations: bool = True,
) -> tuple[list[Sample], list[Sample]]:
    if not train_pool:
        return [], []

    if group_augmentations:
        group_to_samples: dict[str, list[Sample]] = {}
        for s in train_pool:
            group_to_samples.setdefault(s.group_id, []).append(s)
        group_ids = list(group_to_samples.keys())
        group_labels = [group_to_samples[g][0].label for g in group_ids]
        g_train, g_val = train_test_split(
            group_ids,
            test_size=val_fraction,
            stratify=group_labels,
            random_state=seed,
        )
        train_samples = [s for g in g_train for s in group_to_samples[g]]
        val_samples = [s for g in g_val for s in group_to_samples[g]]
        return train_samples, val_samples

    labels = [s.label for s in train_pool]
    idx = np.arange(len(train_pool))
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_fraction, stratify=labels, random_state=seed
    )
    train_samples = [train_pool[i] for i in tr_idx]
    val_samples = [train_pool[i] for i in va_idx]
    return train_samples, val_samples


def subset_train_stratified(
    train_samples: list[Sample],
    fraction: float,
    seed: int,
    group_augmentations: bool = True,
) -> list[Sample]:
    if fraction >= 1.0:
        return list(train_samples)
    if group_augmentations:
        group_to_samples: dict[str, list[Sample]] = {}
        for s in train_samples:
            group_to_samples.setdefault(s.group_id, []).append(s)
        group_ids = list(group_to_samples.keys())
        group_labels = [group_to_samples[g][0].label for g in group_ids]
        g_subset, _ = train_test_split(
            group_ids,
            train_size=fraction,
            stratify=group_labels,
            random_state=seed,
        )
        return [s for g in g_subset for s in group_to_samples[g]]

    labels = [s.label for s in train_samples]
    idx = np.arange(len(train_samples))
    sub_idx, _ = train_test_split(
        idx, train_size=fraction, stratify=labels, random_state=seed
    )
    return [train_samples[i] for i in sub_idx]


class CastingImageDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: Callable | None = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        img = Image.open(sample.path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, sample.label


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_loaders(
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
    image_size: int,
    batch_size: int,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = CastingImageDataset(train_samples, build_transforms(image_size, train=True))
    val_ds = CastingImageDataset(val_samples, build_transforms(image_size, train=False))
    test_ds = CastingImageDataset(test_samples, build_transforms(image_size, train=False))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def resolve_data_root(cfg: dict) -> Path:
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        root = project_root() / root
    return root
