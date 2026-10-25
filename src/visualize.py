"""Plots: curves, confusion matrix, misclassified samples, Grad-CAM, ViT attention."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import Sample
from src.metrics import CLASS_NAMES


def plot_training_history(history: list[dict], out_path: Path, title: str = "") -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], label="val macro-F1")
    axes[1].plot(epochs, [h["val_accuracy"] for h in history], label="val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].set_title("Validation metrics")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray | list, out_path: Path, title: str = "") -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_misclassified_grid(
    samples: list[Sample],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    max_images: int = 12,
    image_size: int = 224,
) -> None:
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        return
    wrong_idx = wrong_idx[:max_images]
    cols = 4
    rows = int(np.ceil(len(wrong_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).flatten()
    tfm = transforms.Compose([transforms.Resize((image_size, image_size))])
    for ax, i in zip(axes, wrong_idx):
        img = Image.open(samples[i].path).convert("RGB")
        ax.imshow(tfm(img))
        ax.set_title(f"T:{CLASS_NAMES[y_true[i]]}\nP:{CLASS_NAMES[y_pred[i]]}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(wrong_idx) :]:
        ax.axis("off")
    fig.suptitle("Misclassified examples")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class ResNetGradCAM:
    """Grad-CAM for torchvision ResNet (target layer: layer4)."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target = model.layer4[-1]
        target.register_forward_hook(self._forward_hook)
        target.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inp, out):
        self.activations = out.detach()

    def _backward_hook(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.detach().requires_grad_(True)
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()
        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    from matplotlib import cm

    heatmap = cm.jet(cam)[..., :3]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    blended = (1 - alpha) * (image / 255.0) + alpha * heatmap
    return (blended * 255).clip(0, 255).astype(np.uint8)


def save_gradcam_examples(
    model: nn.Module,
    loader: DataLoader,
    out_path: Path,
    device: torch.device,
    n: int = 4,
) -> None:
    cam = ResNetGradCAM(model.to(device))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    count = 0
    for images, labels in loader:
        for i in range(images.size(0)):
            if count >= n:
                break
            img = images[i : i + 1].to(device)
            img.requires_grad_(True)
            heat = cam(img, class_idx=int(labels[i].item()))
            denorm = (images[i] * std + mean).permute(1, 2, 0).numpy()
            denorm = denorm.clip(0, 1)
            import cv2

            heat_resized = cv2.resize(heat, (denorm.shape[1], denorm.shape[0]))
            overlay = overlay_cam(denorm, heat_resized)
            axes[0, count].imshow(denorm)
            axes[0, count].set_title(f"GT: {CLASS_NAMES[labels[i]]}", fontsize=8)
            axes[0, count].axis("off")
            axes[1, count].imshow(overlay)
            axes[1, count].set_title("Grad-CAM", fontsize=8)
            axes[1, count].axis("off")
            count += 1
        if count >= n:
            break
    fig.suptitle("ResNet-18 Grad-CAM")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_vit_attention_maps(
    model,
    samples: list[Sample],
    out_path: Path,
    device: torch.device,
    image_size: int = 224,
    n: int = 4,
) -> None:
    """Average attention from CLS token to patch tokens (last layer)."""
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model.eval()
    model.to(device)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    axes = np.atleast_1d(axes).flatten()
    for ax, sample in zip(axes, samples[:n]):
        img = tfm(Image.open(sample.path).convert("RGB")).unsqueeze(0).to(device)
        outputs = model(pixel_values=img, output_attentions=True)
        # attentions: (layers, batch, heads, seq, seq)
        attn = outputs.attentions[-1][0].mean(dim=0)
        cls_attn = attn[0, 1:].cpu().numpy()
        side = int(np.sqrt(len(cls_attn)))
        if side * side != len(cls_attn):
            ax.imshow(Image.open(sample.path).convert("RGB"))
            ax.set_title("Attention N/A", fontsize=8)
        else:
            attn_map = cls_attn.reshape(side, side)
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            ax.imshow(attn_map, cmap="inferno")
            ax.set_title(IDX_LABEL(sample.label), fontsize=8)
        ax.axis("off")
    fig.suptitle("ViT CLS→patch attention (last layer mean)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def IDX_LABEL(label: int) -> str:
    return CLASS_NAMES[label]
