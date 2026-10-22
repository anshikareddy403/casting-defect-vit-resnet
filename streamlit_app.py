"""Streamlit deployment — live demo, experiment results, project overview."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.inference import (
    ensure_checkpoint_from_url,
    list_checkpoints,
    load_model_from_checkpoint,
    pil_to_display_array,
    predict_image,
    preprocess_pil,
)
from src.utils import load_config, project_root, resolve_device
from src.visualize import ResNetGradCAM, overlay_cam

st.set_page_config(
    page_title="Casting defect inspection",
    page_icon="🔍",
    layout="wide",
)

CFG = load_config()
IMAGE_SIZE = int(CFG["data"]["image_size"])
CSV_PATH = project_root() / CFG["paths"]["comparison_csv"]


@st.cache_resource
def get_device():
    return resolve_device("cpu")  # Streamlit Cloud has no GPU; CPU is enough for single-image demo


@st.cache_resource
def load_cached_model(checkpoint_str: str, _device_str: str):
    device = torch.device(_device_str)
    path = Path(checkpoint_str)
    model, model_type = load_model_from_checkpoint(path, device)
    return model, model_type


def resolve_checkpoint_sidebar() -> Path | None:
    st.sidebar.header("Model weights")
    checkpoints = list_checkpoints(CFG)
    try:
        hf = dict(st.secrets.get("huggingface", {}))
    except Exception:
        hf = {}
    preset_resnet = hf.get("resnet_checkpoint")
    preset_vit = hf.get("vit_checkpoint")

    preset_choice = st.sidebar.selectbox(
        "Preset (Hugging Face / URL in secrets)",
        ["None", "ResNet (secrets)", "ViT (secrets)"],
    )
    if preset_choice == "ResNet (secrets)" and preset_resnet:
        return ensure_checkpoint_from_url(preset_resnet, "resnet_best.pt")
    if preset_choice == "ViT (secrets)" and preset_vit:
        return ensure_checkpoint_from_url(preset_vit, "vit_best.pt")

    uploaded_ckpt = st.sidebar.file_uploader("Or upload best.pt", type=["pt"])
    if uploaded_ckpt is not None:
        cache_dir = project_root() / "results" / "streamlit_uploads"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / uploaded_ckpt.name
        path.write_bytes(uploaded_ckpt.getvalue())
        return path

    if checkpoints:
        labels = [str(p.parent.name) for p in checkpoints]
        choice = st.sidebar.selectbox("Local checkpoint", labels, index=0)
        return checkpoints[labels.index(choice)]

    st.sidebar.info("Add Hugging Face URLs in Streamlit **Secrets**, upload `best.pt`, or train locally.")
    return None


def gradcam_overlay(model, tensor, device) -> np.ndarray:
    cam_engine = ResNetGradCAM(model.to(device))
    img = tensor.unsqueeze(0).to(device)
    img.requires_grad_(True)
    pred = int(model(img).argmax(dim=1).item())
    heat = cam_engine(img, class_idx=pred)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm = (tensor.cpu() * std + mean).permute(1, 2, 0).numpy().clip(0, 1)
    import cv2

    heat_resized = cv2.resize(heat, (denorm.shape[1], denorm.shape[0]))
    return overlay_cam(denorm, heat_resized)


def page_live_demo() -> None:
    st.header("Live inspection demo")
    st.caption("Upload a casting image — binary QC: acceptable vs defective.")

    device = get_device()
    checkpoint_path = resolve_checkpoint_sidebar()

    file = st.file_uploader("Casting image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if file is None:
        st.info("Upload a test image from `casting_data/test/ok_front` or `def_front`.")
        return

    pil_image = Image.open(file)
    if checkpoint_path is None or not checkpoint_path.is_file():
        st.image(pil_image, caption="No model loaded yet", use_container_width=True)
        return

    model, model_type = load_cached_model(str(checkpoint_path.resolve()), str(device))
    is_vit = model_type == "vit"
    tensor = preprocess_pil(pil_image, IMAGE_SIZE)
    result = predict_image(model, tensor, device, is_vit=is_vit)

    c1, c2 = st.columns(2)
    with c1:
        st.image(pil_to_display_array(pil_image, IMAGE_SIZE), use_container_width=True)
    with c2:
        st.metric("Prediction", str(result["label"]).upper())
        st.write(f"Acceptable: **{result['prob_acceptable']:.1%}**")
        st.write(f"Defective: **{result['prob_defective']:.1%}**")
        st.caption(f"Checkpoint: `{checkpoint_path.name}` · {model_type.upper()} · CPU")

    if not is_vit and st.checkbox("Grad-CAM (ResNet)", value=True):
        st.image(gradcam_overlay(model, tensor, device), caption="Grad-CAM", use_container_width=True)


def load_results_df() -> pd.DataFrame | None:
    uploaded = st.file_uploader("Upload comparison_table.csv from Colab", type=["csv"])
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if CSV_PATH.is_file():
        return pd.read_csv(CSV_PATH)
    bundled = project_root() / "results" / "comparison_table_demo.csv"
    if bundled.is_file():
        return pd.read_csv(bundled)
    return None


def page_results() -> None:
    st.header("Experiment results (ViT vs ResNet, data efficiency)")
    df = load_results_df()
    if df is None or df.empty:
        st.warning(
            "Upload `results/comparison_table.csv` after notebook 02 runs, "
            "or commit a copy as `results/comparison_table_demo.csv` for the public app."
        )
        return

    st.dataframe(df, use_container_width=True)

    if {"model", "train_fraction", "accuracy"}.issubset(df.columns):
        st.subheader("Accuracy vs training fraction")
        pivot = df.pivot_table(index="train_fraction", columns="model", values="accuracy", aggfunc="mean")
        st.line_chart(pivot)
    if "defect_recall" in df.columns:
        st.subheader("Defect recall (production-critical)")
        pivot_r = df.pivot_table(index="train_fraction", columns="model", values="defect_recall", aggfunc="mean")
        st.line_chart(pivot_r)


def page_about() -> None:
    st.header("Project overview")
    st.markdown(
        """
**Aim:** Compare **Hugging Face ViT-B/16** and **ResNet-18** for casting defect classification
under limited labelled data (25% / 50% / 100% training subsets).

**Dataset:** Kaggle *Real-life industrial dataset of casting product* (organic photos, binary labels).

**Metrics:** Accuracy, macro-F1, defect precision/recall, training time, latency, model size.

**Ethics / UAE:** Supports industrial QC goals (Operation 300bn); models need human oversight,
monitor defect recall, and avoid deploying beyond training conditions.

**Repo:** [github.com/anshikareddy403/casting-defect-vit-resnet](https://github.com/anshikareddy403/casting-defect-vit-resnet)
        """
    )


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Live demo", "Results", "About"], label_visibility="collapsed")

    if page == "Live demo":
        page_live_demo()
    elif page == "Results":
        page_results()
    else:
        page_about()


if __name__ == "__main__":
    main()
