# Data-Efficient Visual Quality Inspection: Comparing Vision Transformers and CNNs for Casting Defect Classification

**Author:** [YOUR NAME]  
**Program:** [Masters Program, University]  
**Supervisor:** [SUPERVISOR NAME]  
**Date:** October 2026

---

## Abstract

Automated visual inspection of cast metal components reduces scrap, rework, and safety risk in high-volume manufacturing. This study compares a **Vision Transformer (ViT-B/16)** implemented via Hugging Face Transformers with a **ResNet-18** convolutional baseline on the Pilot Technocast casting impeller image dataset. Experiments emphasize **data efficiency**: models are trained on stratified subsets containing 25%, 50%, and 100% of the available training pool while validation and test sets remain fixed. We report accuracy, macro-F1, defect-class precision and recall, confusion matrices, training time, inference latency, and model size, and provide qualitative interpretability through Grad-CAM (ResNet) and CLS-to-patch attention maps (ViT). Results indicate [INSERT RESULT — e.g., whether ViT closes the gap with CNNs under low-label regimes]. The work aligns with UAE industrial diversification goals under **Operation 300bn** by documenting transferable, data-frugal quality-inspection workflows suitable for SMEs adopting AI-enabled manufacturing.

---

## 1. Introduction

Cast submersible pump impellers must meet strict surface-quality standards; human inspection is slow, subjective, and costly at scale. Deep learning offers consistent binary decisions (acceptable vs defective) from top-view camera images captured under controlled lighting. Two architectural families dominate image classification: **convolutional neural networks (CNNs)**, which encode local translation-equivariant features, and **vision transformers (ViTs)**, which model long-range dependencies via self-attention on patch tokens.

Research questions:

1. How do ViT-B/16 and ResNet-18 compare on casting defect detection when trained on **limited labeled data**?
2. Which architecture offers a better trade-off among **accuracy**, **training cost**, and **inference latency** for deployment on factory edge hardware?
3. Do attention-based explanations align with metallurgical defect regions perceived by human inspectors?

Contributions: a reproducible PyTorch/Hugging Face pipeline; leakage-aware splitting for augmented Kaggle releases; a six-run experiment matrix; and interpretability artifacts for thesis defense.

---

## 2. Related work

**Vision Transformers.** Dosovitskiy et al. (2020) introduced ViT, demonstrating that pure transformers match or exceed CNNs on large-scale classification when pretrained on massive data and fine-tuned on downstream tasks. Their patch embedding and positional encoding design motivates our use of `google/vit-base-patch16-224-in21k`.

**Residual CNNs.** He et al. (2016) proposed ResNet with skip connections, enabling very deep networks and strong ImageNet transfer learning—standard practice for industrial vision baselines such as our ResNet-18.

**Industrial surface-defect detection.** Tabernik et al. (2020) presented a segmentation-based deep-learning approach for surface-defect detection on production lines, highlighting class imbalance and small defect regions—challenges analogous to casting flaw localization.

**Data augmentation and label efficiency.** Shorten and Khoshgoftaar (2019) surveyed image augmentation for deep learning; we cite their guidance when documenting the Kaggle dataset’s pre-applied augmentations and our **group-wise splitting** to avoid leakage across duplicated views.

**Casting quality inspection with CNNs.** Industrial case studies on the same Kaggle casting dataset (e.g., blog and notebook tutorials, 2020–2021) establish CNN feasibility on `def_front` / `ok_front` classes; this thesis extends that line by **comparing ViT**, reporting **data-efficiency curves**, and standardizing metrics for academic evaluation.

---

## 3. Methodology

The project follows five stages aligned with the masters workflow:

### Stage 1 — Problem definition and literature review

Formalize binary inspection, select ViT-B/16 and ResNet-18, and define metrics and data-efficiency protocol (Sections 1–2).

### Stage 2 — Dataset acquisition and preprocessing

- **Source:** Kaggle *Real-life industrial dataset of casting product* (grayscale top-view impellers; classes `ok_front`, `def_front`).
- **Preferred split:** original **non-augmented** ~1,300 images; stratified 80/20 train-pool/test, then 15% validation from train pool.
- **Alternative:** Kaggle `train/` / `test/` augmented folders; **group IDs** parsed from filenames (`cast_ok_12_3` → group `cast_ok_12`) to keep augmented siblings out of conflicting splits.
- **Preprocessing:** resize to 224×224, replicate grayscale to RGB for ImageNet-pretrained backbones; ImageNet normalization; light train-time augmentation (horizontal flip, ±10° rotation).

### Stage 3 — Model implementation

| Component | ViT | ResNet-18 |
|-----------|-----|-----------|
| Library | Hugging Face `transformers` | `torchvision.models` |
| Weights | `google/vit-base-patch16-224-in21k` | ImageNet-1K v1 |
| Head | 2-class linear classifier | replaced `fc` layer |
| Optimizer | AdamW, lr 2e-5, wd 0.01 | AdamW, lr 1e-4, wd 1e-4 |
| Loss | cross-entropy | cross-entropy |

Implementation modules: `src/data.py`, `src/train_vit.py`, `src/train_resnet.py`.

### Stage 4 — Training under data-efficiency protocol

- **Train fractions:** 0.25, 0.50, 1.00 (stratified on training pool; group-aware when augmented).
- **Fixed validation and test** for all runs.
- **Early stopping** on validation **macro-F1** (patience 5, max 30 epochs).
- **Seed:** 42 for splits and PyTorch/NumPy.

### Stage 5 — Evaluation and interpretability

- **Quantitative:** accuracy, macro-F1, defective precision/recall, confusion matrices, wall-clock training time, per-image inference latency (ms), model size (MB).
- **Qualitative:** training curves, misclassified grids, ResNet Grad-CAM (`src/visualize.py`), ViT last-layer CLS attention averaged over heads.

---

## 4. Evaluation plan

| Experiment ID | Model | Train fraction | Expected checkpoint |
|---------------|-------|----------------|---------------------|
| R-25 | ResNet-18 | 25% | `results/checkpoints/resnet_frac0.25_*` |
| R-50 | ResNet-18 | 50% | … |
| R-100 | ResNet-18 | 100% | … |
| V-25 | ViT-B/16 | 25% | … |
| V-50 | ViT-B/16 | 50% | … |
| V-100 | ViT-B/16 | 100% | … |

**Hypotheses (to be confirmed):**

- H1: At 100% labels, ViT matches or exceeds ResNet macro-F1 [INSERT RESULT].
- H2: Below 50% labels, ResNet retains higher defect recall [INSERT RESULT].
- H3: ViT training time and latency exceed ResNet at comparable batch size [INSERT RESULT].

Statistical note: with a single stratified split, report confidence intervals via bootstrap over test predictions (optional extension).

---

## 5. Results

### 5.1 Summary table

| Model | Train % | Accuracy | Macro-F1 | Defect P | Defect R | Train (s) | Latency (ms) | Size (MB) |
|-------|---------|----------|----------|----------|----------|-------------|--------------|-----------|
| ResNet-18 | 25 | [INSERT RESULT] | [INSERT RESULT] | [INSERT RESULT] | [INSERT RESULT] | [INSERT RESULT] | [INSERT RESULT] | [INSERT RESULT] |
| ResNet-18 | 50 | [INSERT RESULT] | … | … | … | … | … | … |
| ResNet-18 | 100 | [INSERT RESULT] | … | … | … | … | … | … |
| ViT-B/16 | 25 | [INSERT RESULT] | … | … | … | … | … | … |
| ViT-B/16 | 50 | [INSERT RESULT] | … | … | … | … | … | … |
| ViT-B/16 | 100 | [INSERT RESULT] | … | … | … | … | … | … |

Populate from `results/comparison_table.csv` after running `scripts/run_experiment.py`.

### 5.2 Figures (placeholders)

- Figure 1: Dataset class distribution — `[INSERT FIGURE: notebooks/01]`
- Figure 2: Macro-F1 vs train fraction — `[INSERT FIGURE]`
- Figure 3: Confusion matrices (6-panel) — `[INSERT FIGURE]`
- Figure 4: Grad-CAM vs ViT attention — `[INSERT FIGURE]`

### 5.3 Discussion

[INSERT RESULT narrative: data-efficiency crossover point, failure modes on subtle defects, false positives on lighting artifacts.]

---

## 6. Ethics, UAE context, and responsible deployment

- **Dataset consent:** Kaggle terms require attribution; images are real production captures—avoid re-identification of facility metadata in publications.
- **Bias & safety:** Models optimized for offline accuracy may **fail under lighting drift**; false negatives (missed defects) carry higher ethical weight than false positives in safety-critical pumps— tune threshold on validation defect recall.
- **UAE Operation 300bn:** The UAE’s industrial growth strategy emphasizes advanced manufacturing and technology adoption. Deploying **data-efficient** inspection reduces dependency on large labeled datasets—a practical constraint for regional foundries and SMEs joining Industry 4.0 programs. Document energy/compute use (GPU hours) for sustainability reporting.

---

## 7. Limitations and future work

- Single dataset and camera geometry; generalization to other cast parts unverified.
- Binary labels ignore defect type/severity segmentation (cf. Tabernik et al., 2020).
- No hyperparameter search budget; ViT may benefit from layer-wise learning rates.
- Future: test-time augmentation, lightweight ViT distillation, Pi/edge deployment benchmark.

---

## 8. Conclusion

This thesis scaffold implements a fair comparison between ViT-B/16 and ResNet-18 for casting defect classification under controlled data budgets. After completing the six scheduled runs, final conclusions will state [INSERT RESULT — preferred architecture for low-label vs full-label regimes and deployment recommendation].

---

## References (APA-style, no page numbers)

- Dosovitskiy, A., et al. (2020). *An image is worth 16x16 words: Transformers for image recognition at scale.*
- He, K., et al. (2016). *Deep residual learning for image recognition.*
- Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on image data augmentation for deep learning.*
- Tabernik, D., et al. (2020). *Segmentation-based deep-learning approach for surface-defect detection.*
- Ravirajsinh, R., et al. (2020). *Real-life industrial dataset of casting product* [Kaggle dataset].

---

## Appendix A — Reproducibility checklist

- [ ] Kaggle data placed in `data/raw/`
- [ ] `configs/default.yaml` committed
- [ ] Six CLI runs completed
- [ ] GitHub repo URL: `[YOUR_GITHUB_REPO_URL]`
- [ ] Colab link: `[YOUR_COLAB_NOTEBOOK_URL]`
