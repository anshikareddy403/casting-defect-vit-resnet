# Best way to demo on Streamlit (recommended)

## Recommended setup (for class + report URL)

| Piece | Where | Why |
|--------|--------|-----|
| **App code** | GitHub → **Streamlit Community Cloud** | Free public URL; professor opens one link |
| **Model weights** | **Hugging Face Hub** (public model repos) | Fits course HF requirement; no manual upload at demo time |
| **Result tables** | Small CSV in GitHub (`results/comparison_table_demo.csv`) | Results tab works without Colab |
| **Training** | Still **Google Colab** | GPU for fine-tuning; Streamlit is **inference + presentation only** |

Do **not** train inside Streamlit — only show pre-trained weights and pre-computed metrics.

---

## Step-by-step deployment

### 1 — Finish at least two Colab runs

Minimum for a convincing demo:

- `resnet` @ fraction `1.0` → download `results/checkpoints/resnet_frac1.0_seed42/best.pt`
- `vit` @ fraction `1.0` → download `results/checkpoints/vit_frac1.0_seed42/best.pt`
- Download `results/comparison_table.csv` (or save a copy as `comparison_table_demo.csv`)

### 2 — Publish weights on Hugging Face Hub

1. [huggingface.co](https://huggingface.co) → **New model** (two repos or one with two files).
2. Upload each `best.pt` (e.g. `casting-resnet-demo`, `casting-vit-demo`).
3. Copy **resolve** URLs, e.g.  
   `https://huggingface.co/YOUR_USER/casting-resnet-demo/resolve/main/best.pt`

### 3 — Push app files to GitHub

Include:

- `streamlit_app.py`, `src/` (including `inference.py`), `configs/`
- `requirements.txt`, `.streamlit/config.toml`
- `results/comparison_table_demo.csv` (your metrics; no huge files)
- **Not** `data/` or full checkpoints (use HF instead)

### 4 — Deploy on Streamlit Cloud

1. [share.streamlit.io](https://share.streamlit.io) → **Create app**
2. Repository: `anshikareddy403/casting-defect-vit-resnet`
3. Main file: **`streamlit_app.py`**
4. **Advanced settings → Secrets** (paste):

```toml
[huggingface]
resnet_checkpoint = "https://huggingface.co/YOUR_USER/casting-resnet-demo/resolve/main/best.pt"
vit_checkpoint = "https://huggingface.co/YOUR_USER/casting-vit-demo/resolve/main/best.pt"
```

5. Deploy → share the `*.streamlit.app` URL in your report and PPT.

### 5 — Live demo flow (2–3 min)

1. **About** tab — problem, dataset, RQs, ethics.
2. **Results** tab — accuracy & defect recall vs 25/50/100% (your CSV).
3. **Live demo** tab — preset **ResNet** → upload defective image → Grad-CAM → switch **ViT** → same image.

---

## Fallback: laptop only (no cloud)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Upload `best.pt` in the sidebar each session — fine for a lab machine, weaker for a shareable link.

---

## What to put in the report

- Streamlit public URL  
- Screenshot of Live demo + Results tabs  
- Hugging Face model links  
- Note: inference on CPU via Streamlit; training on Colab GPU  
