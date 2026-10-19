# Data-Efficient Visual Quality Inspection: ViT vs ResNet for Casting Defects

Masters research codebase comparing **Hugging Face ViT-B/16** and **torchvision ResNet-18** on the Kaggle *Real-life industrial dataset of casting product* (binary: defective vs acceptable).

## Dataset license & ethics

- Source: [Kaggle — casting product image data](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (Pilot Technocast Ltd.; CC BY-SA 4.0 on Kaggle).
- Use only for research/education; attribute the dataset authors in publications.
- **Prefer the original non-augmented release** (~1,300 grayscale top-view impeller images) to avoid inflated sample counts from rotations/flips. If you use the augmented 300×300 split, enable `group_augmentations: true` in config (default) so variants of the same capture stay in one split.

## Expected folder layout

Place data under `data/raw/` (not committed). Two supported layouts:

### A — Original non-augmented (recommended)

```
data/raw/
  ok_front/
    cast_ok_*.jpeg
  def_front/
    cast_def_*.jpeg
```

The code performs an **80/20 stratified train+val pool / test** split (seed 42), then holds out **15% of the train pool for validation**.

### B — Kaggle pre-split augmented (300×300)

```
data/raw/
  casting_data/
    train/
      ok_front/
      def_front/
    test/
      ok_front/
      def_front/
```

Or the same structure with `train/` and `test/` directly under `data/raw/` if your unzip path differs.

Class folders **`ok_front`** (acceptable, label 0) and **`def_front`** (defective, label 1).

## Setup (local)

```bash
cd docs/casting-defect-inspection   # or repo root after copying to GitHub
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download Kaggle data (requires [Kaggle API](https://www.kaggle.com/docs/api)):

```bash
mkdir -p data/raw
kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product -p data/raw
unzip data/raw/*.zip -d data/raw
# Move ok_front/def_front or casting_data/ so paths match layouts above.
```

## Google Colab

1. Upload this repo or clone from GitHub (`[YOUR_GITHUB_REPO_URL]`).
2. **Runtime → Change runtime type → GPU** (T4 or better).
3. Install deps: `pip install -r requirements.txt`
4. Optional: mount Drive and set `data.root` in `configs/default.yaml` to your Drive path.
5. Open `notebooks/01_data_exploration.ipynb` then `02_train_and_compare.ipynb`.

## Run experiments (CLI)

Single run (appends to `results/comparison_table.csv`):

```bash
python scripts/run_experiment.py --model resnet --fraction 0.25
python scripts/run_experiment.py --model vit --fraction 0.25
```

Full grid (bash):

```bash
for m in resnet vit; do
  for f in 0.25 0.5 1.0; do
    python scripts/run_experiment.py --model $m --fraction $f
  done
done
```

Copy `results/comparison_table_template.csv` before a fresh study if you want a clean sheet.

## Project layout

```
configs/default.yaml      # hyperparameters, fractions, model IDs
src/                      # data, training, metrics, visualization
scripts/run_experiment.py
notebooks/
results/                  # CSV, checkpoints, figures (gitkeep only in template)
report/                   # technical report & presentation outline
```

## GitHub setup (first time)

### No GitHub account yet

1. Go to [https://github.com/signup](https://github.com/signup) and create a **free** account (use your university email if the course asks for it).
2. Verify your email when GitHub prompts you.
3. Sign in → **+** (top right) → **New repository**.
4. Name it e.g. `casting-defect-vit-resnet`, leave it **Public** unless your instructor requires private, **do not** add a README (this folder already has one).
5. Copy the repo URL (looks like `https://github.com/YOUR_USERNAME/casting-defect-vit-resnet.git`).

You only need GitHub to host the code link for the report; training still runs on Colab.

### Push this project

```bash
git init
git add .
git commit -m "Initial casting defect inspection scaffold"
git branch -M main
git remote add origin https://github.com/anshikareddy403/casting-defect-vit-resnet.git
git push -u origin main
```

Replace placeholders in this README and in `report/TECHNICAL_REPORT.md` after creating the remote.

## Reproducibility

- Global seed: `configs/default.yaml` → `seed: 42`
- Early stopping on **validation macro-F1** (patience 5)
- Same test set and validation split for all model/fraction runs

## Links (fill after submission)

- GitHub: `[YOUR_GITHUB_REPO_URL]`
- Colab: `[YOUR_COLAB_NOTEBOOK_URL]`
- Thesis / report PDF: `[LINK]`
