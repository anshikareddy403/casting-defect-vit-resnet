# Runbook — after GitHub + Kaggle download

Deadline: **27 October 2026**. Training is easiest on **Google Colab (GPU)**; your **GitHub repo** holds code only (dataset is not committed).

## Step 1 — Fix the folder layout (5 min)

Inside your project copy, create:

```
casting-defect-inspection/
  data/raw/
    ok_front/     ← all acceptable images
    def_front/    ← all defective images
```

**If Kaggle gave you `casting_data/train/` and `test/`** instead of flat folders, either:

- Move/copy so you use layout B in the README (`data/raw/casting_data/train|test/...`), **or**
- Merge all `ok_front` and `def_front` images into the flat layout above (recommended only for the **original** non-augmented set).

Open a few images: grayscale casting photos, two classes only.

## Step 2 — Push code to GitHub (10 min)

From the **project root** (folder that contains `README.md`, `src/`, `notebooks/`):

```bash
git init
git add .
git commit -m "Casting defect inspection: ViT vs ResNet scaffold"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

`data/` is in `.gitignore` — that is intentional (license + size). Describe the Kaggle download in the README.

**No git?** Upload the project files via GitHub **Add file → Upload files** (exclude `data/`).

## Step 3 — Colab setup (10 min)

1. [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → `notebooks/01_data_exploration.ipynb`  
   Or: **File → Open notebook → GitHub** and paste your repo URL.
2. **Runtime → Change runtime type → T4 GPU** (needed for notebook 02).
3. Clone or sync the repo so Colab sees `src/` and `configs/`:

```python
!git clone https://github.com/YOUR_USER/YOUR_REPO.git
%cd YOUR_REPO
!pip install -q -r requirements.txt
```

4. **Get data into Colab** (pick one):

- **Drive (best for many runs):** upload `data/raw` to Google Drive, then mount and point config:

```python
from google.colab import drive
drive.mount('/content/drive')
# Edit configs/default.yaml in Colab: data.root: /content/drive/MyDrive/casting/data/raw
```

- **Session upload:** zip `data/raw`, upload to Colab, unzip to `./data/raw` (lost when runtime disconnects).

## Step 4 — Sanity check (notebook 01)

Run all cells in `notebooks/01_data_exploration.ipynb`.

You should see:

- A detected layout (`original_flat` or `kaggle_split`)
- Train / val / test counts with **both classes** in each split
- Sample images and class balance

If `FileNotFoundError`, fix paths in Step 1 or `data.root` in `configs/default.yaml`.

## Step 5 — Training runs (notebook 02 or CLI)

### Option A — Notebook (one run per session)

Open `notebooks/02_train_and_compare.ipynb`. In the second code cell, set:

```python
FRACTION = 0.25   # then 0.5, then 1.0
MODEL = 'resnet'  # then 'vit'
```

Run all cells **six times** (2 models × 3 fractions). Save checkpoints under `results/checkpoints/` on Drive if sessions reset.

### Option B — CLI (full grid in one session)

```bash
for m in resnet vit; do
  for f in 0.25 0.5 1.0; do
    python scripts/run_experiment.py --model $m --fraction $f
  done
done
```

Results append to `results/comparison_table.csv`.

**Suggested order:** `resnet` @ `1.0` first (fastest sanity check), then `vit` @ `1.0`, then smaller fractions.

## Step 6 — Report & submission

1. Copy metrics from `results/comparison_table.csv` into `report/TECHNICAL_REPORT.md` (`[INSERT RESULT]`).
2. Export figures from `results/figures/` into the report/PPT.
3. Push **code updates** to GitHub; optionally commit `results/comparison_table.csv` and key figures (remove those paths from `.gitignore` if you want them in the repo).
4. Put the **GitHub URL** and **Colab notebook link** in the report.

## If Colab disconnects

- Save `results/` and checkpoints to Drive after each run.
- Reduce `epochs` in `configs/default.yaml` only if you document it in the report (prefer full settings for the final submission).
