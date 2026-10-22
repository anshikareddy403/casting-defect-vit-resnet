# Colab guide — Part 4 & 5 (detailed)

**Before you start:** Part 3 done (GitHub has `src`, `notebooks`, …). Part 2 done (`data\raw\casting_data\` on your PC).

---

## Part 4 — Notebook 01 (check data only, ~15–20 min)

Goal: prove Colab can see your images and splits. **No training yet.**

### 4.1 Open Colab

1. Browser → **https://colab.research.google.com**
2. Sign in with Google.
3. **File → Upload notebook**
4. Pick from your PC:  
   `...\casting-defect-inspection\notebooks\01_data_exploration.ipynb`

### 4.2 Turn on GPU (optional for this notebook)

**Runtime → Change runtime type → T4 GPU → Save**  
(CPU also works for Part 4.)

### 4.3 Add ONE setup cell at the very top

Click **+ Code** above the first cell. Paste and run (**Shift+Enter**):

```python
# --- Colab setup (run once per session) ---
!pip install -q pillow pandas matplotlib scikit-learn pyyaml torch torchvision transformers

!git clone https://github.com/anshikareddy403/casting-defect-vit-resnet.git
%cd /content/casting-defect-vit-resnet

import sys
from pathlib import Path
ROOT = Path('/content/casting-defect-vit-resnet')
sys.path.insert(0, str(ROOT))
print('Project root:', ROOT)
```

You should see `Project root: /content/casting-defect-vit-resnet`.

### 4.4 Put your dataset where the code expects it

The zip on your PC is **not** on Colab until you upload it.

```python
from google.colab import files
from pathlib import Path

ROOT = Path('/content/casting-defect-vit-resnet')
raw = ROOT / 'data' / 'raw'
raw.mkdir(parents=True, exist_ok=True)
print('Choose casting_data.zip from your PC')
uploaded = files.upload()
for name, data in uploaded.items():
    (raw / name).write_bytes(data)

!unzip -q -o /content/casting-defect-vit-resnet/data/raw/casting_data.zip -d /content/casting-defect-vit-resnet/data/raw
!ls /content/casting-defect-vit-resnet/data/raw/casting_data/train/
```

**Check:** this path must exist:

`/content/casting-defect-vit-resnet/data/raw/casting_data/train/ok_front`

(Colab: Files → click through folders.)

You do **not** need to change `configs/default.yaml` if you use this path.

### 4.5 Run the rest of notebook 01

**Runtime → Run all** (or run each cell top to bottom).

**Success:**

- `Detected layout: kaggle_split`
- Numbers for train / val / test (all > 0)
- Table with **acceptable** and **defective** in each row
- Grid of gray casting photos
- `Overlapping groups between train and val: 0`

**If red error:** copy full error text → Project chat.

---

## Part 5 — Notebook 02 (train models, ~2–6 hours total for all runs)

Goal: train **6 models**, save metrics to CSV.

| Run # | MODEL      | FRACTION | Notes              |
|-------|------------|----------|--------------------|
| 1     | `'resnet'` | `1.0`    | **Do this first** (fastest test) |
| 2     | `'vit'`    | `1.0`    | ViT, full data     |
| 3     | `'resnet'` | `0.5`    |                    |
| 4     | `'vit'`    | `0.5`    |                    |
| 5     | `'resnet'` | `0.25`   |                    |
| 6     | `'vit'`    | `0.25`   |                    |

### 5.1 Open notebook 02

**File → Upload notebook** → `02_train_and_compare.ipynb`  
(New Colab tab is fine.)

### 5.2 GPU required

**Runtime → Change runtime type → T4 GPU → Save**

### 5.3 Same setup as Part 4

Run the **same setup cell** from 4.3, then **upload/unzip `casting_data` again** (Colab deletes files when the session ends).

### 5.4 Change model settings (one run at a time)

Find the cell with:

```python
FRACTION = 0.25
MODEL = 'resnet'
```

For **Run 1**, change to:

```python
FRACTION = 1.0
MODEL = 'resnet'
```

**Runtime → Run all.** Wait until it finishes (may take 10–30+ minutes).

Then edit the same cell for Run 2, 3, … and **Run all** again.

### 5.5 What you should see after each run

- Training progress / epochs
- A small **pandas table** with accuracy, F1, defect recall, etc.
- Files under **Files → casting-defect-vit-resnet → results**  
  (`comparison_table.csv`, `checkpoints/`, `figures/`)

### 5.6 Save results before Colab disconnects

Download **`results/comparison_table.csv`** to your PC (right-click → Download).

Optional: zip `results` folder and save to Google Drive.

### 5.7 Use results in your report

Open **`report/TECHNICAL_REPORT.md`** on your PC (or on GitHub). Replace **`[INSERT RESULT]`** with numbers from the CSV.

---

## Common problems

| Problem | Fix |
|---------|-----|
| `FileNotFoundError` for images | Fix path in 4.4 — `casting_data` must be under `data/raw/` |
| `ModuleNotFoundError: src` | Run setup cell; `%cd` must be the cloned repo |
| Colab says session crashed | Shorter run first (ResNet 1.0); reconnect GPU |
| ViT very slow | Normal; run ResNet runs while you have time for ViT |

---

**Next (Part 6):** write report + PPT + submit before **27 Oct 2026**.
