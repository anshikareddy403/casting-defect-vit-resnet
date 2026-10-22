# Notebook 02 — training (after Part 4 works)

Runtime → **T4 GPU** required. Restart session → run Cell 2–3 from `COLAB_01_EVERY_CELL.md` first (setup + data upload), then continue here.

---

## Cell 1 (Markdown)

```markdown
# 02 — Train ViT & ResNet (change MODEL and FRACTION each run)
```

---

## Cell 2 (Code) — Same as 01 Cell 2 + 3 combined shortcut

Run **01 Cell 2** then **01 Cell 3** before this notebook, OR paste setup+upload from notebook 01.

---

## Cell 3 (Code) — Pick model run (edit before each of 6 runs)

```python
import os, sys
from pathlib import Path
import pandas as pd
import torch

ROOT = Path('/content/casting-defect-vit-resnet')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.data import (
    load_all_samples, make_loaders, resolve_data_root,
    split_train_val, subset_train_stratified,
)
from src.evaluate import evaluate_model
from src.metrics import format_metrics_row
from src.train_resnet import train_resnet
from src.train_vit import train_vit
from src.utils import ensure_dir, load_config, resolve_device, set_seed
from src.visualize import plot_confusion_matrix, plot_training_history, save_gradcam_examples

# === CHANGE THESE FOR EACH RUN ===
FRACTION = 1.0      # 1.0, then 0.5, then 0.25
MODEL = 'resnet'    # 'resnet' or 'vit'
# =================================

cfg = load_config(ROOT / 'configs' / 'default.yaml')
set_seed(cfg['seed'])
device = resolve_device(cfg.get('device', 'auto'))
print('Device:', device, '| MODEL:', MODEL, '| FRACTION:', FRACTION)

data_root = resolve_data_root(cfg)
train_pool, test_samples, layout = load_all_samples(data_root, layout=cfg['data'].get('layout', 'auto'))
train_samples, val_samples = split_train_val(
    train_pool,
    val_fraction=cfg['data']['val_fraction'],
    seed=cfg['seed'],
    group_augmentations=cfg['data'].get('group_augmentations', True),
)
train_subset = subset_train_stratified(
    train_samples, fraction=FRACTION, seed=cfg['seed'],
    group_augmentations=cfg['data'].get('group_augmentations', True),
)
train_loader, val_loader, test_loader = make_loaders(
    train_subset, val_samples, test_samples,
    image_size=cfg['data']['image_size'],
    batch_size=cfg['data']['batch_size'],
)
print(f'n_train={len(train_subset)} layout={layout}')
```

---

## Cell 4 (Code) — Train + evaluate

```python
run_name = f"{MODEL}_frac{FRACTION}_seed{cfg['seed']}"
ckpt_path = ensure_dir(ROOT / cfg['paths']['checkpoints'] / run_name) / 'best.pt'

if MODEL == 'resnet':
    model, train_meta, history = train_resnet(train_loader, val_loader, device, cfg, ckpt_path)
    is_vit = False
else:
    model, train_meta, history = train_vit(train_loader, val_loader, device, cfg, ckpt_path)
    is_vit = True

eval_out = evaluate_model(model, test_loader, device, is_vit=is_vit)
row = format_metrics_row(
    model=MODEL,
    train_fraction=FRACTION,
    metrics=eval_out['metrics'],
    train_time_s=train_meta['train_time_s'],
    infer_latency_ms=eval_out['infer_latency_ms'],
    model_size_mb=eval_out['model_size_mb'],
    seed=cfg['seed'],
)
display(pd.DataFrame([row]))

csv_path = ROOT / cfg['paths']['comparison_csv']
csv_path.parent.mkdir(parents=True, exist_ok=True)
row_df = pd.DataFrame([row])
if csv_path.exists():
    row_df.to_csv(csv_path, mode='a', header=False, index=False)
else:
    row_df.to_csv(csv_path, index=False)
print('Saved to', csv_path)
```

---

## Cell 5 (Code) — Plots (optional)

```python
fig_dir = ensure_dir(ROOT / cfg['paths']['figures'] / run_name)
plot_training_history(history, fig_dir / 'training_curves.png')
plot_confusion_matrix(eval_out['metrics']['confusion_matrix'], fig_dir / 'confusion_matrix.png', title=run_name)
if MODEL == 'resnet':
    save_gradcam_examples(model, test_loader, device, fig_dir, max_images=4, is_vit=False)
print('Figures saved under', fig_dir)
```

**Six runs:** edit Cell 3 → Run Cell 3 + 4 (+ 5 optional) each time.
