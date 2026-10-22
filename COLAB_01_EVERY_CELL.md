# Notebook 01 — copy each block into its own Colab cell (in order)

**Before you start:** Runtime → **Restart session**  
**GPU:** optional for this notebook (CPU is fine)

---

## Cell 1 (Markdown)

```markdown
# 01 — Casting dataset exploration (Colab)

Run cells **top to bottom**. Do not skip Cell 2 or 3.
```

---

## Cell 2 (Code) — Setup + clone GitHub

```python
%cd /content

!pip install -q pillow pandas matplotlib scikit-learn pyyaml torch torchvision transformers

!rm -rf /content/casting-defect-vit-resnet
!git clone https://github.com/anshikareddy403/casting-defect-vit-resnet.git

%cd /content/casting-defect-vit-resnet
!ls src/

import os
import sys
from pathlib import Path

ROOT = Path('/content/casting-defect-vit-resnet')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

print('Setup OK — data.py should appear in the ls output above')
```

---

## Cell 3 (Code) — Upload casting_data.zip from your PC

```python
from google.colab import files
from pathlib import Path

raw = ROOT / 'data' / 'raw'
raw.mkdir(parents=True, exist_ok=True)

print('Click Choose Files and select casting_data.zip from your computer')
uploaded = files.upload()

for name, data in uploaded.items():
    dest = raw / name
    dest.write_bytes(data)
    print('Saved', dest, 'bytes', dest.stat().st_size)

zip_path = raw / 'casting_data.zip'
assert zip_path.exists(), 'Upload casting_data.zip first'

!unzip -q -o /content/casting-defect-vit-resnet/data/raw/casting_data.zip -d /content/casting-defect-vit-resnet/data/raw
!ls /content/casting-defect-vit-resnet/data/raw/casting_data/train/
```

---

## Cell 4 (Code) — Imports + load splits

```python
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.data import detect_layout, load_all_samples, resolve_data_root, split_train_val
from src.utils import load_config

cfg = load_config(ROOT / 'configs' / 'default.yaml')
data_root = resolve_data_root(cfg)
print('Data root:', data_root)
print('Detected layout:', detect_layout(data_root))

train_pool, test_samples, layout = load_all_samples(
    data_root, layout=cfg['data'].get('layout', 'auto')
)
train_samples, val_samples = split_train_val(
    train_pool,
    val_fraction=cfg['data']['val_fraction'],
    seed=cfg['seed'],
    group_augmentations=cfg['data'].get('group_augmentations', True),
)
print(f'layout={layout} | train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}')
```

---

## Cell 5 (Code) — Class balance table

```python
def to_df(samples, split_name):
    return pd.DataFrame({
        'split': split_name,
        'label': ['defective' if s.label else 'acceptable' for s in samples],
        'group_id': [s.group_id for s in samples],
        'path': [str(s.path) for s in samples],
    })

df = pd.concat([
    to_df(train_samples, 'train'),
    to_df(val_samples, 'val'),
    to_df(test_samples, 'test'),
])
display(df.groupby(['split', 'label']).size().unstack(fill_value=0))
```

---

## Cell 6 (Code) — Sample images

```python
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, sample in zip(axes.flatten(), test_samples[:8]):
    ax.imshow(Image.open(sample.path).convert('L'), cmap='gray')
    ax.set_title('defective' if sample.label else 'acceptable', fontsize=9)
    ax.axis('off')
plt.suptitle('Sample test images')
plt.tight_layout()
plt.show()
```

---

## Cell 7 (Markdown)

```markdown
## Augmentation grouping check
Groups like `cast_ok_3_1` and `cast_ok_3_2` must not appear in both train and val.
```

---

## Cell 8 (Code) — Leakage check

```python
train_groups = set(s.group_id for s in train_samples)
val_groups = set(s.group_id for s in val_samples)
overlap = train_groups & val_groups
print('Overlapping groups between train and val:', len(overlap))
assert len(overlap) == 0, 'Leakage: same augmentation group in train and val'
print('Part 4 complete — you can open notebook 02 for training.')
```
