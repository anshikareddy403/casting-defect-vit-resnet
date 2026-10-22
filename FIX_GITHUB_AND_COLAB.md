# Fix: “src not found” on Colab

## Why this happens

Colab runs `git clone` from **GitHub**. Your repo must contain a folder named **`src`**.  
If GitHub only has `README.md` and a few files (no `src`), Colab cannot import `from src.data import ...`.

**Fix order:** complete **Part A (GitHub)** first, then **Part B (Colab)**.

---

## Part A — Add missing folders on GitHub (~10 min)

1. Open **https://github.com/anshikareddy403/casting-defect-vit-resnet**
2. Look at the file list. You **must** see a folder **`src`**. If you only see `.md` files, continue below.
3. Click **Add file** → **Upload files**
4. On your PC, open:

   `C:\Users\anshi\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-d09787e4-d620-44a2-aaf8-ac08d395a963\files\docs\casting-defect-inspection`

5. Drag these **folders** into GitHub (one upload is OK):

   - **`src`**
   - **`configs`**
   - **`notebooks`**
   - **`scripts`**
   - **`report`**
   - **`results`**

6. **Commit changes**
7. Refresh GitHub — click **`src`** — you should see **`data.py`**, **`train_vit.py`**, etc.

---

## Part B — Colab again (fresh clone)

### B1 — Setup cell (run first)

```python
!pip install -q pillow pandas matplotlib scikit-learn pyyaml torch torchvision transformers

!rm -rf /content/casting-defect-vit-resnet
!git clone https://github.com/anshikareddy403/casting-defect-vit-resnet.git
%cd /content/casting-defect-vit-resnet

!ls src/
print('If you see data.py above, GitHub is fixed.')
```

### B2 — Import cell (run second)

```python
import os
import sys
from pathlib import Path

ROOT = Path('/content/casting-defect-vit-resnet')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from src.data import detect_layout, load_all_samples, resolve_data_root, split_train_val
from src.utils import load_config

print('ROOT OK')
```

### B3 — Data upload (run third)

```python
from google.colab import files
from pathlib import Path

raw = Path('/content/casting-defect-vit-resnet/data/raw')
raw.mkdir(parents=True, exist_ok=True)
print('Choose casting_data.zip from your PC')
files.upload()

!unzip -q -o /content/casting-defect-vit-resnet/data/raw/casting_data.zip -d /content/casting-defect-vit-resnet/data/raw
!ls /content/casting-defect-vit-resnet/data/raw/casting_data/train/
```

### B4 — Run rest of notebook 01

Run remaining cells or **Runtime → Run all**.

---

## Checklist

- [ ] GitHub shows **`src/data.py`**
- [ ] B1 prints **`data.py`** in `ls src/`
- [ ] B2 prints **`ROOT OK`**
- [ ] B3 shows **`ok_front`** and **`def_front`**
- [ ] Notebook 01 shows train/val/test counts
