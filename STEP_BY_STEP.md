# Step-by-step guide (start here)

Repo: **https://github.com/anshikareddy403/casting-defect-vit-resnet**  
Deadline: **27 October 2026**

Do **one part at a time**. Only go to the next part when the current one works.

---

## Part 1 — Open your project folder on your computer

1. Open **File Explorer** (Windows).
2. Paste this into the address bar and press Enter:

   `C:\Users\anshi\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-d09787e4-d620-44a2-aaf8-ac08d395a963\files\docs\casting-defect-inspection`

3. You should see folders named **`src`**, **`notebooks`**, **`report`**, and a file **`README.md`**.

If that path does not exist, in Cursor open **Context** → **casting-defect-inspection** and note where those files live on your PC.

---

## Part 2 — Put the Kaggle images in the right place

1. Inside **casting-defect-inspection**, create folders if they are missing:

   - `data`
   - `data\raw`

2. Find your **extracted Kaggle** folders. Many downloads give **`casting_data`** and sometimes **`casting 512x512`**.

3. **Use `casting_data` (recommended for this download):**
   - Open **`casting_data`** → you should see **`train`** and **`test`**, each with **`ok_front`** and **`def_front`**.
   - Copy the whole **`casting_data`** folder into `data\raw` so you have:

   ```
   casting-defect-inspection\data\raw\casting_data\train\ok_front\
   casting-defect-inspection\data\raw\casting_data\test\def_front\
   ```

4. **Ignore `casting 512x512` for now** (different resolution; not needed if `casting_data` works).

   **Alternative:** If you find **`ok_front`** and **`def_front`** directly (no train/test), copy those two folders into `data\raw` only.

4. Quick check: open a few images — casting photos, not random files.

---

## Part 3 — Put your **code** on GitHub (no Git required)

GitHub must **not** receive the huge `data` folder. We upload everything else.

1. Open: **https://github.com/anshikareddy403/casting-defect-vit-resnet**
2. Click **Add file** → **Upload files**.
3. From **casting-defect-inspection**, select and drag these (hold Ctrl to multi-select):

   - `configs` (folder)
   - `notebooks` (folder)
   - `report` (folder)
   - `scripts` (folder)
   - `src` (folder)
   - `results` (folder — may only have `.gitkeep` and a CSV template)
   - Files: `README.md`, `requirements.txt`, `.gitignore`, `RUNBOOK.md`, `STEP_BY_STEP.md`

   **Do not upload** the `data` folder.

4. Scroll down → **Commit message**: `Initial project upload` → green **Commit changes**.
5. Refresh the page — you should see `src`, `notebooks`, etc.

**Optional later:** Install Git and use the command line; not required for this upload.

---

## Part 4 — Run notebook 01 on Google Colab

1. Go to **https://colab.research.google.com** (Google account login).
2. **File → Upload notebook** → choose  
   `casting-defect-inspection\notebooks\01_data_exploration.ipynb` from your PC.
3. **Runtime → Change runtime type → T4 GPU** (GPU optional for notebook 01; OK to use CPU).
4. Add a **new cell at the top** and run:

   ```python
   !pip install -q pillow pandas matplotlib scikit-learn pyyaml torch torchvision transformers
   ```

5. Upload your **data** to Colab (dataset is too big for GitHub):

   - Zip the folder `data\raw` on your PC (right-click → Send to → Compressed folder).
   - In Colab left sidebar: **Files** (folder icon) → **Upload** → select the zip.
   - New cell:

   ```python
   !unzip -q your_zip_name.zip -d /content/data
   # If unzip creates data/raw/ok_front, you are good.
   # If paths differ, tell your Project agent the folder names you see under /content.
   ```

6. Fix the notebook path to your project files — new cell before the rest:

   ```python
   import sys
   from pathlib import Path
   # After Part 3, clone from GitHub:
   !git clone https://github.com/anshikareddy403/casting-defect-vit-resnet.git
   ROOT = Path('/content/casting-defect-vit-resnet')
   sys.path.insert(0, str(ROOT))
   %cd /content/casting-defect-vit-resnet
   ```

   Edit `configs/default.yaml` in Colab if data is not at `data/raw`:

   ```yaml
   data:
     root: /content/data/raw
   ```

   (Use **Files** → open `configs/default.yaml` → edit → save.)

7. **Runtime → Run all**.  
   **Success looks like:** printed train/val/test counts and a small table with **acceptable** and **defective** in each split.

If you get an error, copy the **red error text** and send it in the Project chat.

---

## Part 5 — Train models (after Part 4 works)

1. Upload **`02_train_and_compare.ipynb`** to Colab (same way as 01).
2. **Runtime → GPU** (required for ViT).
3. Same setup cells as Part 4 (pip, clone repo, data path).
4. In the notebook, find:

   ```python
   FRACTION = 0.25
   MODEL = 'resnet'
   ```

   **First run only:** set `FRACTION = 1.0` and `MODEL = 'resnet'`, then **Run all**.

5. Repeat with different `FRACTION` (0.25, 0.5, 1.0) and `MODEL` (`resnet`, `vit`) — **6 runs total**.

6. Open `results/comparison_table.csv` (download from Colab Files) — paste numbers into `report/TECHNICAL_REPORT.md`.

---

## Part 6 — Submit

1. Report: fill `report/TECHNICAL_REPORT.md` → export PDF if required.
2. PPT: use `report/PRESENTATION_OUTLINE.md`.
3. Put link **https://github.com/anshikareddy403/casting-defect-vit-resnet** in the report.
4. Add screenshots from `results/figures/` if you generated them.

---

**Where you are now:** finish **Part 1 → 2 → 3** today. **Part 4** tomorrow if tired.
