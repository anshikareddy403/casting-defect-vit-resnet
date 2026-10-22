# Streamlit — step-by-step (open & connect)

Two options:

- **Path A — On your laptop** (fastest to try today)  
- **Path B — Streamlit Cloud** (public link for report / professor)

You need **at least one** trained file: `best.pt` from Colab notebook 02.

---

## Path A — Open on your laptop

### Step A1 — Get the project folder on your PC

Open in File Explorer:

`C:\Users\anshi\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-d09787e4-d620-44a2-aaf8-ac08d395a963\files\docs\casting-defect-inspection`

Confirm you see **`streamlit_app.py`** and folder **`src`**.

### Step A2 — Install Python packages

1. Open **PowerShell** (Windows key → type PowerShell).
2. Go to the project:

```powershell
cd "C:\Users\anshi\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-d09787e4-d620-44a2-aaf8-ac08d395a963\files\docs\casting-defect-inspection"
```

3. Install (first time only; can take several minutes):

```powershell
pip install -r requirements.txt
```

If `pip` is not found, try `py -m pip install -r requirements.txt`.

### Step A3 — Start Streamlit

Same folder, run:

```powershell
streamlit run streamlit_app.py
```

Or:

```powershell
py -m streamlit run streamlit_app.py
```

Your browser should open **http://localhost:8501**.  
If not, copy that address into Chrome/Edge.

### Step A4 — Load a model in the app

1. Left sidebar → **Live demo** (use sidebar **Navigation** if needed).
2. Under **Or upload best.pt** → choose your checkpoint from Colab, e.g.  
   `best.pt` from folder `resnet_frac1.0_seed42`.
3. Upload a test image from  
   `data\raw\casting_data\test\ok_front` or `def_front`.

You should see **Prediction** and probabilities.

### Step A5 — Stop the app

In PowerShell press **Ctrl+C**.

---

## Path B — Connect to Streamlit Cloud (public URL)

### Step B1 — Put new files on GitHub

On **https://github.com/anshikareddy403/casting-defect-vit-resnet**:

**Add file → Upload files** (or push with git if you use it):

| Upload |
|--------|
| `streamlit_app.py` |
| `src/` (whole folder, includes `inference.py`) |
| `configs/` |
| `requirements.txt` |
| `.streamlit/` folder (`config.toml`) |

Do **not** upload `data/` or large `best.pt` files.

Optional: copy your metrics CSV to  
`results/comparison_table_demo.csv` and upload **`results/`** so the **Results** tab works.

### Step B2 — Create a Streamlit Cloud account

1. Go to **https://share.streamlit.io**
2. Click **Sign up** / **Continue with GitHub**
3. Authorize Streamlit to read your GitHub

### Step B3 — Create the app

1. Click **Create app**
2. **Repository:** `anshikareddy403/casting-defect-vit-resnet`
3. **Branch:** `main`
4. **Main file path:** `streamlit_app.py`
5. Click **Deploy**

Wait 2–5 minutes. First deploy may fail if `requirements.txt` is missing — fix on GitHub and **Reboot app**.

Your URL looks like:  
`https://casting-defect-vit-resnet-xxxxx.streamlit.app`

### Step B4 — Model weights (choose one)

**Option 1 — Upload during demo (easiest, no Hugging Face)**

1. Open your Streamlit URL
2. Sidebar → **Or upload best.pt**
3. Select file from your PC (downloaded from Colab)

**Option 2 — Hugging Face (best for report; no upload each time)**

1. **https://huggingface.co** → sign up → **New model**
2. Name e.g. `casting-resnet-demo` → upload `best.pt`
3. On the file page → **Copy download link** (URL contains `/resolve/main/best.pt`)
4. Streamlit Cloud → your app → **Settings** (⚙️) → **Secrets**
5. Paste:

```toml
[huggingface]
resnet_checkpoint = "https://huggingface.co/YOUR_USERNAME/casting-resnet-demo/resolve/main/best.pt"
vit_checkpoint = "https://huggingface.co/YOUR_USERNAME/casting-vit-demo/resolve/main/best.pt"
```

6. **Save** → **Reboot app**
7. In the app sidebar → **Preset** → **ResNet (secrets)** or **ViT (secrets)**

### Step B5 — Test the three tabs

1. **About** — project text loads  
2. **Results** — table/charts (if you uploaded demo CSV)  
3. **Live demo** — image + prediction  

Put the Streamlit URL in your **report** and **PPT**.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit` not recognized | `pip install streamlit` or use `py -m streamlit run streamlit_app.py` |
| `No module named 'src'` | Run from project root (folder with `streamlit_app.py`) |
| Cloud app crashes on start | Check **Manage app → Logs**; ensure `requirements.txt` on GitHub |
| No checkpoint | Train notebook 02 in Colab or upload `best.pt` in sidebar |
| App slow | Normal on free tier (CPU); one image at a time is fine |

---

## What to tell your professor

“We deploy a **Streamlit** front-end on **Streamlit Cloud**; models are fine-tuned with **Hugging Face ViT** and **PyTorch ResNet** on **Colab**; code and metrics are on **GitHub**.”
