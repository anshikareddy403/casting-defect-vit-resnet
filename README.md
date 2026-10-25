# Casting defect inspection — ViT-B/16 vs ResNet-18

MSc project (MAAI901): data-efficient binary classification on the [Kaggle casting product image dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (CC BY-SA 4.0).

## Deliverables

- `report/Casting_Defect_Inspection_Report.docx` — written report  
- `report/Casting_Defect_Inspection_Presentation.pptx` — defence slides  

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place data under `data/raw/` (not committed). See `configs/default.yaml` for layout (`kaggle_split` or original folders). Upload `casting_data.zip` on Colab or unzip locally under `data/raw/`.

## Training

**Notebooks (Colab, GPU):** `notebooks/01_data_exploration.ipynb`, `notebooks/02_train_and_compare.ipynb`

**CLI:**

```bash
python scripts/run_experiment.py --model resnet --fraction 0.25
python scripts/run_experiment.py --model vit --fraction 1.0
```

Metrics append to `results/comparison_table.csv`.

## Streamlit demo

```bash
streamlit run streamlit_app.py
```

Point the app at a checkpoint under `results/checkpoints/` (train first) or a local `.pt` file.

## Layout

```
configs/default.yaml
src/                   # data, train, evaluate, metrics, visualize
scripts/run_experiment.py
notebooks/
streamlit_app.py
results/comparison_table.csv
report/                # report + presentation only
```


