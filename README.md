# M3 ML — S4: Predicting Ad Click-Through Rate (CTR)

Team project for the Model 3 ML presentation. We predict whether a user will **click** an ad
(binary classification) and use it to teach class imbalance and proper evaluation metrics.

## Problem
Each row is one ad impression. Target = `click` (1 = clicked, 0 = not clicked).
Because most impressions are *not* clicked, this is an **imbalanced** classification problem —
so we evaluate with **AUC-ROC** and **log-loss**, not accuracy.

## Project structure
```
.
├── data/          # raw data — NOT committed (see .gitignore)
├── notebooks/     # eda.ipynb (exploration + modeling)
├── src/           # reusable code (features.py, etc.)
├── requirements.txt
└── README.md
```

## Getting started
```bash
# 1. Clone
git clone https://github.com/cheauyuin/m3-ml-s4-predicting-ad-ctr.git
cd m3-ml-s4-predicting-ad-ctr

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install libraries
pip install -r requirements.txt

# 4. Register the kernel for Jupyter/VSCode
python -m ipykernel install --user --name m3-ctr --display-name "Python (m3-ctr)"
```

## Data
Download a sample of the **Avazu CTR** dataset from Kaggle and place `train.csv` in `data/`:
https://www.kaggle.com/competitions/avazu-ctr-prediction/data

> ⚠️ Do **not** commit the data — it stays local. Each teammate downloads it once.

## Roadmap
1. EDA — click rate, feature exploration
2. Feature prep — time features, categorical encoding
3. Baseline → Logistic Regression → LightGBM
4. Evaluation — AUC, log-loss, feature importance
5. Slides

## Team workflow
`git pull` → work → `git add .` → `git commit -m "..."` → `git push`.
To avoid notebook merge conflicts, each person works in their own notebook file.
