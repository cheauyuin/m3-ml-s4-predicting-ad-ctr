# Project progress & handoff

Session handoff note — read this first when resuming in the VSCode extension.

## What this is
Team ML presentation project — **Scenario S4: Predicting Ad Click-Through Rate (CTR)**.
Binary classification (click / no-click). Chosen because it's beginner-safe with clean public
data and a strong teaching story (class imbalance → why accuracy lies → use AUC / log-loss).

Repo: https://github.com/cheauyuin/m3-ml-s4-predicting-ad-ctr

## Done so far
- [x] Picked scenario S4
- [x] Created GitHub repo and cloned to `NTU/M3/m3-ml-s4-predicting-ad-ctr`
- [x] Scaffolded project: `.gitignore`, `README.md`, `requirements.txt`, `notebooks/eda.ipynb`, `src/features.py`
- [ ] **Commit & push scaffold** (not done yet)
- [ ] Create venv + install requirements
- [ ] Download Avazu `train.csv` into `data/`
- [ ] Run `notebooks/eda.ipynb` end to end

## Next commands to run (VSCode terminal, inside the project folder)
```bash
# 1. Commit & push the scaffold
git add .
git commit -m "Scaffold CTR project: structure, notebook, requirements"
git push -u origin main

# 2. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name m3-ctr --display-name "Python (m3-ctr)"
```

## Then
1. Download an Avazu CTR sample and put `train.csv` in `data/`
   https://www.kaggle.com/competitions/avazu-ctr-prediction/data
2. Open `notebooks/eda.ipynb` → Select Kernel = "Python (m3-ctr)" → Run All
3. Report the AUC / log-loss numbers (or any errors) to continue.

## Plan (the presentation arc)
EDA (imbalance) → feature prep → baseline → Logistic Regression → LightGBM → evaluation (AUC,
log-loss, ROC curve, feature importance) → slides.
Story: baseline → logistic → boosting with AUC climbing.

## Gotchas to remember
- Never commit `data/` or large CSVs (already in `.gitignore`).
- Notebooks cause git merge conflicts — each teammate works in their own notebook file.
- Don't report training-set scores; always use the held-out test set. AUC ~0.99 = likely leakage.
