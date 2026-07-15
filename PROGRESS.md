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
- [x] Commit & push scaffold
- [x] Create venv + install requirements
- [x] Download Avazu `train.csv` into `data/`
- [x] Run `notebooks/eda.ipynb` end to end

## Results (first 200k rows, 7 low-cardinality categorical features)
| Model | AUC | Log-loss |
|---|---|---|
| Baseline (constant) | 0.500 | 0.4633 |
| Logistic Regression | 0.6156 | 0.4499 |
| LightGBM | 0.6377 | 0.4448 |

Both AUC and log-loss improve monotonically baseline → LogReg → LightGBM — clean story for slides.
No leakage (AUC well below the ~0.99 red flag).

Note: LogReg deliberately does **not** use `class_weight='balanced'` — it improved AUC but wrecked
log-loss (0.67, worse than baseline) by distorting predicted probabilities. Left unbalanced for a
clean monotonic story; calibration is listed as a "next step" if we want to revisit it as a teaching
point.

## Next up
1. Add high-cardinality features (`site_id`, `device_model`) via frequency/target encoding
2. Tune LightGBM hyperparameters
3. Split train/test **by time** instead of randomly (more realistic for CTR)
4. Feature importance chart + ROC curve → slides

## Plan (the presentation arc)
EDA (imbalance) → feature prep → baseline → Logistic Regression → LightGBM → evaluation (AUC,
log-loss, ROC curve, feature importance) → slides.
Story: baseline → logistic → boosting with AUC climbing.

## Gotchas to remember
- Never commit `data/` or large CSVs (already in `.gitignore`).
- Notebooks cause git merge conflicts — each teammate works in their own notebook file.
- Don't report training-set scores; always use the held-out test set. AUC ~0.99 = likely leakage.
- macOS: LightGBM needs `brew install libomp` (OpenMP) or it fails on import with a
  `dlopen`/`libomp.dylib not found` error.
- VSCode kernel picker can lag behind `jupyter kernelspec list` — if a registered kernel doesn't
  show up, reload the window or use "Select Another Kernel..." → "Jupyter Kernel...".
