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
- [x] Full-scale validated pipeline (40M rows, time-based split, tuned models) — `notebooks/full_pipeline.ipynb`

## Results — quick demo (`eda.ipynb`, first 200k rows, random split, 7 low-card features)
| Model | AUC | Log-loss |
|---|---|---|
| Baseline (constant) | 0.500 | 0.4633 |
| Logistic Regression | 0.6156 | 0.4499 |
| LightGBM | 0.6377 | 0.4448 |

Both AUC and log-loss improve monotonically baseline → LogReg → LightGBM — clean story for slides.
No leakage (AUC well below the ~0.99 red flag). Kept fast (~30s Run All) for live demoing.

Note: LogReg deliberately does **not** use `class_weight='balanced'` — it improved AUC but wrecked
log-loss (0.67, worse than baseline) by distorting predicted probabilities. Left unbalanced for a
clean monotonic story; calibration is listed as a "next step" if we want to revisit it as a teaching
point.

## Results — full-scale validated pipeline (`full_pipeline.ipynb`, all 40M rows)
Time-based split: train = days 21-28, val (early stopping/tuning) = day 29, **test = day 30,
touched exactly once.** Adds leak-safe frequency encoding for `site_id`, `app_id`, `device_model`,
`device_id`, `device_ip` (frequency maps fit on train only). All metrics below are on day 30.

| Model | Train AUC | Test AUC | Test log-loss | Train-Test AUC gap |
|---|---|---|---|---|
| Baseline | — | 0.500 | 0.4633 | — |
| Logistic Regression (C=0.1) | 0.6269 | 0.6509 | 0.4371 | -0.024 (stable, no overfit — test even beats train) |
| LightGBM (tuned + early-stopped) | 0.7545 | 0.7278 | 0.4085 | +0.027 (small, healthy) |

LightGBM best config: `num_leaves=63, learning_rate=0.1, min_child_samples=100, reg_alpha=1.0,
reg_lambda=1.0, subsample=0.7, colsample_bytree=0.7`, early-stopped at 860 trees (not guessed).
Chosen via a 5-config search on a 3M-row subsample, confirmed at full scale.

Takeaway for slides: LogReg is very stable but capacity-limited (linear, plateaus regardless of
`C`); LightGBM has a much higher ceiling and, thanks to tuned regularization + early stopping,
still generalizes well (small train-test gap) despite far more capacity.

Runtime: `full_pipeline.ipynb` takes ~20-25 min for Run All (LightGBM search+final ~16 min,
LogReg ~8 min) — not meant for a live demo, use `eda.ipynb` for that.

## Next up
1. Try target encoding (with cross-fitting to avoid leakage) — may beat frequency encoding
2. If submitting to Kaggle leaderboard: refit frequency maps on all of `train.csv` (no future
   data to hold out at submission time), run the same pipeline on Kaggle's `test.gz`, submit
3. Pull the full-pipeline results/plots into slides

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
