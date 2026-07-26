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

Runtime: `full_pipeline.ipynb` takes ~35-40 min for Run All (LightGBM search+final ~16 min,
LogReg ~8 min, target encoding comparison in section 7 ~13 min) — not meant for a live demo,
use `eda.ipynb` for that.

## Target encoding — tested, not adopted
Tried cross-fitted (5-fold) target encoding as an alternative to frequency encoding for the
same high-cardinality columns, same time-based split, same LightGBM hyperparams.

| Encoding | Train AUC | Test AUC | Test log-loss | Train-Test AUC gap |
|---|---|---|---|---|
| Frequency (adopted) | 0.7545 | 0.7278 | 0.4085 | 0.027 |
| Target (cross-fitted) | 0.7765 | 0.7299 | 0.4064 | 0.047 |

Target encoding wins on both test metrics, but only marginally (+0.002 AUC, +0.002 log-loss),
while the train-test gap nearly doubles — and it costs ~9 extra minutes per run (5-fold
cross-fitting per column) vs. frequency encoding's near-instant `value_counts()`. Given the
project's stability goal, **frequency encoding stays the primary model**; target encoding is
documented in `full_pipeline.ipynb` section 7 as a tested-and-rejected alternative — a good,
honest trade-off story for the slides.

Gotcha hit along the way: early-stopping on `eval_metric='auc'` alone stopped training after
only ~6-100 rounds, because AUC (ranking) saturates almost immediately with target-encoded
features while log-loss (calibration) keeps improving for longer — switching early stopping
to `eval_metric='binary_logloss'` fixed it. Worth remembering any time a target-encoded/strong
feature makes a model converge suspiciously fast.

## Calibration check — done (not just a "next step" anymore)
AUC/log-loss only check ranking; calibration checks whether "70% predicted" really means a
~70% real-world click rate — which is exactly what the original Kaggle competition graded
submissions on (log-loss, not AUC), and what would matter if these probabilities were ever
used for pricing (expected value ≈ predicted CTR × value-per-click) rather than just ranking
impressions for placement.

Checked via a reliability diagram (10 quantile bins, predicted vs. actual click rate) on the
untouched day-30 test set, using the adopted LightGBM + frequency-encoding model:

- AUC 0.7278, log-loss 0.4085, **Brier score 0.1271**
- The model **over-predicts at every probability level** — worst in the 10-20% predicted
  range (predicts ~15-19%, actual is only ~9-16%, a 3-4 point overestimate)
- Ranking (AUC) is unaffected, since calibration errors don't change relative order

Practical takeaway: fine to use as-is for **placement/targeting** (ranking-based decisions);
would need recalibration (Platt scaling or isotonic regression) before using the raw
probability for **pricing**. Documented in `full_pipeline.ipynb` section 8. (The dedicated
"Diligence Checks" slide covering this + target encoding was removed from the deck for length;
the finding still lives in the notebook and here.)

## Slides
Final version — `slides/presentation.html` (10 slides): Title, Data & Task
(Kaggle context), Problem, The Trap (imbalance + honest hour-of-day chart), Reading the Chart
Carefully (site-category-mix scatter plot), Approach, Design Decisions, Results (ROC +
feature importance), Trustworthy & Trade-offs, Thank You. 

## Gotchas to remember
- Never commit `data/` or large CSVs (already in `.gitignore`).
- Notebooks cause git merge conflicts — each teammate works in their own notebook file.
- Don't report training-set scores; always use the held-out test set. AUC ~0.99 = likely leakage.
- macOS: LightGBM needs `brew install libomp` (OpenMP) or it fails on import with a
  `dlopen`/`libomp.dylib not found` error.
- VSCode kernel picker can lag behind `jupyter kernelspec list` — if a registered kernel doesn't
  show up, reload the window or use "Select Another Kernel..." → "Jupyter Kernel...".
