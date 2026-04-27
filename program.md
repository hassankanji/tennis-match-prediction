# program.md — AutoResearch Specification
## Tennis Match Prediction | STAT 390 Spring 2026

This file is read by the AI agent at the start of every AutoResearch session.
It defines the goal, constraints, experiment loop, and stopping criteria.

---

## 1. Setup (run once per session)

1. Read these files for full context:
   - `program.md` — this file (rules + goal)
   - `src/evaluate.py` — fixed evaluation harness, data splits, metric definition. **Read-only.**
   - `src/run.py` — experiment runner. **Read-only.**
   - `src/model.py` — **the only file you modify.**
2. Verify data exists at `data/clean/tennis_model_ready.csv`.
3. Check `results.tsv` for the current best `val_brier` (lowest = best).
4. Run the baseline: `python src/run.py "baseline"` to establish the starting point.

---

## 2. The Goal

**Minimize `val_brier` on the 2021 validation set.**

The Brier score measures probabilistic calibration — lower is better. It penalizes both wrong predictions and overconfident wrong ones.

Secondary metrics (reported but not used for keep/discard):
- `val_accuracy` — fraction of matches predicted correctly
- `val_auc` — ROC-AUC, discriminative ability

**The research question**: What combination of pre-match features and first-set features best predicts ATP Grand Slam match outcomes? Specifically, is a purely first-set model better than a combined model?

---

## 3. Data Splits — CRITICAL

The dataset is split **temporally** and is fixed inside `evaluate.py`:

| Split | Years | Matches | Purpose |
|-------|-------|---------|---------|
| **train** | 2011–2020 | 2,944 | Model fitting |
| **val** | 2021 | 236 | Agent optimizes against this |
| **test** | 2022–2023 | 458 | **NEVER TOUCH — final evaluation only** |

The test set does **not exist** to the agent. `evaluate.py` exposes `evaluate()` for train/val only. `evaluate_test()` is called once at the end of the project by the human.

---

## 4. What You CAN Do (modify `src/model.py` only)

- Change the `FEATURES` list — add, remove, or reorder feature columns
- Change the model architecture: logistic regression, ridge, random forest, gradient boosting, SVM, etc.
- Change hyperparameters: regularization strength, n_estimators, depth, learning rate, etc.
- Add feature engineering inside `build_model()` using sklearn `Pipeline` and `FunctionTransformer`
- Add interaction terms or polynomial features via `PolynomialFeatures`
- Try ensemble methods combining pre-match and first-set sub-models

**Available feature groups** (defined in `evaluate.py`):
- `ALL_PREMATCH` (8 features): rank_diff, rank_pts_A/B, age_diff, ht_diff, surface_code, hand_A/B_code
- `ALL_FIRSTSET` (25 features): s1_margin, s1_A_won, serve stats, return stats, aces, DFs, winners, UEs, pts_won
- Note: ~16-18% nulls on serve % columns (2011-2013) — evaluate.py handles imputation automatically

## 5. What You CANNOT Do

- Modify `src/evaluate.py` or `src/run.py`
- Change the train/val/test split
- Access the test set (2022-2023) in any way during the loop
- Install new packages beyond `requirements.txt`
- Use features not in `evaluate.ALL_FEATURES` (they won't exist at inference time)
- Hard-code val set labels (the evaluator will catch this)

---

## 6. Experiment Loop

```
LOOP FOREVER:

1. Decide on one change to make to model.py
   — one idea at a time; don't change 5 things at once
2. Edit src/model.py
3. Run: python src/run.py "<short description of change>"
4. Read val_brier from stdout
5. Compare to current best val_brier

   IF val_brier improved (LOWER):
     → KEEP: update best = val_brier, note what worked
     → update results.tsv status from 'pending' to 'keep'

   IF val_brier equal or worse:
     → DISCARD: revert model.py to previous version
     → update results.tsv status from 'pending' to 'discard'

6. Log insight: why did it work / not work?
7. Go to step 1
```

**Never stop to ask the human whether to continue.**
**Never access the test set.**
**Log every experiment — crashes too.**

---

## 7. Output Format

`run.py` prints:
```
---
val_brier:    0.205534
val_accuracy: 0.699153
val_auc:      0.743612
n_val:        236
n_features:   8
commit:       a1b2c3d
description:  baseline pre-match LR only
---
```

Results are appended to `results.tsv` automatically.

---

## 8. Logging (`results.tsv`)

Tab-separated, one row per experiment:
```
commit    val_brier   val_accuracy  val_auc   status    description
```

- `status`: `keep`, `discard`, or `crash`
- Do not sort or edit past rows
- File is gitignored — it tracks the full experiment history including discarded runs

---

## 9. Simplicity Criterion

All else being equal, simpler is better:
- A 0.001 brier improvement that adds 50 lines of hacky code? Probably not worth it.
- A 0.001 brier improvement from removing features? Definitely keep.
- Equal performance but simpler model? Keep the simpler one.

The final model must be explainable to a sports analytics audience.

---

## 10. Stopping Criteria

The human will stop the loop. Do not stop on your own unless:
- You have run ≥ 6 experiments
- val_brier has not improved in the last 4 consecutive experiments
- In that case, note "plateau reached" and summarize what you tried

At end of session, produce a summary:
- Best val_brier achieved and the model that achieved it
- Top 3 features by importance (if tree-based) or coefficient magnitude (if linear)
- Recommendation for what to try next session
