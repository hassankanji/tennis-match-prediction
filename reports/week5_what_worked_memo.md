# "What Actually Worked" Memo — Week 5
## STAT 390 Capstone | Tennis Match Prediction
**Date:** 2026-05-11

---

## Summary

Week 5 ran 12 agent experiments across 3 independent iterations plus an exhaustive
n_estimators search (8 runs). **No experiment improved on the week 4 best**
(RF n=100, max_depth=10, all features, val_brier=0.1648). However, the week 5 block
produced four concrete, interpretable discoveries that explain WHY no improvement
was found and what the path forward must be.

---

## What Actually Worked (across all weeks)

### 1. Adding first-set features (Week 3) — the largest single gain
**Brier improvement: +0.036 (−17.6% from baseline)**

The single most effective modification was including live first-set statistics
alongside pre-match features. The pre-match-only LR model (val_brier=0.2055)
improved to 0.1694 when first-set features were added. This is the dominant signal
in the entire project to date.

*Why it worked:* First-set outcomes directly reveal within-match form. s1_A_won,
s1_margin, and s1_A_return_pts_won_pct all carry strong signal (top 4 in permutation
importance). These are not available pre-match, so they represent genuine information
gain that a pre-match model cannot capture.

### 2. RF with max_depth=10 constraint (Week 4) — modest but real gain
**Brier improvement: +0.003 (−1.8% from RF baseline)**

Constraining RF depth from None to 10 improved val_brier from 0.1681 to 0.1648.
This is a small but reproducible gain (deterministic at random_state=42).

*Why it worked:* The RF with unlimited depth memorizes training data too aggressively.
Depth=10 provides partial regularization, reducing the overfit gap slightly.
However, the gap (0.11) remains large — the depth constraint is a band-aid,
not a solution.

---

## What We Discovered This Week (even without improvement)

### Discovery 1: n_estimators is already optimal at n=100
Exhaustive search over [50, 100, 150, 200, 250, 300, 400, 500] showed that
n=100 is already optimal. More trees plateau val_brier or make it slightly worse.
Train_brier stays nearly flat (~0.054) regardless of n — the RF overfits the
same way with more or fewer trees.

### Discovery 2: Feature selection hurts — all 34 features carry signal
Removing the bottom-K features (by permutation importance) consistently hurts
val_brier. Top-15 gives 0.1678 (+0.003 worse). Top-10 gives 0.1716 (+0.007 worse).
Even features with near-zero permutation importance contribute to ensemble
diversity in ways that the permutation test misses. This is a known limitation
of permutation importance in correlated feature sets.

### Discovery 3: Overfitting is the dominant problem
The overfit gap (train_brier=0.054, val_brier=0.165) is 0.11 — enormous.
The model memorizes 2,944 training matches but only generalized to 236 val matches.
No model tested in iteration 2 (slow GBM, heavy L2 LR, ExtraTrees, calibration)
reduced both the overfit gap AND the val_brier simultaneously. The bias-variance
tradeoff is unavoidable: reducing variance increases bias, and on this small val set,
neither side clearly wins.

### Discovery 4: Surface stratification increases variance
Fitting 3 separate surface models (Clay/Grass/Hard) gives each model only ~700-900
training matches. The reduced training set increases variance enough to offset
any surface-specific signal. Conclusion: surface is better included as a feature
(surface_code) than as a split criterion at this data scale.

---

## What Consistently Failed

| Category | Example | Why it failed |
|----------|---------|---------------|
| Feature pruning | RF top-15, RF top-10 | Removes signal — all features contribute |
| More trees | RF n=150-500 | No additional benefit; val floor already hit |
| Slow GBM | n=500 lr=0.01 | Lower overfit but worse val (high bias) |
| Heavy regularization | LR C=0.01 | Near-zero overfit but LR biased (0.168) |
| Surface split | 3 surface models | Variance increases from smaller training sets |
| Calibration | CalibratedCV isotonic | Probabilities shift but val_brier does not improve |

---

## Week 5 Checkpoint Answers (Lightning Round)

**Block length:** 12 agent runs across 3 iterations (4 runs each), plus 8 n_estimators runs

**Best result vs baseline:** RF max_depth=10 → val_brier=0.1648 vs baseline 0.2055 (−19.8%)

**Keep/Discard/Crash rates:** 0 keep / 12 discard / 0 crash (100% discard rate this week)

**Most helpful modification type:** Adding first-set features (weeks 3-4). Nothing in week 5 helped.

**Biggest current uncertainty:** Whether the 0.1648 val_brier reflects the true model quality
or is just the noise floor on the 236-match val set. The val set is too small to
distinguish modifications smaller than ~0.003 Brier points (~1 correct prediction).

---

## Path Forward (Week 6)

1. **5-fold cross-validation on the train set** — bypass the small val set; get reliable
   estimates of generalisation without touching the test set.
2. **XGBoost** — native L1/L2 regularization and subsampling may reduce overfit more
   effectively than RF/GBM from sklearn.
3. **SHAP analysis** — understand which first-set features drive predictions for
   specific match types (surface/round/ranking), connecting back to the research question.
4. **Consider the test set unlock** — the val_brier of 0.1648 may be the project's
   ceiling without architectural changes.
