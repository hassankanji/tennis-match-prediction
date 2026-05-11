# "What Actually Worked" Memo — Week 5
## STAT 390 Capstone | Tennis Match Prediction
**Date:** 2026-05-11

---

## Summary

Week 5 ran 12 agent experiments across 3 independent iterations plus an exhaustive
n_estimators search (8 runs). No experiment improved on the week 4 best
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
However, the gap (0.11) remains large — the depth constraint is a band-aid, not a fix.

---

## What We Discovered This Week (even without improvement)

### Discovery 1: n_estimators is already optimal at n=100
Exhaustive search over [50, 100, 150, 200, 250, 300, 400, 500] showed n=100 is optimal.
More trees plateau or worsen val_brier. Train_brier stays flat (~0.054) regardless —
the RF overfits the same way with more or fewer trees.

### Discovery 2: Feature selection hurts — all 34 features carry signal
Top-15 gives val_brier=0.1678 (+0.003 worse). Top-10 gives 0.1716 (+0.007 worse).
Even low-importance features contribute to ensemble diversity.

### Discovery 3: Overfitting is the dominant problem
Train_brier=0.054, val_brier=0.165 → gap=0.11. No model tested reduced both
overfit gap AND val_brier simultaneously. Bias-variance tradeoff is unavoidable
on this small val set (n=236).

### Discovery 4: Surface stratification increases variance
3 surface models × ~700-900 training matches each < 1 pooled model × 2944 matches.
Surface is better as a feature (surface_code) than as a split criterion here.

---

## What Consistently Failed

| Category | Example | Why it failed |
|----------|---------|---------------|
| Feature pruning | RF top-15 | Removes signal — all features contribute |
| More trees | RF n=150-500 | No benefit; val floor already hit |
| Slow GBM | n=500 lr=0.01 | Lower overfit but worse val (high bias) |
| Heavy regularization | LR C=0.01 | Near-zero overfit but LR biased (0.168) |
| Surface split | 3 surface models | Variance from smaller training sets |
| Calibration | CalibratedCV | Probabilities shift but val_brier does not improve |

---

## Week 5 Checkpoint Answers

**Block length:** 3 iterations × 4 runs each = 12 agent runs + 8 n_estimators runs = 20 total

**Best result vs baseline:** val_brier 0.2055 → 0.1648 (−19.8%). No improvement this week.

**Keep/Discard/Crash rates:** 0 keep / 12 discard / 0 crash

**Most helpful modification type:** Adding first-set features (Week 3).

**Biggest uncertainty:** Whether val_brier=0.1648 is the true noise floor on the
236-match val set. Differences < 0.003 Brier points cannot be reliably detected.

---

## Path Forward (Week 6)

1. 5-fold cross-validation on train set — get reliable estimates without the small val set
2. XGBoost — native L1/L2 regularization may reduce overfit more effectively
3. SHAP analysis — understand which first-set features matter by surface/round
4. Consider whether the test set can now be partially unlocked for calibration assessment
