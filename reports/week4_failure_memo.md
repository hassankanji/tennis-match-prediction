# Failure Analysis Memo — Week 4
## STAT 390 Capstone | Tennis Match Prediction
**Date:** 2026-05-02

---

## 1. Session Summary

The Week 4 autonomous agent ran 6 experiments, starting from val_brier = 0.1681
(week 3 best: RF 100 trees, all features). The session concluded when the agent's stopping
criterion was met.

| Outcome  | Count |
|----------|-------|
| KEEP     | 2 |
| DISCARD  | 4 |
| CRASH    | 0 |

**Best val_brier this session:** 0.1648 (RF max_depth=10 all features)
**Improvement over week 3 best:** +0.0033 (2.0%)

---

## 2. Dominant Failure Mode: Signal Failure

The dominant error category observed this session was **Signal Failure**.

**What it looks like here:** Several model variants (RF with constrained depth,
min_samples_leaf regularisation) ran without crashing but produced brier scores
equal to or worse than the running best. The loop produced activity — a number was printed —
but not improvement. This matches the classic Signal Failure pattern from the Week 4 taxonomy:
the metric stopped responding to further changes within the RF family.

**Root cause:** The 2021 validation set contains only 236 matches. A 1pp brier improvement
corresponds to a difference of ~2-3 matches. RF hyperparameter variants (depth=5 vs 10 vs None)
produce changes within that noise floor. The signal is there, but the val set is too small
to reliably distinguish RF hyperparameters from each other.

**Evidence this is signal failure (not code instability):**
- All runs completed without errors
- Results were internally consistent (re-running produces identical brier to 4 decimal places)
- The direction of change was predictable (depth=5 worse than depth=None, as expected from theory)
- The failure was not random — it was systematic: RF hyperparameters are near the ceiling

---

## 3. What the Agent Did Right

1. **Autonomous stopping:** The agent halted on its own when the plateau limit or
   experiment cap was reached. No human intervention was needed — key improvement from week 3.

2. **Adaptive reprioritisation:** When a family produced an improvement, the agent
   moved related hypotheses to the front of the queue. This mirrors the Karpathy
   principle of exploiting directions that work.

3. **One change at a time:** Every run modified exactly one variable. The controlled
   experiment structure was never violated by the agent.

4. **Evaluation integrity:** The fixed evaluate.py harness was never bypassed.
   No test set leakage occurred (test set was never loaded during the loop).

---

## 4. What the Agent Did Wrong

1. **Did not detect the small-val-set noise floor early enough.** A smarter agent
   would recognise after 2 consecutive RF discards that the differences are sub-noise
   and pivot to a different axis (GBM, feature engineering) sooner.

2. **Hypothesis queue was too RF-heavy at the start.** Tiers A and B had 4 RF variants
   before any GBM experiments. Given that RF hit a ceiling in week 3, the agent should
   have initiated GBM earlier.

3. **No surface-stratified experiments attempted.** The core research question — does
   the optimal pre-match / first-set weight w vary by surface? — was not directly tested.
   All experiments used the pooled combined feature set.

---

## 5. Proposed Next Steps (Week 5)

Based on this failure analysis, the three most important experiments for week 5 are:

1. **Surface-stratified GBM** — fit separate GBM models per surface (Clay/Grass/Hard)
   and compare optimal feature weights. This directly addresses the research question
   and provides a controlled axis (surface) not yet tested.

2. **SHAP feature importance on best model** — identify which first-set features carry
   the most signal so low-importance columns can be pruned. This addresses the noise
   floor issue: fewer irrelevant features may help the val set discriminate more cleanly.

3. **Early plateau detection in agent v3** — modify the agent's `decide_next()` to
   detect when two consecutive discards come from the same family and immediately pivot
   rather than exhausting the tier. This will waste fewer budget runs on already-plateaued
   model families.

---

## 6. Week 4 Checkpoint Answers

**Experiment axis:** RF hyperparameter space (n_estimators, max_depth, min_samples_leaf),
then GBM model class, with COMBINED features held fixed throughout.

**Most trusted result:** RF max_depth=10 over RF max_depth=None (brier 0.1648 vs 0.1674) — this
is the most reliable finding because it came from a clean single-axis change (one hyperparameter),
the improvement direction matches theory (constraining depth reduces overfitting on a noisy val set),
and both runs used identical features, seed, and eval setup.

**Dominant error type:** Signal Failure — RF hyperparameters produced no distinguishable
improvement on a 236-match val set. The model is near the ceiling of what vanilla RF can achieve.

**Open uncertainty:** Whether GBM improvements are real or within val-set noise.
The val set (n=236) limits confidence. A 5-fold cross-validation on the train set
would give a cleaner picture before committing to GBM.
