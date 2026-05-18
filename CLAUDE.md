# CLAUDE.md — Project Brief for AI Agent

This file is read automatically at the start of every Claude Code session.
It is the single source of truth for project state, conventions, and next steps.

---

## Project

**Predicting ATP Grand Slam Match Outcomes Using Pre-Match History and Live First-Set Performance**
STAT 390 | Spring 2026 | GitHub: https://github.com/hassankanji/tennis-match-prediction

**Research question:**
> After the first set of a Grand Slam match is complete, what is the optimal weight `w` between pre-match historical features and live first-set features for predicting match outcome?
> Formally: `P(A wins) = f(w · x_firstset + (1−w) · x_prematch)` — find `w*`, and test whether it varies by surface.

---

## Current Status

**Week completed: 5 of 8**

| Week | Deliverable | Status |
|------|-------------|--------|
| 1 | Project charter, research question, AutoResearch diagram, risk list, repo structure | ✅ Done — `notebooks/week1_charter.ipynb` |
| 2 | Data pipeline complete, EDA done, check-in notebook | ✅ Done — `notebooks/week2_checkin.ipynb` |
| 3 | AutoResearch setup, 5 dry-run experiments, reflection | ✅ Done — `notebooks/week3_autoresearch.ipynb` |
| 4 | Autonomous agent v2, 6 experiments, best val_brier=0.1648 | ✅ Done — `notebooks/week4_autoresearch.ipynb` |
| 5 | Agent v3: 3 iterations, feature selection, n_est search, 5 deliverables | ✅ Done — `notebooks/week5_autoresearch.ipynb` |
| 6 | Ablation study, scope lock, prove RF(n=100,depth=10) is champion | ✅ Done — `notebooks/week6_ablation.ipynb` |
| 7 | Error analysis, upset analysis, year-over-year trends | ⬜ |
| 8 | Final writeup, calibration curves, full results notebook | ⬜ |

**When the user comes back each week:** update this table, create the next notebook, commit and push to GitHub.

## AutoResearch Infrastructure (added Week 3)

- `program.md` — agent specification (goal, constraints, loop rules, stopping criteria)
- `src/model.py` — THE ONLY FILE THE AGENT MODIFIES (contains `build_model()` + `FEATURES`)
- `src/run.py` — fixed experiment runner, call with `python src/run.py "<description>"`
- `src/evaluate.py` — fixed eval harness, enforces train/val/test split, DO NOT MODIFY
- `results.tsv` — experiment log (gitignored, appended automatically by run.py)

**Metric:** `val_brier` (Brier score, lower = better). Val set = 2021 only (n=236). Test set = 2022–2023 (n=458), LOCKED until final eval.

**Dry-run results so far:**
| Experiment | val_brier | val_acc | val_auc | Status |
|---|---|---|---|---|
| Baseline pre-match LR | 0.2055 | 69.9% | 0.744 | keep |
| First-set only LR | 0.1838 | 74.6% | 0.786 | keep |
| Combined LR C=1.0 | 0.1694 | 75.4% | 0.827 | keep |
| Combined LR C=0.1 | 0.1690 | 75.4% | 0.828 | keep |
| Random Forest 100 trees | **0.1681** | **75.8%** | **0.829** | keep (best) |

---

## Repository Layout

```
tennis-match-prediction/
├── CLAUDE.md                          ← YOU ARE HERE — update each week
├── README.md                          ← Public-facing overview
├── requirements.txt
├── .gitignore
│
├── src/                               ← Pipeline scripts (numbered, run in order)
│   ├── 01_download_data.py
│   ├── 02_build_firstset_features.py
│   ├── 03_join_and_clean.py
│   └── 04_explore_clean_data.py
│
├── notebooks/                         ← One notebook per week (ipynb preferred)
│   ├── week1_charter.ipynb            ✅ Week 1 assignment
│   ├── week2_checkin.ipynb            ✅ Week 2 assignment
│   └── week3_baseline_models.ipynb    ← create next session
│
├── data/
│   ├── raw/slam_pbp/                  ← 49 tournament files (gitignored)
│   ├── interim/firstset_features.csv  ← 10,377 matches (gitignored)
│   ├── clean/tennis_model_ready.csv   ← 3,638 × 52 model-ready (gitignored)
│   └── plots/                         ← EDA figures (committed)
│
└── reports/
    └── project_proposal.txt
```

**Notebook naming convention:** `weekN_<slug>.ipynb` — e.g. `week3_baseline_models.ipynb`

---

## Data Facts (memorize these)

| Fact | Value |
|------|-------|
| Final dataset | `data/clean/tennis_model_ready.csv` |
| Shape | 3,638 rows × 52 columns |
| Coverage | ATP Grand Slam men's singles, 2011–2023 |
| Target | `A_won` (binary, balanced 50/50 by random A/B assignment) |
| Train set | 2011–2021 matches |
| Test set | 2022–2023 matches (held out — never used during tuning) |
| Pre-match features (11) | `rank_A/B`, `rank_diff`, `rank_pts_A/B`, `age_A/B`, `age_diff`, `ht_A/B`, `ht_diff`, `surface`, `hand_A/B` |
| First-set features (29) | games won, set margin, serve pts, first serve %, first/second serve won %, break pts faced/saved, return pts won %, aces, double faults, winners, unforced errors |
| Dropped feature | `avg_rally_length` — 44% null, r ≈ 0 with target |
| Null rate | ~16–18% on `first_srv_pct` and per-serve-type won % (early years only) |

---

## Key EDA Findings (already established — don't re-derive)

- Set-1 winner wins the match **77.8%** overall (Clay 81.2%, Grass 77.7%, Hard 76.7%)
- Higher-ranked player wins **72.9%** — strong pre-match baseline
- Strongest first-set predictors: `s1_A_won` (r=0.556), `s1_margin` (0.548), `s1_return_pts_won_pct` (0.436)
- `s1_first_srv_pct` is nearly useless (r=0.063) — getting the serve in ≠ winning points
- Surface should be included as an interaction term (win rates differ ~4pp across surfaces)

---

## Modeling Roadmap (detailed)

### Stage 1 — Pre-match logistic regression (Week 3)
- Features: `rank_diff`, `rank_pts_A/B`, `age_diff`, `ht_diff`, `surface` (encoded), `hand_A/B`
- Expected accuracy: ~66–68%
- Purpose: historical-only benchmark

### Stage 2 — First-set logistic regression (Week 3)
- Features: `s1_margin`, `s1_A_return_pts_won_pct`, `s1_A_first_srv_won_pct`, `s1_A_aces`, `s1_A_df`, `s1_A_win`, `s1_A_ue`, `s1_A_bp_faced`, `s1_A_bp_saved_pct`
- Impute null serve stats with per-surface-per-year median
- Expected accuracy: ~78–80%

### Stage 3 — Weighted combination (Week 4–5)
- Approach A: grid search `w` from 0→1 in steps of 0.05, 5-fold CV
- Approach B: ridge logistic regression on all features combined
- Approach C: repeat grid search stratified by surface (Clay / Grass / Hard)
- Use bootstrap CIs to test if `w*` differs significantly by surface

### Stage 4 — Tree-based models (Week 6)
- Random Forest + XGBoost on all 40 features
- SHAP values for interpretation
- Compare vs Stage 3 to quantify nonlinearity gain

### Evaluation (same every stage)
- Metrics: accuracy, Brier score, ROC-AUC
- CV: 5-fold on 2011–2021 train set
- Final eval: 2022–2023 held-out test set

---

## Technical Decisions Already Made

| Decision | Rationale |
|----------|-----------|
| Sorted name-pair join key | slam_pbp assigns player order arbitrarily; directional join only matched 30.8% |
| Random A/B assignment per row | Ensures target is balanced 50/50; player-order effects don't leak into features |
| Drop `avg_rally_length` | 44% null + r≈0 with target — imputing it would be unreliable with no signal gain |
| Impute serve stats with per-surface-per-year median | Nulls are from early years (2011–2013); imputing within surface×year cohort minimizes era bias |
| Hold out 2022–2023 as test set | Temporal split prevents any data leakage from recent matches |
| IBM vs. Infosys format detection | Two data formats exist in the raw files; single parsing fails ~50% — multi-fallback logic handles both |

---

## Assignment Format

Each week's Canvas submission needs:
1. A Jupyter notebook at `notebooks/weekN_<slug>.ipynb` committed and pushed to GitHub
2. The notebook should be self-contained and runnable
3. Submit the GitHub repo URL + direct link to the notebook

**Week 2 check-in format** (repeat this structure each week):
- This week's goal
- What you completed
- One key artifact (figure, table, or output cell)
- Biggest blocker
- Plan for next week
- Help needed from instructor/TA

---

## How to Rebuild the Dataset from Scratch

```bash
python src/01_download_data.py
python src/02_build_firstset_features.py
python src/03_join_and_clean.py
python src/04_explore_clean_data.py
```

All scripts use relative paths from the repo root. Run from `figuring_out_claude/`.
