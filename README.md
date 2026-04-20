# Tennis Match Prediction
### Predicting ATP Grand Slam Outcomes Using Pre-Match History + Live First-Set Performance
**STAT 390 | Spring 2026**

---

## Research Question

> After the first set of a Grand Slam match is complete, what is the optimal way to combine a player's pre-match historical profile with their live first-set performance to predict who wins the match?

$$P(A\ \text{wins}) = f\bigl(w \cdot \mathbf{x}_{\text{first-set}} + (1-w) \cdot \mathbf{x}_{\text{pre-match}}\bigr)$$

Find $w^*$ — and test whether it varies by surface (clay vs. grass vs. hard).

---

## Progress

| Week | Topic | Notebook |
|------|-------|----------|
| 1 | Project charter, AutoResearch diagram, risk list | [week1_charter.ipynb](notebooks/week1_charter.ipynb) |
| 2 | Data pipeline complete, EDA | [week2_checkin.ipynb](notebooks/week2_checkin.ipynb) |
| 3 | Baseline models (pre-match + first-set LR) | _coming soon_ |
| 4 | Weight grid search + combined model | |
| 5 | Surface-stratified analysis | |
| 6 | Random Forest + XGBoost + SHAP | |
| 7 | Error analysis + year-over-year trends | |
| 8 | Final results + writeup | |

---

## Dataset

| | |
|---|---|
| Source | [JeffSackmann/tennis_slam_pointbypoint](https://github.com/JeffSackmann/tennis_slam_pointbypoint) + [tennis_atp](https://github.com/JeffSackmann/tennis_atp) |
| Scope | ATP men's Grand Slam singles, 2011–2023 |
| Final size | **3,638 matches × 52 features** |
| Target | Binary match winner (`A_won`, balanced 50/50) |
| Train / Test | 2011–2021 / 2022–2023 (temporal split) |

**Feature groups:**
- **Pre-match (11):** ATP ranking, ranking points, age, height, handedness, surface
- **First-set (29):** games won, serve %, serve win %, break points, return %, aces, double faults, winners, unforced errors
- **Dropped:** avg rally length (44% null, r ≈ 0 with target)

---

## Key EDA Findings

| Finding | Value |
|---------|-------|
| Set-1 winner wins match | **77.8%** overall (Clay 81.2%, Grass 77.7%, Hard 76.7%) |
| Higher-ranked player wins | **72.9%** — strong pre-match baseline |
| Best first-set predictor | `return_pts_won_pct` (r = 0.436) |
| First serve % alone | r = 0.063 — getting the serve in ≠ winning points |

---

## Modeling Plan

| Stage | Model | Purpose |
|-------|-------|---------|
| 1 | Logistic regression — pre-match features only | Historical baseline (~67% expected) |
| 2 | Logistic regression — first-set features only | In-match baseline (~79% expected) |
| 3 | Weighted combination + grid search over `w` | Core research question |
| 4 | Random Forest / XGBoost + SHAP | Nonlinear upper bound |

---

## Repo Structure

```
├── src/          # Pipeline scripts (run 01→04 to rebuild dataset)
├── notebooks/    # Weekly check-in notebooks (one per week)
├── data/plots/   # EDA figures
└── reports/      # Proposal and milestone writeups
```

**To rebuild the dataset:**
```bash
python src/01_download_data.py
python src/02_build_firstset_features.py
python src/03_join_and_clean.py
python src/04_explore_clean_data.py
```
