# Predicting ATP Grand Slam Match Outcomes
## Pre-Match History + Live First-Set Performance

**STAT 390 | Spring 2026**

---

### Research Question

> *After the first set of a Grand Slam match is complete, what is the optimal way to combine a player's historical pre-match profile with their live first-set performance to predict who wins the match?*

Formally: `P(A wins) = f(w · first_set_features + (1−w) · pre_match_features)` — find the optimal `w`.

---

### Dataset

| | Detail |
|---|---|
| Source | [JeffSackmann/tennis_slam_pointbypoint](https://github.com/JeffSackmann/tennis_slam_pointbypoint) + [tennis_atp](https://github.com/JeffSackmann/tennis_atp) |
| Coverage | ATP Grand Slam singles, 2011–2023 (4 slams × 13 years) |
| Final size | **3,638 matches × 52 features** |
| Target | Binary match winner (`A_won`, balanced 50/50 by design) |

**Feature groups:**
- **Pre-match (11):** ATP ranking, ranking points, age, height, hand, surface
- **First-set (29):** games won, serve %, first/second serve win %, break points, return %, aces, double faults, winners, unforced errors
- **Dropped:** avg rally length (44% null, r ≈ 0 with target)

---

### Project Structure

```
├── src/                        # Data pipeline scripts
│   ├── 01_download_data.py     # Download raw tournament files
│   ├── 02_build_firstset_features.py  # Aggregate point-by-point → first-set stats
│   ├── 03_join_and_clean.py    # Join with ATP metadata, clean
│   └── 04_explore_clean_data.py       # EDA, generate plots
│
├── notebooks/                  # Jupyter notebooks (analysis & check-ins)
│   └── week2_checkin.ipynb     # Week 2 deliverable
│
├── data/
│   ├── raw/                    # Raw tournament files (gitignored)
│   ├── interim/                # Intermediate aggregated data (gitignored)
│   ├── clean/                  # Final model-ready CSV (gitignored)
│   └── plots/                  # EDA figures
│
├── reports/                    # Milestone writeups
└── project_proposal.txt        # Full project proposal
```

---

### Pipeline

```bash
python src/01_download_data.py          # ~49 tournament files → data/raw/
python src/02_build_firstset_features.py  # → data/interim/firstset_features.csv
python src/03_join_and_clean.py         # → data/clean/tennis_model_ready.csv
python src/04_explore_clean_data.py     # → data/plots/
```

---

### Key EDA Findings

- Set-1 winner wins the match **77.8%** of the time overall (Clay: 81.2%, Grass: 77.7%, Hard: 76.7%)
- Higher-ranked player wins **72.9%** of matches — strong pre-match baseline
- Strongest first-set predictors: `s1_A_won` (r = 0.556), `s1_margin` (0.548), `s1_return_pts_won_pct` (0.436)
- First serve % alone is nearly useless (r = 0.063); *winning* the serve point matters far more

---

### Modeling Plan

| Stage | Model | Purpose |
|---|---|---|
| 1 | Logistic regression on pre-match features | Historical-only baseline |
| 2 | Logistic regression on first-set features | In-match-only baseline |
| 3 | Weighted combination + grid search over `w` | Core research question |
| 4 | Random Forest / XGBoost + SHAP | Nonlinear benchmark |

Evaluation: **accuracy**, **Brier score**, **ROC-AUC** on held-out 2022–2023 test set.

---

### Progress

- [x] Data download & format handling (IBM Slamtracker vs. Infosys MatchBeats)
- [x] First-set feature engineering (10,377 matches processed)
- [x] ATP metadata join (3,638 clean ATP men's matches)
- [x] Exploratory data analysis
- [ ] Baseline models (Stage 1 & 2)
- [ ] Weight grid search (Stage 3)
- [ ] Tree-based models (Stage 4)
