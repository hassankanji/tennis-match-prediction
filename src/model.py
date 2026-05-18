"""
model.py — THE ONLY FILE THE AGENT MODIFIES.

Contains:
  build_model() → returns a fitted-ready sklearn estimator
  FEATURES      → list of feature column names to use

Rules:
  - Only modify this file. Never modify run.py or evaluate.py.
  - build_model() must return an sklearn-compatible estimator
    (has .fit(X, y) and .predict_proba(X)).
  - FEATURES must be a list of strings from evaluate.ALL_FEATURES.
  - The model must run in under 60 seconds on the train split.

Week 6 status: CHAMPION model — confirmed by full ablation study.
  val_brier=0.1648, val_accuracy=76.7%, val_auc=0.834
  Beats all LR variants, GBM, XGBoost, and every feature subset tested.
"""

from sklearn.ensemble import RandomForestClassifier

# ── All 34 features (8 pre-match + 26 first-set) — ablation confirmed these are all needed
FEATURES = [
    # Pre-match
    'rank_diff', 'rank_pts_A', 'rank_pts_B',
    'age_diff', 'ht_diff', 'surface_code',
    'hand_A_code', 'hand_B_code',
    # First-set
    's1_margin', 's1_A_won', 's1_A_srv_pts', 's1_B_srv_pts',
    's1_A_first_srv_pct', 's1_B_first_srv_pct',
    's1_A_first_srv_won_pct', 's1_B_first_srv_won_pct',
    's1_A_second_srv_won_pct', 's1_B_second_srv_won_pct',
    's1_A_bp_faced', 's1_B_bp_faced',
    's1_A_bp_saved_pct', 's1_B_bp_saved_pct',
    's1_A_return_pts_won_pct', 's1_B_return_pts_won_pct',
    's1_A_aces', 's1_B_aces',
    's1_A_df', 's1_B_df',
    's1_A_win', 's1_B_win',
    's1_A_ue', 's1_B_ue',
    's1_pts_won_A', 's1_pts_won_B',
]


def build_model():
    """
    Champion model — Random Forest, n_estimators=100, max_depth=10, all 34 features.
    Discovered by AutoResearch agent (Week 4) and confirmed by full ablation (Week 6).
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
