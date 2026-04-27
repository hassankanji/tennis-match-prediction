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
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Features the model uses (agent may change this list)
FEATURES = [
    # Pre-match
    'rank_diff', 'rank_pts_A', 'rank_pts_B',
    'age_diff', 'ht_diff', 'surface_code',
    'hand_A_code', 'hand_B_code',
]


def build_model():
    """
    Baseline: logistic regression on pre-match features only.
    This is the historical-only benchmark — no first-set data used.
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    return model
