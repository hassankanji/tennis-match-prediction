"""
evaluate.py — FIXED EVALUATION HARNESS. DO NOT MODIFY.

Loads the tennis dataset, enforces the train/val/test split,
imputes missing values, and exposes evaluate() for run.py.

Split:
  train : 2011–2020  (2,847 matches)
  val   : 2021       (  278 matches)  ← agent optimizes against this
  test  : 2022–2023  (  513 matches)  ← never touched until final eval
"""

import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "clean" / "tennis_model_ready.csv"

# ── Feature groups (agent may select subsets via model.py FEATURES list)
ALL_PREMATCH = [
    'rank_diff', 'rank_pts_A', 'rank_pts_B',
    'age_diff', 'ht_diff', 'surface_code',
    'hand_A_code', 'hand_B_code',
]
ALL_FIRSTSET = [
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
ALL_FEATURES = ALL_PREMATCH + ALL_FIRSTSET


def _load_splits():
    df = pd.read_csv(DATA_PATH)
    train = df[df['year'] <= 2020].copy()
    val   = df[df['year'] == 2021].copy()
    test  = df[df['year'] >= 2022].copy()
    return train, val, test


def _prep(train, val, features):
    """Median-impute on train statistics, apply to val."""
    imp = SimpleImputer(strategy='median')
    X_train = imp.fit_transform(train[features])
    X_val   = imp.transform(val[features])
    y_train = train['A_won'].values
    y_val   = val['A_won'].values
    return X_train, y_train, X_val, y_val


def evaluate(model, features):
    """
    Train model on train split, evaluate on val split.
    Returns dict with val_brier, val_accuracy, val_auc.
    Called by run.py — do not call directly from model.py.
    """
    train, val, _ = _load_splits()
    X_train, y_train, X_val, y_val = _prep(train, val, features)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return {
        'val_brier'   : round(brier_score_loss(y_val, proba), 6),
        'val_accuracy': round(accuracy_score(y_val, pred), 6),
        'val_auc'     : round(roc_auc_score(y_val, proba), 6),
        'n_val'       : len(y_val),
    }


def evaluate_test(model, features):
    """
    FINAL EVALUATION ONLY — called once at end of project.
    Trains on train+val, evaluates on held-out test set (2022-2023).
    Do NOT call this during the experiment loop.
    """
    train_df, val_df, test_df = _load_splits()
    trainval = pd.concat([train_df, val_df])
    imp = SimpleImputer(strategy='median')
    X_tv   = imp.fit_transform(trainval[features])
    X_test = imp.transform(test_df[features])
    y_tv   = trainval['A_won'].values
    y_test = test_df['A_won'].values
    model.fit(X_tv, y_tv)
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return {
        'test_brier'   : round(brier_score_loss(y_test, proba), 6),
        'test_accuracy': round(accuracy_score(y_test, pred), 6),
        'test_auc'     : round(roc_auc_score(y_test, proba), 6),
        'n_test'       : len(y_test),
    }
