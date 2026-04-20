"""
Tennis Match Prediction Project
Step 3: Clean and engineer features from the raw ATP dataset.

Output schema (one row = one match, from the perspective of "player A vs player B"):
  ── Match context ──────────────────────────────────────────────────────────────
  tourney_date, surface, tourney_level, round, best_of

  ── Pre-match features (historical / prior to first serve) ────────────────────
  rank_A, rank_B, rank_diff          (A_rank − B_rank; positive → A is lower-ranked)
  rank_pts_A, rank_pts_B
  age_A, age_B
  ht_A, ht_B
  hand_A, hand_B

  ── First-set features (observed during the match) ────────────────────────────
  s1_games_A, s1_games_B             raw game counts
  s1_margin                          s1_games_A − s1_games_B  (+ → A led set 1)
  s1_tiebreak                        1 if first set went to a tiebreak
  s1_A_won                           1 if A won the first set

  ── Target ────────────────────────────────────────────────────────────────────
  A_won                              1 if player A won the match
                                     (A is randomly assigned winner or loser,
                                      so the training set is balanced)

NOTE: "A" and "B" are randomly assigned per row (50/50 flip) so the model
      can't simply learn "A always wins".  The original winner/loser identity
      is preserved in winner_name / loser_name for reference.
"""

import io
import os
import random
import sys
import warnings
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

RAW_PATH   = os.path.join("data", "raw",   "atp_matches_2000_2023.csv")
CLEAN_PATH = os.path.join("data", "clean", "atp_clean.csv")
os.makedirs(os.path.join("data", "clean"), exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading raw data …")
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"  Raw shape: {df.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DROP ROWS WE CAN'T USE
# ═══════════════════════════════════════════════════════════════════════════════
initial_len = len(df)
drop_log = {}

# 2a. Walkovers / retirements / defaults — score is incomplete
retire_mask = df["score"].str.contains(r"W/O|RET|DEF\.", na=True, regex=True)
df = df[~retire_mask]
drop_log["walkovers/retirements"] = initial_len - len(df)

# 2b. Rows missing the score column entirely
df = df.dropna(subset=["score"])
drop_log["missing score"] = initial_len - len(df) - sum(drop_log.values())

# 2c. Rows missing both players' rankings (can't build pre-match features)
df = df.dropna(subset=["winner_rank", "loser_rank"])
drop_log["missing rank"] = initial_len - len(df) - sum(drop_log.values())

print(f"\n  Rows dropped:")
for reason, n in drop_log.items():
    print(f"    {reason:<35} {n:>7,}")
print(f"  Remaining: {len(df):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PARSE FIRST-SET SCORE
# ═══════════════════════════════════════════════════════════════════════════════
def parse_first_set(score_str):
    """Return (w_games, l_games, tiebreak) for the first set."""
    if not isinstance(score_str, str):
        return np.nan, np.nan, np.nan
    sets = score_str.strip().split()
    if not sets:
        return np.nan, np.nan, np.nan
    first = sets[0]
    tiebreak = int("(" in first)
    first_clean = first.split("(")[0]
    parts = first_clean.split("-")
    if len(parts) != 2:
        return np.nan, np.nan, np.nan
    try:
        return int(parts[0]), int(parts[1]), tiebreak
    except ValueError:
        return np.nan, np.nan, np.nan

parsed = df["score"].apply(parse_first_set)
df["s1_w_games"] = parsed.apply(lambda x: x[0])
df["s1_l_games"] = parsed.apply(lambda x: x[1])
df["s1_tiebreak"] = parsed.apply(lambda x: x[2])

# Drop rows where first-set parse failed
before = len(df)
df = df.dropna(subset=["s1_w_games", "s1_l_games"])
print(f"\n  Dropped {before - len(df):,} rows with unparseable first-set score")
print(f"  Remaining: {len(df):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PARSE DATE
# ═══════════════════════════════════════════════════════════════════════════════
df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  RANDOM FLIP: assign "player A" and "player B" per row
#     50 % of rows → A = winner ; 50 % → A = loser
#     This prevents the model learning a "slot 1 always wins" bias
# ═══════════════════════════════════════════════════════════════════════════════
flip = np.random.rand(len(df)) < 0.5      # True → A = winner

def pick(col_winner, col_loser, flip_vec, df):
    return np.where(flip_vec, df[col_winner], df[col_loser])

clean = pd.DataFrame()

# Reference columns (not features — kept for inspection)
clean["tourney_date"]   = df["tourney_date"].values
clean["winner_name"]    = df["winner_name"].values
clean["loser_name"]     = df["loser_name"].values
clean["surface"]        = df["surface"].values
clean["tourney_level"]  = df["tourney_level"].values
clean["round"]          = df["round"].values
clean["best_of"]        = df["best_of"].values

# ── Pre-match features ─────────────────────────────────────────────────────────
clean["rank_A"]         = pick("winner_rank",        "loser_rank",        flip, df)
clean["rank_B"]         = pick("loser_rank",         "winner_rank",       flip, df)
clean["rank_pts_A"]     = pick("winner_rank_points", "loser_rank_points", flip, df)
clean["rank_pts_B"]     = pick("loser_rank_points",  "winner_rank_points",flip, df)
clean["age_A"]          = pick("winner_age",         "loser_age",         flip, df)
clean["age_B"]          = pick("loser_age",          "winner_age",        flip, df)
clean["ht_A"]           = pick("winner_ht",          "loser_ht",          flip, df)
clean["ht_B"]           = pick("loser_ht",           "winner_ht",         flip, df)
clean["hand_A"]         = pick("winner_hand",        "loser_hand",        flip, df)
clean["hand_B"]         = pick("loser_hand",         "winner_hand",       flip, df)

# Derived pre-match
clean["rank_diff"]      = clean["rank_A"].astype(float) - clean["rank_B"].astype(float)
clean["age_diff"]       = clean["age_A"].astype(float)  - clean["age_B"].astype(float)
clean["ht_diff"]        = clean["ht_A"].astype(float)   - clean["ht_B"].astype(float)

# ── First-set features ─────────────────────────────────────────────────────────
w_games = df["s1_w_games"].astype(float).values
l_games = df["s1_l_games"].astype(float).values

clean["s1_games_A"]  = np.where(flip, w_games, l_games)
clean["s1_games_B"]  = np.where(flip, l_games, w_games)
clean["s1_margin"]   = clean["s1_games_A"] - clean["s1_games_B"]   # + → A led
clean["s1_tiebreak"] = df["s1_tiebreak"].values
clean["s1_A_won"]    = (clean["s1_games_A"] > clean["s1_games_B"]).astype(int)

# ── Target ─────────────────────────────────────────────────────────────────────
clean["A_won"] = flip.astype(int)    # 1 if A = winner, 0 if A = loser

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ENCODE CATEGORICALS
# ═══════════════════════════════════════════════════════════════════════════════
# Surface → integer (keep original string too)
surface_map = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
clean["surface_code"] = clean["surface"].map(surface_map).fillna(-1).astype(int)

# Hand → integer
hand_map = {"R": 0, "L": 1, "U": 2}
clean["hand_A_code"] = clean["hand_A"].map(hand_map).fillna(2).astype(int)
clean["hand_B_code"] = clean["hand_B"].map(hand_map).fillna(2).astype(int)

# best_of → already numeric
clean["best_of"] = pd.to_numeric(clean["best_of"], errors="coerce").fillna(3).astype(int)

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  FINAL QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Final dataset quality check ────────────────────────────────────────")

# Sanity: first-set games should be 6 or 7 (plus edge cases)
valid_s1 = (
    clean["s1_games_A"].between(0, 7) &
    clean["s1_games_B"].between(0, 7)
)
print(f"  First-set scores in valid range: {valid_s1.sum():,} / {len(clean):,}")
clean = clean[valid_s1]

print(f"\n  Clean shape: {clean.shape}")
print(f"  Target balance: A_won=1 → {clean['A_won'].mean()*100:.1f}%  "
      f"(should be ~50%)")
print(f"\n  Null rates in feature columns:")
feat_cols = [c for c in clean.columns
             if c not in ("winner_name","loser_name","tourney_date","surface","hand_A","hand_B")]
null_rates = (clean[feat_cols].isnull().mean() * 100).sort_values(ascending=False)
null_rates = null_rates[null_rates > 0]
if null_rates.empty:
    print("    All feature columns complete.")
else:
    for col, rate in null_rates.items():
        print(f"    {col:<30} {rate:.1f}% missing")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  SAVE
# ═══════════════════════════════════════════════════════════════════════════════
clean.to_csv(CLEAN_PATH, index=False)
print(f"\nClean dataset saved → {CLEAN_PATH}")
print(f"Final shape: {clean.shape[0]:,} rows  ×  {clean.shape[1]} columns")

# ── Schema printout ────────────────────────────────────────────────────────────
print("\n── Output schema ───────────────────────────────────────────────────────")
print(f"  {'Column':<25} {'Dtype':<12} {'Non-Null %':>10}  Description")
print("  " + "-" * 75)
descriptions = {
    "tourney_date":  "Date of the tournament",
    "winner_name":   "Original match winner (reference only)",
    "loser_name":    "Original match loser  (reference only)",
    "surface":       "Court surface (Hard / Clay / Grass / Carpet)",
    "tourney_level": "Tournament tier (G=Grand Slam, M=Masters, A=ATP250/500 …)",
    "round":         "Match round (R128 → F)",
    "best_of":       "Best of 3 or 5 sets",
    "rank_A":        "Pre-match ATP rank of player A",
    "rank_B":        "Pre-match ATP rank of player B",
    "rank_pts_A":    "Ranking points of player A",
    "rank_pts_B":    "Ranking points of player B",
    "age_A":         "Age of player A",
    "age_B":         "Age of player B",
    "ht_A":          "Height (cm) of player A",
    "ht_B":          "Height (cm) of player B",
    "hand_A":        "Dominant hand of player A (R/L/U)",
    "hand_B":        "Dominant hand of player B",
    "rank_diff":     "rank_A − rank_B  (+ → A is lower-ranked)",
    "age_diff":      "age_A − age_B",
    "ht_diff":       "ht_A − ht_B",
    "s1_games_A":    "Games won by A in first set",
    "s1_games_B":    "Games won by B in first set",
    "s1_margin":     "s1_games_A − s1_games_B  (+ → A led first set)",
    "s1_tiebreak":   "1 if first set reached a tiebreak",
    "s1_A_won":      "1 if player A won the first set",
    "surface_code":  "Surface encoded (Hard=0, Clay=1, Grass=2, Carpet=3)",
    "hand_A_code":   "Hand encoded (R=0, L=1, U=2)",
    "hand_B_code":   "Hand encoded (R=0, L=1, U=2)",
    "A_won":         "TARGET — 1 if player A won the match",
}
for col in clean.columns:
    dtype  = str(clean[col].dtype)
    pct    = f"{(1 - clean[col].isnull().mean())*100:.1f}%"
    desc   = descriptions.get(col, "")
    print(f"  {col:<25} {dtype:<12} {pct:>10}  {desc}")

print("\nDone! Ready for modelling.")
