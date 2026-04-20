"""
Tennis Match Prediction Project
Step 3: Join first-set features (from slam_pointbypoint) with pre-match
        historical features (ATP rankings, age, height from tennis_atp).

Final schema — one row per match, player A vs player B (randomly assigned):

  ── Identifiers ───────────────────────────────────────────────────────────
  match_id, year, slam, round, player_A, player_B

  ── Pre-match features (known before the first ball is struck) ────────────
  rank_A, rank_B, rank_diff        ATP ranking (lower = better)
  rank_pts_A, rank_pts_B
  age_A, age_B, age_diff
  ht_A, ht_B, ht_diff              height in cm
  hand_A, hand_B                   R / L / U
  surface_code                     Hard=0  Clay=1  Grass=2

  ── First-set features (observed during set 1) ───────────────────────────
  s1_games_A, s1_games_B           games won in set 1
  s1_margin                        s1_games_A − s1_games_B
  s1_A_won                         1 if A won set 1
  s1_A_srv_pts, s1_B_srv_pts       serve points played in set 1
  s1_A_first_srv_pct               % first serves in
  s1_A_first_srv_won_pct           % of first-serve points won
  s1_A_second_srv_won_pct          % of second-serve points won
  s1_A_bp_faced, s1_B_bp_faced     break point opportunities faced
  s1_A_bp_saved_pct                % of break points saved
  s1_A_return_pts_won_pct          % of opponent's serve pts won (return game)
  s1_A_aces, s1_B_aces
  s1_A_df, s1_B_df                 double faults
  s1_A_win, s1_B_win               winners
  s1_A_ue, s1_B_ue                 unforced errors
  s1_avg_rally                     avg rally length

  [same set of stats for player B]

  ── Target ────────────────────────────────────────────────────────────────
  A_won                            1 if player A won the match
"""

import io
import os
import sys
import warnings
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
np.random.seed(42)

INTERIM_DIR = os.path.join("data", "interim")
ATP_PATH    = os.path.join("data", "raw", "atp_grand_slams_2000_2023.csv")
CLEAN_DIR   = os.path.join("data", "clean")
os.makedirs(CLEAN_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
fs  = pd.read_csv(os.path.join(INTERIM_DIR, "firstset_features.csv"), low_memory=False)
atp = pd.read_csv(ATP_PATH, low_memory=False)

print(f"  First-set features: {len(fs):,} matches")
print(f"  ATP Grand Slam rows: {len(atp):,} matches")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Prepare ATP side: build a name-lookup table
# ══════════════════════════════════════════════════════════════════════════════
SLAM_MAP = {
    "ausopen":     "Australian Open",
    "frenchopen":  "Roland Garros",
    "wimbledon":   "Wimbledon",
    "usopen":      "US Open",
}

atp["year"]       = pd.to_datetime(atp["tourney_date"], format="%Y%m%d", errors="coerce").dt.year
# Normalise "Us Open" → "US Open" (typo present in some ATP data files)
atp["slam_label"] = atp["tourney_name"].str.strip().str.replace("Us Open", "US Open", regex=False)

# Keep only the columns we need for pre-match features
atp_cols = [
    "year", "slam_label",
    "winner_name", "loser_name",
    "winner_rank", "loser_rank",
    "winner_rank_points", "loser_rank_points",
    "winner_age", "loser_age",
    "winner_ht", "loser_ht",
    "winner_hand", "loser_hand",
    "surface", "round",
]
atp_slim = atp[[c for c in atp_cols if c in atp.columns]].copy()

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Join firstset_features → ATP pre-match features
#     Key: year + slam + sorted player-name pair (order-independent)
#     The slam_pbp assigns player1/player2 by draw order (arbitrary),
#     so we must not assume player1 = winner or player1 = loser.
# ══════════════════════════════════════════════════════════════════════════════
fs["slam_label"] = fs["slam"].map(SLAM_MAP)
fs["year_int"]   = pd.to_numeric(fs["year"], errors="coerce").astype("Int64")
atp_slim["year"] = atp_slim["year"].astype("Int64")

# Sorted name-pair key (order-independent)
fs["name_pair"] = fs.apply(
    lambda r: "|".join(sorted([str(r.player1), str(r.player2)])), axis=1
)
atp_slim["name_pair"] = atp_slim.apply(
    lambda r: "|".join(sorted([str(r.winner_name), str(r.loser_name)])), axis=1
)

# Single join on (year, slam_label, name_pair)
joined = fs.merge(
    atp_slim,
    left_on=["year_int", "slam_label", "name_pair"],
    right_on=["year",     "slam_label", "name_pair"],
    how="left",
)

# Assign p1/p2 suffix based on which ATP row player is player1 vs player2
# winner_name from ATP might be player1 or player2 in slam_pbp
is_p1_winner = joined["winner_name"] == joined["player1"]

for stat, w_col, l_col in [
    ("rank",  "winner_rank",         "loser_rank"),
    ("rpts",  "winner_rank_points",  "loser_rank_points"),
    ("age",   "winner_age",          "loser_age"),
    ("ht",    "winner_ht",           "loser_ht"),
    ("hand",  "winner_hand",         "loser_hand"),
]:
    if w_col in joined.columns and l_col in joined.columns:
        joined[f"{stat}_p1"] = np.where(is_p1_winner, joined[w_col], joined[l_col])
        joined[f"{stat}_p2"] = np.where(is_p1_winner, joined[l_col], joined[w_col])

match_rate = joined["rank_p1"].notna().mean()
print(f"\n  ATP join match rate: {match_rate*100:.1f}%")
print(f"  Rows with pre-match features: {joined['rank_p1'].notna().sum():,}")
print(f"  (Remaining are WTA or 2024 matches not in ATP data)")

# Filter to ATP men's only (rows with at least one pre-match feature)
joined = joined[joined["rank_p1"].notna()].copy()
print(f"  After filtering to ATP matches only: {len(joined):,}")

# Surface from ATP data
surf_fallback = {"ausopen": "Hard", "frenchopen": "Clay",
                 "wimbledon": "Grass", "usopen": "Hard"}
if "surface" not in joined.columns:
    joined["surface"] = joined["slam"].map(surf_fallback)
else:
    joined["surface"] = joined["surface"].fillna(joined["slam"].map(surf_fallback))

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Random A/B flip (50/50) so model can't learn "player 1 always wins"
# ══════════════════════════════════════════════════════════════════════════════
flip = np.random.rand(len(joined)) < 0.5   # True → A = player1 (winner in slam data if winner==1)

def pick(col1, col2):
    """Return np.where(flip, col1, col2) with correct dtypes."""
    return np.where(flip, joined[col1].values, joined[col2].values)

clean = pd.DataFrame()

# Identifiers
clean["match_id"]  = joined["match_id"].values
clean["year"]      = joined["year_int"].values
clean["slam"]      = joined["slam"].values
# round comes from ATP data (slam_pbp matches files have it null)
clean["round"]     = joined["round"].values if "round" in joined.columns else np.nan
clean["player_A"]  = np.where(flip, joined["player1"].values, joined["player2"].values)
clean["player_B"]  = np.where(flip, joined["player2"].values, joined["player1"].values)

# ── Pre-match features ─────────────────────────────────────────────────────────
clean["rank_A"]      = pick("rank_p1",  "rank_p2")
clean["rank_B"]      = pick("rank_p2",  "rank_p1")
clean["rank_pts_A"]  = pick("rpts_p1",  "rpts_p2")
clean["rank_pts_B"]  = pick("rpts_p2",  "rpts_p1")
clean["age_A"]       = pick("age_p1",   "age_p2")
clean["age_B"]       = pick("age_p2",   "age_p1")
clean["ht_A"]        = pick("ht_p1",    "ht_p2")
clean["ht_B"]        = pick("ht_p2",    "ht_p1")
clean["hand_A"]      = pick("hand_p1",  "hand_p2")
clean["hand_B"]      = pick("hand_p2",  "hand_p1")
clean["rank_diff"]   = clean["rank_A"].astype(float) - clean["rank_B"].astype(float)
clean["age_diff"]    = clean["age_A"].astype(float)  - clean["age_B"].astype(float)
clean["ht_diff"]     = clean["ht_A"].astype(float)   - clean["ht_B"].astype(float)

# Surface (from ATP data; fill any gaps from slam name)
surf_fallback2 = {"ausopen": "Hard", "frenchopen": "Clay", "wimbledon": "Grass", "usopen": "Hard"}
surface_series = pd.Series(joined["surface"].values).fillna(
    pd.Series(joined["slam"].values).map(surf_fallback2)
)
surf_map = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
clean["surface"]      = surface_series.values
clean["surface_code"] = surface_series.map(surf_map).fillna(-1).astype(int)

# Hand encoding
hand_map = {"R": 0, "L": 1, "U": 2}
clean["hand_A_code"] = pd.Series(clean["hand_A"]).map(hand_map).fillna(2).astype(int)
clean["hand_B_code"] = pd.Series(clean["hand_B"]).map(hand_map).fillna(2).astype(int)

# ── First-set features (swap P1/P2 if flipped) ────────────────────────────────
# player1 stats have prefix "s1_p1_*", player2 have "s1_p2_*"
# After flip: A stats come from p1 if flip=True, else from p2

def s1pick(stat):
    """Pick first-set stat for A (flip-aware)."""
    c1 = f"s1_p1_{stat}"
    c2 = f"s1_p2_{stat}"
    if c1 in joined.columns and c2 in joined.columns:
        return np.where(flip, joined[c1].values, joined[c2].values)
    return np.full(len(joined), np.nan)

def s1pick_b(stat):
    c1 = f"s1_p1_{stat}"
    c2 = f"s1_p2_{stat}"
    if c1 in joined.columns and c2 in joined.columns:
        return np.where(flip, joined[c2].values, joined[c1].values)
    return np.full(len(joined), np.nan)

# Games in set 1
clean["s1_games_A"]  = np.where(flip, joined["s1_games_p1"].values, joined["s1_games_p2"].values)
clean["s1_games_B"]  = np.where(flip, joined["s1_games_p2"].values, joined["s1_games_p1"].values)
clean["s1_margin"]   = clean["s1_games_A"].astype(float) - clean["s1_games_B"].astype(float)
clean["s1_A_won"]    = (clean["s1_games_A"].astype(float) > clean["s1_games_B"].astype(float)).astype(int)

# Serve stats
for stat in ["srv_pts", "first_srv_pct", "first_srv_won_pct",
             "second_srv_won_pct", "bp_faced", "bp_saved_pct"]:
    clean[f"s1_A_{stat}"] = s1pick(stat)
    clean[f"s1_B_{stat}"] = s1pick_b(stat)

# Return
clean["s1_A_return_pts_won_pct"] = np.where(flip,
    joined["s1_p1_return_pts_won_pct"].values if "s1_p1_return_pts_won_pct" in joined.columns else np.nan,
    joined["s1_p2_return_pts_won_pct"].values if "s1_p2_return_pts_won_pct" in joined.columns else np.nan)
clean["s1_B_return_pts_won_pct"] = np.where(flip,
    joined["s1_p2_return_pts_won_pct"].values if "s1_p2_return_pts_won_pct" in joined.columns else np.nan,
    joined["s1_p1_return_pts_won_pct"].values if "s1_p1_return_pts_won_pct" in joined.columns else np.nan)

# Shot stats
for stat in ["aces", "df", "win", "ue", "fe"]:
    clean[f"s1_A_{stat}"] = s1pick(stat)
    clean[f"s1_B_{stat}"] = s1pick_b(stat)

# Rally (not player-specific)
if "s1_avg_rally" in joined.columns:
    clean["s1_avg_rally"] = joined["s1_avg_rally"].values

# Points won in set 1
clean["s1_pts_won_A"] = np.where(flip,
    joined["s1_pts_won_p1"].values if "s1_pts_won_p1" in joined.columns else np.nan,
    joined["s1_pts_won_p2"].values if "s1_pts_won_p2" in joined.columns else np.nan)
clean["s1_pts_won_B"] = np.where(flip,
    joined["s1_pts_won_p2"].values if "s1_pts_won_p2" in joined.columns else np.nan,
    joined["s1_pts_won_p1"].values if "s1_pts_won_p1" in joined.columns else np.nan)

# ── Target ─────────────────────────────────────────────────────────────────────
# winner in joined: 1=player1 won, 2=player2 won
# flip: True → A=player1  → A_won if winner==1
#       False → A=player2  → A_won if winner==2
winner_vec = joined["winner"].values.astype(float)
clean["A_won"] = np.where(flip, (winner_vec == 1).astype(int),
                                (winner_vec == 2).astype(int))

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Final quality filters
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nBefore final filtering: {len(clean):,} rows")

# Must have set 1 score
clean = clean.dropna(subset=["s1_games_A", "s1_games_B"])

# Set 1 games must be in valid range
valid = (
    clean["s1_games_A"].between(0, 7) &
    clean["s1_games_B"].between(0, 7)
)
clean = clean[valid]

# Must have the target
clean = clean.dropna(subset=["A_won"])

# Drop columns that are >90% null (not useful for modelling)
high_null = [c for c in clean.columns if clean[c].isnull().mean() > 0.9]
if high_null:
    print(f"  Dropping high-null columns (>90% missing): {high_null}")
    clean = clean.drop(columns=high_null)

print(f"After final filtering:  {len(clean):,} rows")
print(f"Target balance:         A_won=1 → {clean['A_won'].mean()*100:.1f}%  (expected ~50%)")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Save
# ══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(CLEAN_DIR, "tennis_model_ready.csv")
clean.to_csv(out_path, index=False)

print(f"\nClean dataset saved → {out_path}")
print(f"Final shape: {clean.shape[0]:,} rows  ×  {clean.shape[1]} columns")

# ── Schema ─────────────────────────────────────────────────────────────────────
print(f"\n{'Column':<35} {'Dtype':<12} {'Non-Null%':>9}  Notes")
print("-" * 80)
notes = {
    "match_id":                 "Unique match identifier",
    "year":                     "Tournament year",
    "slam":                     "ausopen / frenchopen / wimbledon / usopen",
    "round":                    "R128 / R64 / R32 / R16 / QF / SF / F",
    "player_A":                 "Name of player A (randomly assigned)",
    "player_B":                 "Name of player B",
    "rank_A":                   "ATP ranking at time of match",
    "rank_B":                   "ATP ranking at time of match",
    "rank_diff":                "rank_A − rank_B  (+→A lower-ranked)",
    "rank_pts_A":               "Ranking points",
    "age_A":                    "Age in years",
    "ht_A":                     "Height in cm",
    "hand_A":                   "R/L/U",
    "surface":                  "Hard / Clay / Grass",
    "surface_code":             "0=Hard 1=Clay 2=Grass",
    "s1_games_A":               "Games won by A in set 1",
    "s1_margin":                "s1_games_A − s1_games_B",
    "s1_A_won":                 "1 if A won set 1",
    "s1_A_first_srv_pct":       "% of first serves in (set 1)",
    "s1_A_first_srv_won_pct":   "% of first-serve points won",
    "s1_A_second_srv_won_pct":  "% of second-serve points won",
    "s1_A_bp_faced":            "Break points faced in set 1",
    "s1_A_bp_saved_pct":        "% of break points saved",
    "s1_A_return_pts_won_pct":  "% of opponent serve points won (return)",
    "s1_A_aces":                "Aces in set 1",
    "s1_A_df":                  "Double faults in set 1",
    "s1_A_win":                 "Winners in set 1",
    "s1_A_ue":                  "Unforced errors in set 1",
    "s1_avg_rally":             "Average rally length in set 1",
    "s1_pts_won_A":             "Total points won by A in set 1",
    "A_won":                    "TARGET — 1 if player A won the match",
}
for col in clean.columns:
    dtype   = str(clean[col].dtype)
    pct_val = f"{(1 - clean[col].isnull().mean())*100:.0f}%"
    note    = notes.get(col, "")
    print(f"  {col:<33} {dtype:<12} {pct_val:>6}   {note}")

print("\nReady for modelling.")
