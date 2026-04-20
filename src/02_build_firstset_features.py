"""
Tennis Match Prediction Project
Step 2: Aggregate point-by-point data into first-set statistics per match.

Handles two data formats from the repo:
  - IBM Slamtracker (2011-2017 + Wimbledon/USO after 2018):
      P1GamesWon/P2GamesWon populated per row, P1FirstSrvIn/P2FirstSrvIn available
  - Infosys MatchBeats (2018-2022 AO/FO):
      P1GamesWon etc. are NaN per row; use GameWinner + ServeNumber instead

Match winner is derived from set wins in the points file (the matches metadata
winner column is null in all files).

Output: data/interim/firstset_features.csv
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

SLAM_DIR    = os.path.join("data", "raw", "slam_pbp")
INTERIM_DIR = os.path.join("data", "interim")
os.makedirs(INTERIM_DIR, exist_ok=True)

TOURNAMENTS = [
    ("2011","ausopen"),("2011","frenchopen"),("2011","usopen"),("2011","wimbledon"),
    ("2012","ausopen"),("2012","frenchopen"),("2012","usopen"),("2012","wimbledon"),
    ("2013","ausopen"),("2013","frenchopen"),("2013","usopen"),("2013","wimbledon"),
    ("2014","ausopen"),("2014","frenchopen"),("2014","usopen"),("2014","wimbledon"),
    ("2015","ausopen"),("2015","frenchopen"),("2015","usopen"),("2015","wimbledon"),
    ("2016","ausopen"),("2016","frenchopen"),("2016","usopen"),("2016","wimbledon"),
    ("2017","ausopen"),("2017","frenchopen"),("2017","usopen"),("2017","wimbledon"),
    ("2018","ausopen"),("2018","frenchopen"),("2018","usopen"),("2018","wimbledon"),
    ("2019","ausopen"),("2019","frenchopen"),("2019","usopen"),("2019","wimbledon"),
    ("2020","ausopen"),("2020","frenchopen"),("2020","usopen"),
    ("2021","ausopen"),("2021","frenchopen"),("2021","usopen"),("2021","wimbledon"),
    ("2022","usopen"),("2022","wimbledon"),
    ("2023","usopen"),("2023","wimbledon"),
    ("2024","usopen"),("2024","wimbledon"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def safe_pct(num, den):
    return round(float(num) / float(den), 4) if den and den > 0 else np.nan

def to_int(x):
    try:
        v = float(x)
        return int(v) if not np.isnan(v) else np.nan
    except Exception:
        return np.nan

# ── Derive match winner from set results in points file ────────────────────────
def get_match_winner(match_pts: pd.DataFrame) -> int:
    """
    Determine which player (1 or 2) won the match by counting set wins.
    Tries in order: SetWinner column → P1/P2GamesWon → GameWinner counts.
    Returns 1, 2, or np.nan if indeterminate.
    """
    set_wins = {1: 0, 2: 0}
    for set_no, grp in match_pts.groupby("SetNo"):
        # Method 1: SetWinner column
        sw_vals = grp["SetWinner"].dropna()
        sw_vals = sw_vals[sw_vals.isin([1, 2])]
        if len(sw_vals) > 0:
            set_wins[int(sw_vals.iloc[-1])] += 1
            continue

        # Method 2: P1GamesWon / P2GamesWon on the last row
        last = grp.iloc[-1]
        p1g = pd.to_numeric(last.get("P1GamesWon", np.nan), errors="coerce")
        p2g = pd.to_numeric(last.get("P2GamesWon", np.nan), errors="coerce")
        if not (np.isnan(p1g) or np.isnan(p2g)):
            if p1g > p2g:
                set_wins[1] += 1
            elif p2g > p1g:
                set_wins[2] += 1
            continue

        # Method 3: GameWinner column (MatchBeats format, when populated)
        if "GameNo" in grp.columns and "GameWinner" in grp.columns:
            tmp = grp[["GameNo","GameWinner"]].copy()
            tmp["GameWinner"] = pd.to_numeric(tmp["GameWinner"], errors="coerce")
            tmp = tmp.dropna(subset=["GameNo","GameWinner"])
            tmp = tmp[tmp["GameWinner"].isin([1,2])]
            if len(tmp) > 0:
                game_winners = tmp.groupby("GameNo")["GameWinner"].last()
                p1g = int((game_winners == 1).sum())
                p2g = int((game_winners == 2).sum())
                if p1g > p2g:
                    set_wins[1] += 1
                elif p2g > p1g:
                    set_wins[2] += 1
                continue

        # Method 4: Infer game winner from last PointWinner in each game
        # (2018+ AO/FO — only PointWinner + GameNo are available)
        if "GameNo" in grp.columns and "PointWinner" in grp.columns:
            tmp = grp[["GameNo","PointNumber","PointWinner"]].copy()
            tmp["GameNo"]      = pd.to_numeric(tmp["GameNo"],      errors="coerce")
            tmp["PointWinner"] = pd.to_numeric(tmp["PointWinner"], errors="coerce")
            tmp = tmp.dropna(subset=["GameNo","PointWinner"])
            if len(tmp) > 0:
                last_pts = tmp.groupby("GameNo").last()["PointWinner"]
                last_pts = last_pts[last_pts.isin([1, 2])]
                p1g = int((last_pts == 1).sum())
                p2g = int((last_pts == 2).sum())
                if p1g > p2g:
                    set_wins[1] += 1
                elif p2g > p1g:
                    set_wins[2] += 1

    if set_wins[1] >= 3:
        return 1
    elif set_wins[2] >= 3:
        return 2
    elif set_wins[1] == 2 and set_wins[2] <= 1:   # best-of-3 (WTA)
        return 1
    elif set_wins[2] == 2 and set_wins[1] <= 1:
        return 2
    return np.nan

# ── Derive set 1 game score from GameWinner (MatchBeats fallback) ──────────────
def get_s1_score_from_games(s1: pd.DataFrame):
    """
    When P1GamesWon/P2GamesWon are NaN, derive set score from game winners.
    Tries GameWinner column first, then last PointWinner per game.
    Returns (p1_games, p2_games, set_winner).
    """
    if "GameNo" not in s1.columns:
        return np.nan, np.nan, np.nan

    # Try GameWinner column
    if "GameWinner" in s1.columns:
        tmp = s1[["GameNo","GameWinner"]].copy()
        tmp["GameWinner"] = pd.to_numeric(tmp["GameWinner"], errors="coerce")
        tmp = tmp.dropna(subset=["GameNo","GameWinner"])
        tmp = tmp[tmp["GameWinner"].isin([1,2])]
        if len(tmp) > 0:
            gw = tmp.groupby("GameNo")["GameWinner"].last()
            p1g, p2g = int((gw==1).sum()), int((gw==2).sum())
            sw = 1 if p1g > p2g else 2 if p2g > p1g else np.nan
            return p1g, p2g, sw

    # Fallback: last PointWinner per game
    if "PointWinner" in s1.columns:
        tmp = s1[["GameNo","PointWinner"]].copy()
        tmp["GameNo"]      = pd.to_numeric(tmp["GameNo"],      errors="coerce")
        tmp["PointWinner"] = pd.to_numeric(tmp["PointWinner"], errors="coerce")
        tmp = tmp.dropna()
        if len(tmp) > 0:
            gw = tmp.groupby("GameNo")["PointWinner"].last()
            gw = gw[gw.isin([1, 2])]
            p1g, p2g = int((gw==1).sum()), int((gw==2).sum())
            sw = 1 if p1g > p2g else 2 if p2g > p1g else np.nan
            return p1g, p2g, sw

    return np.nan, np.nan, np.nan

# ── Aggregate one match's set-1 rows ──────────────────────────────────────────
def agg_match_s1(s1: pd.DataFrame) -> dict:
    d = {}
    if len(s1) < 8:
        return {}

    # ── Set 1 game score ──────────────────────────────────────────────────────
    last = s1.iloc[-1]
    p1g = pd.to_numeric(last.get("P1GamesWon", np.nan), errors="coerce")
    p2g = pd.to_numeric(last.get("P2GamesWon", np.nan), errors="coerce")
    sw  = pd.to_numeric(last.get("SetWinner",  np.nan), errors="coerce")

    # If running totals are NaN (MatchBeats format), compute from GameWinner
    if np.isnan(p1g) or np.isnan(p2g):
        p1g, p2g, sw = get_s1_score_from_games(s1)

    # If SetWinner still NaN, infer from game counts
    if np.isnan(sw) and not (np.isnan(p1g) or np.isnan(p2g)):
        if p1g > p2g:
            sw = 1
        elif p2g > p1g:
            sw = 2

    d["s1_games_p1"] = to_int(p1g)
    d["s1_games_p2"] = to_int(p2g)
    d["s1_winner"]   = to_int(sw)

    if d["s1_games_p1"] is np.nan or d["s1_games_p2"] is np.nan:
        return {}

    total_pts = len(s1)
    d["s1_total_points"] = total_pts

    # ── Points won ────────────────────────────────────────────────────────────
    pw = pd.to_numeric(s1.get("PointWinner", pd.Series(dtype=float)), errors="coerce")
    d["s1_pts_won_p1"] = int((pw == 1).sum())
    d["s1_pts_won_p2"] = int((pw == 2).sum())

    # ── Server splits ─────────────────────────────────────────────────────────
    ps = pd.to_numeric(s1.get("PointServer", pd.Series(dtype=float)), errors="coerce")
    p1_srv_mask = ps == 1
    p2_srv_mask = ps == 2
    p1_srv = s1[p1_srv_mask]
    p2_srv = s1[p2_srv_mask]

    def serve_stats(srv_rows, p_num):
        n = len(srv_rows)
        if n == 0:
            return {}
        prefix = f"s1_p{p_num}"
        other  = 2 if p_num == 1 else 1

        # --- First serve % ---
        # Primary: P{p}FirstSrvIn column
        # Fallback: ServeNumber == 1 means first serve went in
        fsi_col = f"P{p_num}FirstSrvIn"
        if fsi_col in srv_rows.columns:
            fsi_vals = pd.to_numeric(srv_rows[fsi_col], errors="coerce")
            fsi = fsi_vals.sum() if fsi_vals.notna().any() else np.nan
        else:
            fsi = np.nan

        if np.isnan(fsi) and "ServeNumber" in srv_rows.columns:
            sn = pd.to_numeric(srv_rows["ServeNumber"], errors="coerce")
            fsi = (sn == 1).sum()

        # --- First serve points won ---
        fsw_col = f"P{p_num}FirstSrvWon"
        if fsw_col in srv_rows.columns:
            fsw_vals = pd.to_numeric(srv_rows[fsw_col], errors="coerce")
            fsw = fsw_vals.sum() if fsw_vals.notna().any() else np.nan
        else:
            fsw = np.nan

        if np.isnan(fsw) and "ServeNumber" in srv_rows.columns:
            sn = pd.to_numeric(srv_rows["ServeNumber"], errors="coerce")
            pw_srv = pd.to_numeric(srv_rows.get("PointWinner", pd.Series(dtype=float)), errors="coerce")
            fsw = ((sn == 1) & (pw_srv == p_num)).sum()

        # --- Second serve ---
        ssi_col = f"P{p_num}SecondSrvIn"
        ssw_col = f"P{p_num}SecondSrvWon"
        if ssi_col in srv_rows.columns:
            ssi_vals = pd.to_numeric(srv_rows[ssi_col], errors="coerce")
            ssi = ssi_vals.sum() if ssi_vals.notna().any() else np.nan
        else:
            ssi = np.nan

        if np.isnan(ssi) and "ServeNumber" in srv_rows.columns:
            sn = pd.to_numeric(srv_rows["ServeNumber"], errors="coerce")
            ssi = (sn == 2).sum()

        if ssw_col in srv_rows.columns:
            ssw_vals = pd.to_numeric(srv_rows[ssw_col], errors="coerce")
            ssw = ssw_vals.sum() if ssw_vals.notna().any() else np.nan
        else:
            ssw = np.nan

        if np.isnan(ssw) and "ServeNumber" in srv_rows.columns:
            sn = pd.to_numeric(srv_rows["ServeNumber"], errors="coerce")
            pw_srv = pd.to_numeric(srv_rows.get("PointWinner", pd.Series(dtype=float)), errors="coerce")
            ssw = ((sn == 2) & (pw_srv == p_num)).sum()

        # --- Break points faced (opponent has break point opp) ---
        bp_col  = f"P{other}BreakPoint"
        bpw_col = f"P{other}BreakPointWon"
        if bp_col in srv_rows.columns:
            bp_vals = pd.to_numeric(srv_rows[bp_col], errors="coerce")
            bp_opps = bp_vals.sum() if bp_vals.notna().any() else np.nan
        else:
            bp_opps = np.nan

        if bpw_col in srv_rows.columns:
            bpw_vals = pd.to_numeric(srv_rows[bpw_col], errors="coerce")
            bp_won = bpw_vals.sum() if bpw_vals.notna().any() else np.nan
        else:
            bp_won = np.nan

        out = {
            f"{prefix}_srv_pts":            n,
            f"{prefix}_first_srv_pct":      safe_pct(fsi, n),
            f"{prefix}_first_srv_won_pct":  safe_pct(fsw, fsi),
            f"{prefix}_second_srv_won_pct": safe_pct(ssw, ssi),
            f"{prefix}_bp_faced":           to_int(bp_opps) if not (isinstance(bp_opps, float) and np.isnan(bp_opps)) else np.nan,
            f"{prefix}_bp_saved_pct":       safe_pct(bp_opps - bp_won, bp_opps) if (
                                                not (isinstance(bp_opps, float) and np.isnan(bp_opps)) and
                                                not (isinstance(bp_won,  float) and np.isnan(bp_won))
                                            ) else np.nan,
        }
        return out

    d.update(serve_stats(p1_srv, 1))
    d.update(serve_stats(p2_srv, 2))

    # ── Return points won ─────────────────────────────────────────────────────
    if len(p2_srv) > 0:
        d["s1_p1_return_pts_won_pct"] = safe_pct((pw[p2_srv_mask] == 1).sum(), len(p2_srv))
    if len(p1_srv) > 0:
        d["s1_p2_return_pts_won_pct"] = safe_pct((pw[p1_srv_mask] == 2).sum(), len(p1_srv))

    # ── Aces, DFs, winners, UEs, forced errors ────────────────────────────────
    for stat, (c1, c2) in [
        ("aces",  ("P1Ace",         "P2Ace")),
        ("df",    ("P1DoubleFault", "P2DoubleFault")),
        ("win",   ("P1Winner",      "P2Winner")),
        ("ue",    ("P1UnfErr",      "P2UnfErr")),
        ("fe",    ("P1ForcedError", "P2ForcedError")),
    ]:
        for i, col in ((1, c1), (2, c2)):
            if col in s1.columns:
                vals = pd.to_numeric(s1[col], errors="coerce")
                d[f"s1_p{i}_{stat}"] = int(vals.sum()) if vals.notna().any() else np.nan
            else:
                d[f"s1_p{i}_{stat}"] = np.nan

    # ── Rally length ──────────────────────────────────────────────────────────
    for rally_col in ("RallyCount", "Rally"):
        if rally_col in s1.columns:
            rv = pd.to_numeric(s1[rally_col], errors="coerce")
            rv = rv[rv > 0]
            d["s1_avg_rally"] = round(rv.mean(), 2) if len(rv) else np.nan
            break

    return d

# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
all_features = []
all_meta     = []

print("=" * 65)
print("Building first-set features from point-by-point data …")
print("=" * 65)

for year, slam in TOURNAMENTS:
    stem       = f"{year}-{slam}"
    pts_path   = os.path.join(SLAM_DIR, f"{stem}-points.csv")
    match_path = os.path.join(SLAM_DIR, f"{stem}-matches.csv")

    if not os.path.exists(pts_path) or not os.path.exists(match_path):
        print(f"  SKIP  {stem}  (files not found)")
        continue

    try:
        pts  = pd.read_csv(pts_path,   low_memory=False)
        meta = pd.read_csv(match_path, low_memory=False)
    except Exception as e:
        print(f"  ERROR {stem}: {e}")
        continue

    if "SetNo" not in pts.columns:
        print(f"  SKIP  {stem}  (no SetNo column)")
        continue

    # Numeric coercions for key columns
    pts["SetNo"]      = pd.to_numeric(pts["SetNo"],      errors="coerce")
    pts["SetWinner"]  = pd.to_numeric(pts["SetWinner"],  errors="coerce")
    pts["PointWinner"]= pd.to_numeric(pts["PointWinner"],errors="coerce")
    pts["PointServer"]= pd.to_numeric(pts["PointServer"],errors="coerce")
    if "GameNo"     in pts.columns: pts["GameNo"]     = pd.to_numeric(pts["GameNo"],     errors="coerce")
    if "GameWinner" in pts.columns: pts["GameWinner"] = pd.to_numeric(pts["GameWinner"], errors="coerce")

    # Drop warmup/pre-match rows (PointNumber like "0X", "0Y")
    if "PointNumber" in pts.columns:
        pn = pts["PointNumber"].astype(str).str.strip()
        pts = pts[pn.str.match(r"^\d+(\.\d+)?$")]

    n_matches = 0
    for mid, match_pts in pts.groupby("match_id"):
        # ── Match winner from set results ──────────────────────────────────────
        match_winner = get_match_winner(match_pts)
        if np.isnan(match_winner):
            continue

        # ── First-set features ─────────────────────────────────────────────────
        s1 = match_pts[match_pts["SetNo"] == 1].copy()
        if len(s1) < 8:
            continue

        feats = agg_match_s1(s1)
        if not feats:
            continue

        feats["match_id"] = mid
        feats["winner"]   = int(match_winner)
        all_features.append(feats)
        n_matches += 1

    # Match metadata (player names, round)
    meta["year"] = year
    meta["slam"] = slam
    all_meta.append(meta)
    print(f"  OK    {stem}  ({n_matches} matches)")

# ── Combine ────────────────────────────────────────────────────────────────────
features_df = pd.DataFrame(all_features)
meta_df     = pd.concat(all_meta, ignore_index=True)

keep = [c for c in ["match_id","year","slam","player1","player2","round","status"]
        if c in meta_df.columns]
meta_df = meta_df[keep].drop_duplicates(subset=["match_id"])

combined = meta_df.merge(features_df, on="match_id", how="inner")
combined = combined.dropna(subset=["winner", "s1_winner"])
combined["winner"]    = combined["winner"].astype(int)
combined["s1_winner"] = pd.to_numeric(combined["s1_winner"], errors="coerce").dropna()
combined = combined[combined["s1_winner"].isin([1, 2])]
combined["s1_winner"] = combined["s1_winner"].astype(int)

# Save
out_path = os.path.join(INTERIM_DIR, "firstset_features.csv")
combined.to_csv(out_path, index=False)

s1_correct = (combined["s1_winner"] == combined["winner"]).mean()
print(f"\nMatches processed:          {len(combined):,}")
print(f"Set 1 winner = match winner: {s1_correct*100:.1f}%  (expected ~75-80%)")
print(f"Columns:                    {combined.shape[1]}")
print(f"Saved → {out_path}")

print(f"\nFirst-set feature columns (null rates):")
feat_cols = [c for c in combined.columns if c.startswith("s1_")]
for c in feat_cols:
    null_pct = combined[c].isnull().mean() * 100
    print(f"  {c:<38} {null_pct:.1f}%")
