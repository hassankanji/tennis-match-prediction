"""
Tennis Match Prediction Project
Step 2: Data profiling — understand the shape, quality, and patterns
        of the raw ATP dataset before modelling.
"""

import io
import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Windows terminal may default to cp1252; force UTF-8 so box-drawing chars print
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

RAW_PATH = os.path.join("data", "raw", "atp_matches_2000_2023.csv")
PLOTS_DIR = os.path.join("data", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TENNIS ATP DATASET — DATA PROFILE")
print("=" * 70)

df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"\nShape: {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

# ── Column inventory ───────────────────────────────────────────────────────────
COLUMN_GROUPS = {
    "Match Context": [
        "tourney_id", "tourney_name", "surface", "draw_size",
        "tourney_level", "tourney_date", "match_num",
        "best_of", "round", "minutes", "score",
    ],
    "Winner Info": [
        "winner_id", "winner_seed", "winner_entry",
        "winner_name", "winner_hand", "winner_ht",
        "winner_ioc", "winner_age", "winner_rank", "winner_rank_points",
    ],
    "Loser Info": [
        "loser_id", "loser_seed", "loser_entry",
        "loser_name", "loser_hand", "loser_ht",
        "loser_ioc", "loser_age", "loser_rank", "loser_rank_points",
    ],
    "Winner Match Stats": [
        "w_ace", "w_df", "w_svpt",
        "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
    ],
    "Loser Match Stats": [
        "l_ace", "l_df", "l_svpt",
        "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
    ],
}

print("\n── Column Groups ──────────────────────────────────────────────────────")
for group, cols in COLUMN_GROUPS.items():
    present = [c for c in cols if c in df.columns]
    print(f"  {group}: {present}")

# ── Null / completeness ────────────────────────────────────────────────────────
print("\n── Null Rates ─────────────────────────────────────────────────────────")
null_rates = (df.isnull().mean() * 100).sort_values(ascending=False)
high_null = null_rates[null_rates > 0]
if high_null.empty:
    print("  No missing values.")
else:
    for col, rate in high_null.items():
        flag = "🔴" if rate > 20 else ("🟡" if rate > 5 else "✅")
        print(f"  {flag}  {col:<30} {rate:6.1f}% missing")

# ── Numeric summary ────────────────────────────────────────────────────────────
stat_cols = [c for c in df.columns if df[c].dtype in [float, int, "float64", "int64"]]
print(f"\n── Numeric Summary ({len(stat_cols)} columns) ─────────────────────────────")
print(df[stat_cols].describe(percentiles=[.05, .25, .50, .75, .95]).round(2).to_string())

# ── Categorical profiles ───────────────────────────────────────────────────────
cat_cols = ["surface", "tourney_level", "round", "best_of", "winner_hand", "loser_hand"]
print("\n── Categorical Value Counts ───────────────────────────────────────────")
for col in cat_cols:
    if col not in df.columns:
        continue
    vc = df[col].value_counts(dropna=False)
    print(f"\n  {col}:")
    for val, cnt in vc.items():
        print(f"    {str(val):<25} {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

# ── Date range ─────────────────────────────────────────────────────────────────
if "tourney_date" in df.columns:
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    print(f"\n── Date Range ─────────────────────────────────────────────────────────")
    print(f"  Earliest: {df['tourney_date'].min().date()}")
    print(f"  Latest:   {df['tourney_date'].max().date()}")
    matches_per_year = df.groupby(df["tourney_date"].dt.year).size()
    print(f"\n  Matches per year (sample):")
    print(matches_per_year.to_string())

# ── First-set score extraction preview ────────────────────────────────────────
def parse_first_set(score_str):
    """Return (w_games, l_games) from first set, or (NaN, NaN) on failure."""
    if not isinstance(score_str, str):
        return np.nan, np.nan
    # Strip retirement/walkover suffixes
    score_str = score_str.replace("RET", "").replace("W/O", "").replace("DEF.", "").strip()
    sets = score_str.split()
    if not sets:
        return np.nan, np.nan
    first = sets[0]
    # Handle tiebreak notation like "7-6(5)"
    first = first.split("(")[0]
    parts = first.split("-")
    if len(parts) != 2:
        return np.nan, np.nan
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return np.nan, np.nan

if "score" in df.columns:
    parsed = df["score"].apply(parse_first_set)
    df["s1_w_games"] = parsed.apply(lambda x: x[0])
    df["s1_l_games"] = parsed.apply(lambda x: x[1])
    df["s1_winner_won"] = (df["s1_w_games"] > df["s1_l_games"]).astype(float)

    print(f"\n── First Set Analysis ─────────────────────────────────────────────────")
    valid = df["s1_w_games"].notna()
    print(f"  Rows with parseable first set: {valid.sum():,} / {len(df):,} ({valid.mean()*100:.1f}%)")
    print(f"  First-set winner = match winner: {df.loc[valid,'s1_winner_won'].mean()*100:.1f}%")

    s1_margins = (df.loc[valid, "s1_w_games"] - df.loc[valid, "s1_l_games"]).abs()
    print(f"  Avg first-set game margin: {s1_margins.mean():.2f}")

    print("\n  Top 10 first-set score combinations:")
    combo = df.loc[valid].apply(
        lambda r: f"{int(r.s1_w_games)}-{int(r.s1_l_games)}", axis=1
    ).value_counts().head(10)
    for score, cnt in combo.items():
        print(f"    {score:<8} {cnt:>7,}  ({cnt/valid.sum()*100:.1f}%)")

# ── Rank differential ─────────────────────────────────────────────────────────
if {"winner_rank", "loser_rank"}.issubset(df.columns):
    df["rank_diff"] = df["loser_rank"] - df["winner_rank"]   # positive = winner was higher-ranked
    upset_mask = df["rank_diff"] < 0                          # lower-ranked player won
    print(f"\n── Upset Rate (lower-ranked beat higher-ranked) ───────────────────────")
    print(f"  {upset_mask.sum():,} upsets out of {df['rank_diff'].notna().sum():,} ranked matches")
    print(f"  Upset rate: {upset_mask.mean()*100:.1f}%")

# ── Plots ──────────────────────────────────────────────────────────────────────
print("\nGenerating plots ...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("ATP Match Data — Exploratory Profile (2000–2023)", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Matches per year
ax1 = fig.add_subplot(gs[0, :2])
if "tourney_date" in df.columns:
    matches_per_year.plot(kind="bar", ax=ax1, color="steelblue", edgecolor="white")
    ax1.set_title("Matches per Year")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Matches")
    ax1.tick_params(axis="x", rotation=45)

# 2. Surface distribution
ax2 = fig.add_subplot(gs[0, 2])
if "surface" in df.columns:
    surface_counts = df["surface"].value_counts(dropna=False)
    ax2.pie(surface_counts, labels=surface_counts.index, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Surface Distribution")

# 3. Null rates bar chart
ax3 = fig.add_subplot(gs[1, :])
cols_with_nulls = null_rates[null_rates > 0]
if not cols_with_nulls.empty:
    colors = ["#d73027" if v > 20 else "#fc8d59" if v > 5 else "#91cf60"
              for v in cols_with_nulls]
    cols_with_nulls.plot(kind="bar", ax=ax3, color=colors, edgecolor="white")
    ax3.set_title("Missing Data Rate by Column")
    ax3.set_ylabel("% Missing")
    ax3.axhline(5,  color="orange", ls="--", lw=1, label="5% threshold")
    ax3.axhline(20, color="red",    ls="--", lw=1, label="20% threshold")
    ax3.legend(fontsize=8)
    ax3.tick_params(axis="x", rotation=60)

# 4. First-set margin distribution
ax4 = fig.add_subplot(gs[2, 0])
if "s1_w_games" in df.columns:
    s1_margins.plot(kind="hist", bins=range(0, 8), ax=ax4,
                    color="teal", edgecolor="white", rwidth=0.8)
    ax4.set_title("First-Set Game Margin")
    ax4.set_xlabel("Game diff"); ax4.set_ylabel("Matches")

# 5. Rank differential histogram
ax5 = fig.add_subplot(gs[2, 1])
if "rank_diff" in df.columns:
    df["rank_diff"].clip(-200, 200).plot(kind="hist", bins=50, ax=ax5,
                                          color="salmon", edgecolor="white")
    ax5.axvline(0, color="black", lw=1.5)
    ax5.set_title("Rank Diff (Loser − Winner)\n>0 = higher-ranked won")
    ax5.set_xlabel("Rank diff")

# 6. Match duration
ax6 = fig.add_subplot(gs[2, 2])
if "minutes" in df.columns:
    df["minutes"].dropna().clip(0, 300).plot(kind="hist", bins=40, ax=ax6,
                                              color="mediumpurple", edgecolor="white")
    ax6.set_title("Match Duration (minutes)")
    ax6.set_xlabel("Minutes")

plot_path = os.path.join(PLOTS_DIR, "data_profile.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {plot_path}")

print("\n" + "=" * 70)
print("Profile complete.")
print("=" * 70)
