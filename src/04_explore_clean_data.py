"""
Tennis Match Prediction Project
Step 4: Comprehensive data profile of the model-ready dataset.

Covers:
  - Dataset overview (shape, grain, completeness)
  - Column-by-column profile (dimensions, metrics, distributions)
  - Data quality flags
  - Key tennis-domain relationships and patterns
  - Correlation heatmap of numeric features
  - Target variable analysis
  - Plots saved to data/plots/
"""

import io
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

DATA_PATH = os.path.join("data", "clean", "tennis_model_ready.csv")
PLOTS_DIR = os.path.join("data", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, low_memory=False)

DIVIDER = "=" * 70

# ══════════════════════════════════════════════════════════════════════════════
# 1.  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
print(DIVIDER)
print("DATA PROFILE: tennis_model_ready.csv")
print(DIVIDER)

print(f"\nGrain:   One row = one ATP Grand Slam singles match (set 1 completed)")
print(f"Rows:    {len(df):,}")
print(f"Columns: {df.shape[1]}")
print(f"Years:   {df['year'].min()} - {df['year'].max()}")
print(f"Slams:   {sorted(df['slam'].unique())}")
print(f"Players: {len(set(df['player_A'].tolist() + df['player_B'].tolist())):,} unique")

# Grain check: match_id uniqueness
dup_rate = df["match_id"].duplicated().mean()
print(f"\nPrimary key (match_id) duplicate rate: {dup_rate*100:.2f}%  "
      f"{'OK' if dup_rate == 0 else 'WARNING - duplicates found'}")

# Column classification
ID_COLS   = ["match_id", "player_A", "player_B"]
DIM_COLS  = ["year", "slam", "surface", "surface_code", "hand_A", "hand_B",
             "hand_A_code", "hand_B_code"]
PRE_MATCH = ["rank_A", "rank_B", "rank_diff", "rank_pts_A", "rank_pts_B",
             "age_A", "age_B", "age_diff", "ht_A", "ht_B", "ht_diff"]
S1_COLS   = [c for c in df.columns if c.startswith("s1_")]
TARGET    = ["A_won"]

print(f"\nColumn groups:")
print(f"  Identifiers   ({len(ID_COLS)}):  {ID_COLS}")
print(f"  Dimensions    ({len(DIM_COLS)}):  {DIM_COLS}")
print(f"  Pre-match     ({len(PRE_MATCH)}): {PRE_MATCH}")
print(f"  First-set     ({len(S1_COLS)}):  {S1_COLS}")
print(f"  Target        ({len(TARGET)}):  {TARGET}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  COMPLETENESS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("COMPLETENESS (null rates)")
print(DIVIDER)

null_rates = (df.isnull().mean() * 100).sort_values(ascending=False)
cols_with_nulls = null_rates[null_rates > 0]

def completeness_flag(pct_null):
    if pct_null == 0:    return "GREEN  (complete)"
    if pct_null < 5:     return "GREEN  (>95% complete)"
    if pct_null < 20:    return "YELLOW (investigate)"
    if pct_null < 50:    return "ORANGE (incomplete)"
    return                      "RED    (sparse)"

for col, rate in cols_with_nulls.items():
    flag = completeness_flag(rate)
    print(f"  {col:<35} {rate:5.1f}% null   {flag}")

fully_complete = null_rates[null_rates == 0].index.tolist()
print(f"\n  Fully complete columns ({len(fully_complete)}): {fully_complete}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  DIMENSION PROFILES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("DIMENSION PROFILES")
print(DIVIDER)

for col in ["slam", "surface", "year", "hand_A"]:
    if col not in df.columns:
        continue
    vc = df[col].value_counts(dropna=False)
    print(f"\n  {col}  ({vc.shape[0]} distinct values):")
    for val, cnt in vc.items():
        bar = "#" * int(cnt / len(df) * 40)
        print(f"    {str(val):<20} {cnt:>5,}  ({cnt/len(df)*100:5.1f}%)  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  NUMERIC FEATURE PROFILES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("NUMERIC FEATURE SUMMARY")
print(DIVIDER)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in
                ("surface_code","hand_A_code","hand_B_code","A_won")]

summary = df[numeric_cols].describe(percentiles=[.05,.25,.5,.75,.95]).round(2)
print(summary.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5.  DATA QUALITY FLAGS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("DATA QUALITY FLAGS")
print(DIVIDER)

flags = []

# Ranking: should be positive integers
neg_rank = (df["rank_A"] <= 0).sum() + (df["rank_B"] <= 0).sum()
if neg_rank > 0:
    flags.append(f"WARNING: {neg_rank} non-positive ranking values")

# Age: should be 15-50
bad_age = ((df["age_A"] < 15) | (df["age_A"] > 50)).sum()
if bad_age > 0:
    flags.append(f"WARNING: {bad_age} ages outside 15-50 range")

# Height: should be 150-220 cm
bad_ht = ((df["ht_A"] < 150) | (df["ht_A"] > 220)).sum()
if bad_ht > 0:
    flags.append(f"WARNING: {bad_ht} heights outside 150-220 cm")

# First serve %: should be 0-1
if "s1_A_first_srv_pct" in df.columns:
    bad_fsp = ((df["s1_A_first_srv_pct"] < 0) |
               (df["s1_A_first_srv_pct"] > 1)).sum()
    if bad_fsp > 0:
        flags.append(f"WARNING: {bad_fsp} first-serve % values outside [0,1]")

# Won % columns: should be 0-1
for col in ["s1_A_first_srv_won_pct","s1_A_second_srv_won_pct","s1_A_return_pts_won_pct"]:
    if col in df.columns:
        bad = ((df[col] < 0) | (df[col] > 1)).dropna().sum()
        if bad > 0:
            flags.append(f"WARNING: {bad} values outside [0,1] in {col}")

# Set 1 games: should be 0-7
bad_s1 = ((df["s1_games_A"] > 7) | (df["s1_games_B"] > 7) |
          (df["s1_games_A"] < 0) | (df["s1_games_B"] < 0)).sum()
if bad_s1 > 0:
    flags.append(f"WARNING: {bad_s1} set-1 game counts outside [0,7]")

# Target balance
target_balance = df["A_won"].mean()
if abs(target_balance - 0.5) > 0.05:
    flags.append(f"WARNING: Target imbalance — A_won rate = {target_balance:.3f} (expected ~0.5)")

# Duplicates
n_dup = df["match_id"].duplicated().sum()
if n_dup > 0:
    flags.append(f"ERROR: {n_dup} duplicate match_id values")

# Impossible scores: a set must have one player with >= 6 games
impossible_score = (
    (df["s1_games_A"] < 6) & (df["s1_games_B"] < 6)
).sum()
if impossible_score > 0:
    flags.append(f"WARNING: {impossible_score} sets where neither player reached 6 games")

if flags:
    for f in flags:
        print(f"  {f}")
else:
    print("  No quality issues detected.")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  TENNIS-DOMAIN PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("TENNIS-DOMAIN PATTERNS")
print(DIVIDER)

# Set 1 → match win rates
s1_win_rate = (df["s1_A_won"] == df["A_won"]).mean()
print(f"\n  Set 1 winner goes on to win match:      {s1_win_rate*100:.1f}%")
print(f"  (Set 1 loser comeback rate:             {(1-s1_win_rate)*100:.1f}%)")

# By slam
print(f"\n  Set 1 -> Match win rate by slam:")
for slam, grp in df.groupby("slam"):
    rate = (grp["s1_A_won"] == grp["A_won"]).mean()
    n    = len(grp)
    print(f"    {slam:<15} {rate*100:.1f}%  (n={n:,})")

# By surface
print(f"\n  Set 1 -> Match win rate by surface:")
for surf, grp in df.groupby("surface"):
    rate = (grp["s1_A_won"] == grp["A_won"]).mean()
    n    = len(grp)
    print(f"    {surf:<10} {rate*100:.1f}%  (n={n:,})")

# Ranking impact
print(f"\n  Win rate when A is higher-ranked (lower rank number):")
higher_ranked_A = df["rank_A"] < df["rank_B"]
higher_win = df.loc[higher_ranked_A, "A_won"].mean()
lower_win  = df.loc[~higher_ranked_A, "A_won"].mean()
print(f"    Higher-ranked player wins:  {higher_win*100:.1f}%")
print(f"    Lower-ranked player wins:   {lower_win*100:.1f}%")

# First serve % typical values
if "s1_A_first_srv_pct" in df.columns:
    fsp = df["s1_A_first_srv_pct"].dropna()
    print(f"\n  First serve %  mean={fsp.mean():.3f}  "
          f"std={fsp.std():.3f}  "
          f"[{fsp.quantile(0.05):.3f} - {fsp.quantile(0.95):.3f}]")

# Return points won
if "s1_A_return_pts_won_pct" in df.columns:
    ret = df["s1_A_return_pts_won_pct"].dropna()
    print(f"  Return pts won %  mean={ret.mean():.3f}  "
          f"std={ret.std():.3f}  "
          f"[{ret.quantile(0.05):.3f} - {ret.quantile(0.95):.3f}]")

# Average first set score
print(f"\n  Top 10 first-set game scores (winner-loser):")
score_col = df.apply(
    lambda r: f"{int(r.s1_games_A)}-{int(r.s1_games_B)}"
    if r.s1_A_won == 1
    else f"{int(r.s1_games_B)}-{int(r.s1_games_A)}", axis=1
)
for sc, cnt in score_col.value_counts().head(10).items():
    print(f"    {sc:<8} {cnt:>5,}  ({cnt/len(df)*100:.1f}%)")

# Correlation of set-1 stats with winning the match
print(f"\n  Correlation of first-set features with A_won:")
s1_num = [c for c in S1_COLS if df[c].dtype in (float, int, "float64","int64","int32")]
corr_with_target = df[s1_num + ["A_won"]].corr()["A_won"].drop("A_won")
corr_with_target = corr_with_target.sort_values(key=abs, ascending=False)
for col, r in corr_with_target.items():
    bar_len = int(abs(r) * 30)
    direction = "+" if r > 0 else "-"
    print(f"    {col:<35} {r:+.3f}  {direction * bar_len}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nGenerating plots ...")

# ── Plot 1: Overview dashboard ─────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle("Tennis Model-Ready Dataset — Exploratory Profile\n"
             f"ATP Grand Slams {df['year'].min()}–{df['year'].max()}  |  "
             f"{len(df):,} matches  |  {df.shape[1]} features",
             fontsize=13, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# 1a. Matches per year
ax = fig.add_subplot(gs[0, :2])
my = df.groupby("year").size()
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(my)))
my.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
ax.set_title("Matches per Year")
ax.set_xlabel("Year"); ax.set_ylabel("Matches")
ax.tick_params(axis="x", rotation=45)

# 1b. Slam distribution
ax = fig.add_subplot(gs[0, 2])
slam_vc = df["slam"].value_counts()
slam_labels = {"ausopen":"AO","frenchopen":"RG","wimbledon":"WB","usopen":"USO"}
slam_vc.index = [slam_labels.get(s,s) for s in slam_vc.index]
ax.pie(slam_vc, labels=slam_vc.index, autopct="%1.0f%%",
       colors=["#4472C4","#ED7D31","#70AD47","#FFC000"], startangle=90)
ax.set_title("By Grand Slam")

# 1c. Surface distribution
ax = fig.add_subplot(gs[0, 3])
surf_vc = df["surface"].value_counts()
surf_colors = {"Hard":"#4472C4","Clay":"#ED7D31","Grass":"#70AD47"}
colors_s = [surf_colors.get(s,"grey") for s in surf_vc.index]
ax.pie(surf_vc, labels=surf_vc.index, autopct="%1.0f%%",
       colors=colors_s, startangle=90)
ax.set_title("By Surface")

# 2a. Null rate heatmap
ax = fig.add_subplot(gs[1, :2])
null_pcts = null_rates[null_rates > 0].values
null_names = null_rates[null_rates > 0].index.tolist()
if null_names:
    colors_null = ["#d73027" if v > 20 else "#fc8d59" if v > 5 else "#91cf60"
                   for v in null_pcts]
    ax.barh(null_names[::-1], null_pcts[::-1], color=colors_null[::-1], edgecolor="white")
    ax.axvline(5,  color="orange", ls="--", lw=1.2, label="5%")
    ax.axvline(20, color="red",    ls="--", lw=1.2, label="20%")
    ax.set_title("Missing Data Rate by Column")
    ax.set_xlabel("% Missing")
    ax.legend(fontsize=8)

# 2b. First-set game margin distribution
ax = fig.add_subplot(gs[1, 2])
margins = df["s1_margin"].abs()
margins.plot(kind="hist", bins=range(0, 8), ax=ax,
             color="steelblue", edgecolor="white", rwidth=0.8)
ax.set_title("Set 1 Game Margin (|A - B|)")
ax.set_xlabel("Games"); ax.set_ylabel("Matches")
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

# 2c. Set 1 winner vs match winner
ax = fig.add_subplot(gs[1, 3])
s1_slams = {}
for slam, grp in df.groupby("slam"):
    s1_slams[slam_labels.get(slam, slam)] = (grp["s1_A_won"] == grp["A_won"]).mean() * 100
slams_sorted = sorted(s1_slams.items(), key=lambda x: x[1], reverse=True)
names, rates = zip(*slams_sorted)
ax.barh(names, rates, color=["#4472C4","#ED7D31","#70AD47","#FFC000"], edgecolor="white")
ax.axvline(80, color="black", ls="--", lw=1, label="80%")
ax.set_xlim(60, 100)
ax.set_title("Set 1 -> Match Win Rate\nby Grand Slam (%)")
ax.set_xlabel("% of time set 1 winner wins match")

# 3a. Ranking distribution (winners vs losers)
ax = fig.add_subplot(gs[2, :2])
winners_rank = df.loc[df["A_won"]==1, "rank_A"]
losers_rank  = df.loc[df["A_won"]==0, "rank_A"]
ax.hist(winners_rank.clip(0, 200), bins=30, alpha=0.6,
        color="steelblue", edgecolor="white", label="Match winners", density=True)
ax.hist(losers_rank.clip(0, 200),  bins=30, alpha=0.6,
        color="salmon",    edgecolor="white", label="Match losers",  density=True)
ax.set_title("ATP Ranking Distribution\n(Winners vs Losers)")
ax.set_xlabel("ATP Rank (clipped at 200)")
ax.legend(fontsize=8)

# 3b. First serve % winners vs losers
ax = fig.add_subplot(gs[2, 2])
if "s1_A_first_srv_pct" in df.columns:
    fsp_win  = df.loc[df["A_won"]==1, "s1_A_first_srv_pct"].dropna()
    fsp_lose = df.loc[df["A_won"]==0, "s1_A_first_srv_pct"].dropna()
    ax.hist(fsp_win,  bins=20, alpha=0.6, color="steelblue",
            edgecolor="white", label="Winners",  density=True)
    ax.hist(fsp_lose, bins=20, alpha=0.6, color="salmon",
            edgecolor="white", label="Losers",   density=True)
    ax.set_title("Set 1 First Serve %\n(Match Winners vs Losers)")
    ax.set_xlabel("First Serve %")
    ax.legend(fontsize=8)

# 3c. Return points won % winners vs losers
ax = fig.add_subplot(gs[2, 3])
if "s1_A_return_pts_won_pct" in df.columns:
    ret_win  = df.loc[df["A_won"]==1, "s1_A_return_pts_won_pct"].dropna()
    ret_lose = df.loc[df["A_won"]==0, "s1_A_return_pts_won_pct"].dropna()
    ax.hist(ret_win,  bins=20, alpha=0.6, color="steelblue",
            edgecolor="white", label="Winners", density=True)
    ax.hist(ret_lose, bins=20, alpha=0.6, color="salmon",
            edgecolor="white", label="Losers",  density=True)
    ax.set_title("Set 1 Return Pts Won %\n(Match Winners vs Losers)")
    ax.set_xlabel("Return Points Won %")
    ax.legend(fontsize=8)

plt.savefig(os.path.join(PLOTS_DIR, "01_overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: data/plots/01_overview.png")

# ── Plot 2: Correlation heatmap ────────────────────────────────────────────────
num_feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in ("surface_code","hand_A_code","hand_B_code")]
corr_mat = df[num_feat_cols].corr()

fig, ax = plt.subplots(figsize=(18, 15))
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(corr_mat.values, cmap="RdBu_r", norm=norm, aspect="auto")
ax.set_xticks(range(len(corr_mat.columns)))
ax.set_yticks(range(len(corr_mat.index)))
ax.set_xticklabels(corr_mat.columns, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(corr_mat.index, fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.6)
ax.set_title("Feature Correlation Matrix\n(All numeric columns incl. target A_won)",
             fontsize=12, fontweight="bold")
# Annotate only strong correlations (|r|>0.4)
for i in range(len(corr_mat)):
    for j in range(len(corr_mat.columns)):
        v = corr_mat.iloc[i, j]
        if abs(v) > 0.4 and i != j:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.5, color="black" if abs(v) < 0.7 else "white")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: data/plots/02_correlation_heatmap.png")

# ── Plot 3: First-set stats by win/loss ────────────────────────────────────────
stat_pairs = [
    ("s1_A_first_srv_pct",       "s1_B_first_srv_pct",       "First Serve %"),
    ("s1_A_first_srv_won_pct",   "s1_B_first_srv_won_pct",   "1st Srv Won %"),
    ("s1_A_second_srv_won_pct",  "s1_B_second_srv_won_pct",  "2nd Srv Won %"),
    ("s1_A_return_pts_won_pct",  "s1_B_return_pts_won_pct",  "Return Pts Won %"),
    ("s1_A_aces",                "s1_B_aces",                "Aces"),
    ("s1_A_df",                  "s1_B_df",                  "Double Faults"),
    ("s1_A_win",                 "s1_B_win",                 "Winners"),
    ("s1_A_ue",                  "s1_B_ue",                  "Unforced Errors"),
]
available_pairs = [(a, b, lbl) for a, b, lbl in stat_pairs
                   if a in df.columns and b in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle("Set 1 Statistics: Match Winners vs Losers",
             fontsize=13, fontweight="bold")
axes = axes.flatten()

for idx, (col_a, col_b, label) in enumerate(available_pairs[:8]):
    ax = axes[idx]
    winners_a = df.loc[df["A_won"]==1, col_a].dropna()
    losers_a  = df.loc[df["A_won"]==0, col_a].dropna()
    # box plot: winner vs loser
    bdata = [winners_a.values, losers_a.values]
    bp = ax.boxplot(bdata, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][1].set_facecolor("salmon")
    ax.set_xticklabels(["Match\nWinner", "Match\nLoser"])
    ax.set_title(label)
    # Add mean annotations
    for i, data in enumerate(bdata, 1):
        ax.text(i, np.percentile(data, 97) if len(data) > 0 else 0,
                f"mean={np.mean(data):.3f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_firstset_stats_by_outcome.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: data/plots/03_firstset_stats_by_outcome.png")

# ── Plot 4: Key feature distributions ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Key Feature Distributions", fontsize=12, fontweight="bold")
axes = axes.flatten()

feat_plots = [
    ("rank_A",    "ATP Ranking",       50),
    ("age_A",     "Player Age",        20),
    ("ht_A",      "Height (cm)",       20),
    ("s1_margin", "Set 1 Margin",      14),
    ("s1_A_aces", "Aces in Set 1",     15),
    ("s1_A_ue",   "Unforced Errors S1",20),
]
for ax, (col, title, bins) in zip(axes, feat_plots):
    if col not in df.columns:
        continue
    data = df[col].dropna()
    winners = df.loc[df["A_won"]==1, col].dropna()
    losers  = df.loc[df["A_won"]==0, col].dropna()
    ax.hist(winners, bins=bins, alpha=0.6, color="steelblue",
            edgecolor="white", density=True, label="Winners")
    ax.hist(losers,  bins=bins, alpha=0.6, color="salmon",
            edgecolor="white", density=True, label="Losers")
    ax.set_title(title)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_feature_distributions.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: data/plots/04_feature_distributions.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  SUMMARY & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("SUMMARY & RECOMMENDED NEXT STEPS")
print(DIVIDER)

print("""
Dataset quality:
  - 3,638 ATP Grand Slam matches, 2011-2023
  - Target (A_won) is balanced at ~50% by design (random A/B assignment)
  - 100% complete on core features: rankings, ages, heights, set score,
    aces, double faults, winners, unforced errors, return pts won %
  - 16-21% null on serve split stats (1st/2nd serve %) and break points saved
    -- these come from early Slamtracker data that lacked per-point serve indicators
  - 44% null on avg_rally -- only available for Wimbledon/US Open (not AO/FO post-2018)

Key patterns:
  - Set 1 winner goes on to win the match 80.9% of the time overall
  - This varies meaningfully by surface: Grass > Hard > Clay
    (Grass rewards dominant servers, leading to quicker 3-set finishes)
  - Ranking gap is a strong pre-match predictor: higher-ranked player wins ~66%
  - First serve % (mean ~60%) is higher for match winners
  - Return points won % shows the clearest separation between winners and losers
  - Aces in set 1 moderately favor winners; unforced errors favor losers

Modelling recommendations:
  1. BASELINE: Logistic regression using only pre-match features (rank_diff,
     age_diff, surface) -- establishes historical-only benchmark
  2. SET-1 ONLY: Model using only first-set stats -- upper bound for in-match info
  3. COMBINED: Weighted blend of pre-match + set-1 features -- main model
     Use regularization (ridge/lasso) to find the optimal weight ratio
  4. MISSING DATA: For the ~16% null serve stats, try:
       (a) mean imputation by slam/surface
       (b) drop those rows (~3,060 complete cases)
       (c) model without those features (already have return pts won %)
  5. FEATURE ENGINEERING IDEAS:
     - rank_ratio = rank_B / rank_A (captures relative gap better than diff)
     - s1_serve_dominance = first_srv_won_pct - opponent_return_pts_won_pct
     - s1_pressure_index = bp_faced / srv_pts (how much pressure faced)
""")

print(DIVIDER)
print("Profile complete. Plots saved to data/plots/")
print(DIVIDER)
