"""
Tennis Match Prediction Project
Step 1: Download data from two sources.

  Source A — JeffSackmann/tennis_slam_pointbypoint
    Point-by-point data for Grand Slam singles matches (2011-present).
    Two files per tournament:
      *-matches.csv  — match metadata (player names, winner, round, etc.)
      *-points.csv   — one row per point (aces, DFs, serve %, winners, etc.)

  Source B — JeffSackmann/tennis_atp (already downloaded in earlier run)
    Match-level ATP data used for pre-match features (ranking, age, height).
    We only need Grand Slam rows from this dataset.
"""

import io
import os
import sys
import time
import requests
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Directories ────────────────────────────────────────────────────────────────
SLAM_DIR = os.path.join("data", "raw", "slam_pbp")
ATP_DIR  = os.path.join("data", "raw")
os.makedirs(SLAM_DIR, exist_ok=True)

BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/"

# All known singles points + matches files (from repo contents listing)
TOURNAMENTS = [
    ("2011", "ausopen"), ("2011", "frenchopen"), ("2011", "usopen"), ("2011", "wimbledon"),
    ("2012", "ausopen"), ("2012", "frenchopen"), ("2012", "usopen"), ("2012", "wimbledon"),
    ("2013", "ausopen"), ("2013", "frenchopen"), ("2013", "usopen"), ("2013", "wimbledon"),
    ("2014", "ausopen"), ("2014", "frenchopen"), ("2014", "usopen"), ("2014", "wimbledon"),
    ("2015", "ausopen"), ("2015", "frenchopen"), ("2015", "usopen"), ("2015", "wimbledon"),
    ("2016", "ausopen"), ("2016", "frenchopen"), ("2016", "usopen"), ("2016", "wimbledon"),
    ("2017", "ausopen"), ("2017", "frenchopen"), ("2017", "usopen"), ("2017", "wimbledon"),
    ("2018", "ausopen"), ("2018", "frenchopen"), ("2018", "usopen"), ("2018", "wimbledon"),
    ("2019", "ausopen"), ("2019", "frenchopen"), ("2019", "usopen"), ("2019", "wimbledon"),
    ("2020", "ausopen"), ("2020", "frenchopen"), ("2020", "usopen"),  # no wimbledon 2020
    ("2021", "ausopen"), ("2021", "frenchopen"), ("2021", "usopen"), ("2021", "wimbledon"),
    ("2022", "usopen"), ("2022", "wimbledon"),
    ("2023", "usopen"), ("2023", "wimbledon"),
    ("2024", "usopen"), ("2024", "wimbledon"),
]

def download(url, local_path, label):
    if os.path.exists(local_path):
        print(f"  cached  {label}")
        return True
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        size_kb = len(r.content) / 1024
        print(f"  OK      {label}  ({size_kb:.0f} KB)")
        time.sleep(0.2)
        return True
    except Exception as e:
        print(f"  FAIL    {label}  — {e}")
        return False

# ── Download slam_pointbypoint files ──────────────────────────────────────────
print("=" * 65)
print("Downloading slam_pointbypoint data …")
print("=" * 65)

ok, fail = 0, 0
for year, slam in TOURNAMENTS:
    stem = f"{year}-{slam}"
    for ftype in ("matches", "points"):
        fname   = f"{stem}-{ftype}.csv"
        url     = BASE + fname
        local   = os.path.join(SLAM_DIR, fname)
        success = download(url, local, fname)
        if success: ok += 1
        else:       fail += 1

print(f"\nDownloaded: {ok}  |  Failed: {fail}")

# ── Source B: ATP data — already in data/raw from previous run ─────────────────
atp_path = os.path.join(ATP_DIR, "atp_matches_2000_2023.csv")
if os.path.exists(atp_path):
    atp = pd.read_csv(atp_path, low_memory=False)
    slams = atp[atp["tourney_level"] == "G"]
    out   = os.path.join(ATP_DIR, "atp_grand_slams_2000_2023.csv")
    slams.to_csv(out, index=False)
    print(f"\nATP Grand Slam rows: {len(slams):,}  →  saved to {out}")
else:
    print(f"\nWARNING: {atp_path} not found — run the old 01_download_data.py first, or "
          "re-download from JeffSackmann/tennis_atp.")

print("\nDone.")
