"""
run.py — FIXED EXPERIMENT RUNNER. DO NOT MODIFY.

Usage:
    python src/run.py "baseline"
    python src/run.py "add surface interaction terms"

The description string is logged to results.tsv.

What this script does:
  1. Imports build_model() and FEATURES from model.py
  2. Calls evaluate.evaluate(model, features) → val metrics
  3. Prints a summary block
  4. Appends one row to results.tsv
  5. Exits with code 0 (success) or 1 (crash)

The agent reads val_brier from stdout, compares to best,
and decides keep/discard/revert.
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime

RESULTS_TSV = Path(__file__).parent.parent / "results.tsv"
SRC_DIR     = Path(__file__).parent


def load_model_module():
    spec = importlib.util.spec_from_file_location("model", SRC_DIR / "model.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_git_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'nohash'


def append_results(commit, val_brier, val_acc, val_auc, status, description):
    header = "commit\tval_brier\tval_accuracy\tval_auc\tstatus\tdescription\n"
    row    = f"{commit}\t{val_brier:.6f}\t{val_acc:.6f}\t{val_auc:.6f}\t{status}\t{description}\n"
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, 'w') as f:
            f.write(header)
    with open(RESULTS_TSV, 'a') as f:
        f.write(row)


def main():
    description = sys.argv[1] if len(sys.argv) > 1 else "unnamed experiment"

    # ── Import model + features
    try:
        mod     = load_model_module()
        model   = mod.build_model()
        features = mod.FEATURES
    except Exception as e:
        print(f"CRASH: failed to import model.py — {e}")
        append_results(get_git_hash(), 0.0, 0.0, 0.0, 'crash', description)
        sys.exit(1)

    # ── Run evaluation
    try:
        from evaluate import evaluate
        results = evaluate(model, features)
    except Exception as e:
        print(f"CRASH: evaluation failed — {e}")
        append_results(get_git_hash(), 0.0, 0.0, 0.0, 'crash', description)
        sys.exit(1)

    # ── Print summary (agent reads this)
    commit = get_git_hash()
    print("---")
    print(f"val_brier:    {results['val_brier']:.6f}")
    print(f"val_accuracy: {results['val_accuracy']:.6f}")
    print(f"val_auc:      {results['val_auc']:.6f}")
    print(f"n_val:        {results['n_val']}")
    print(f"n_features:   {len(features)}")
    print(f"commit:       {commit}")
    print(f"description:  {description}")
    print("---")

    # ── Log (status filled in by agent after keep/discard decision)
    append_results(commit, results['val_brier'], results['val_accuracy'],
                   results['val_auc'], 'pending', description)


if __name__ == '__main__':
    main()
