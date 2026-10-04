import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tidenet.experiment import aggregate_runs, compare_auroc
from tidenet.utils import resolve, save_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/runs/random_slope5")
    ap.add_argument("--reference", default="tidenet")
    ap.add_argument("--against", nargs="*", default=[])
    args = ap.parse_args()
    root = resolve(args.root)
    summary = aggregate_runs(root)
    print(summary[["model", "n_seeds", "auroc", "auprc", "sens_at_90", "brier", "ece", "cal_slope", "slope_mae"]
                  if "slope_mae" in summary else ["model", "auroc", "auprc"]].to_string())
    tests = [compare_auroc(root, args.reference, m) for m in args.against]
    save_json(tests, root / "delong_tests.json")
    print(pd.DataFrame(tests).to_string() if tests else "")
