import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidenet.semisynthetic import run_semisynthetic
from tidenet.utils import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config("data", "train", "evaluation", overrides=args.override)
    scfg = load_config("semisynthetic")
    seeds = args.seeds if args.seeds is not None else cfg["experiment"]["seeds"]
    res, res2 = run_semisynthetic(cfg, scfg, seeds)
    print(res.groupby(["model", "train_alpha", "test_alpha"])[["auroc", "cal_slope"]].mean().to_string())
    print(res2.groupby(["model", "variant", "rate_multiplier"])[["auroc"]].mean().to_string())
