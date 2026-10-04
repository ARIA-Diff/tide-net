import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidenet.search import search_model
from tidenet.utils import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tidenet", "lightgbm", "grud", "tlstm", "mtan", "odernn", "strats",
                                                    "lstm_monthly"])
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config("data", "train", "evaluation", overrides=args.override)
    scfg = load_config("search")
    for m in args.models:
        best = search_model(cfg, scfg, m)
        print(m, best)
