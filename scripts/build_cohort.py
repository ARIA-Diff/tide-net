import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidenet.data.cohort import build_cohort
from tidenet.utils import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config("data", overrides=args.override)
    inst, store, meta, flow = build_cohort(cfg)
    print(flow)
