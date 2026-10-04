import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidenet.experiment import aggregate_runs, run_dir, run_experiment
from tidenet.utils import load_config

GROUPS = {
    "reference": ["kfre", "lmm", "logistic", "lightgbm", "lightgbm_novisit"],
    "deep": ["lstm_monthly", "grud", "tlstm", "mtan", "odernn", "strats"],
    "deep_sl": ["grud_sl", "tlstm_sl", "strats_sl"],
    "tidenet": ["tidenet"],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--ablations", nargs="*", default=[])
    ap.add_argument("--outcome", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--death-positive", action="store_true")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config("data", "train", "evaluation", overrides=args.override)
    if args.outcome:
        cfg["experiment"]["outcome"] = args.outcome
    if args.split:
        cfg["experiment"]["split"] = args.split
    if args.death_positive:
        cfg["experiment"]["include_death_positive"] = True
    seeds = args.seeds if args.seeds is not None else cfg["experiment"]["seeds"]

    jobs = []
    for m in args.models:
        jobs += [(n, None, n) for n in GROUPS.get(m, [m])]
    if args.ablations:
        abl_cfg = load_config("ablations")
        jobs += [("tidenet", abl_cfg[a], f"tidenet__{a}") for a in args.ablations]
        jobs += [("tidenet", {}, "tidenet__full")] if "tidenet__full" not in [j[2] for j in jobs] else []

    for name, ablation, tag in jobs:
        for seed in seeds:
            print(f"=== {tag}  seed {seed}  ({cfg['experiment']['split']}, {cfg['experiment']['outcome']})")
            m = run_experiment(cfg, name, seed=seed, ablation=ablation, tag=tag)
            print({k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    print(aggregate_runs(run_dir(cfg, "_").parent).to_string())
