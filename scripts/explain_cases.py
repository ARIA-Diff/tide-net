import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from tidenet.data.dataset import LandmarkDataset, Normalizer
from tidenet.data.splits import assign_splits, prepare_instances
from tidenet.experiment import build_deep
from tidenet.interpret import explain_instance, integrated_gradients
from tidenet.utils import get_device, load_config, load_json, resolve, save_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cohort-dir", default=None)
    ap.add_argument("--subjects", nargs="*", type=int, default=[])
    ap.add_argument("--ig", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config("data", "train", "evaluation")
    run = resolve(args.run_dir)
    device = get_device(cfg["experiment"]["device"])
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    norm = Normalizer(cfg["input"]["log_vars"])
    norm.__dict__.update(ckpt["norm"])
    hp = load_json(run / "hparams.json")
    model = build_deep("tidenet", hp, norm)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    cdir = resolve(args.cohort_dir or cfg["paths"]["out_dir"])
    inst = pd.read_parquet(cdir / "instances.parquet")
    with open(cdir / "store.pkl", "rb") as f:
        store = pickle.load(f)
    cols = (f"y_{cfg['experiment']['outcome']}", f"p_{cfg['experiment']['outcome']}")
    out = resolve(args.out) if args.out else run / "explain"
    out.mkdir(parents=True, exist_ok=True)
    mk = lambda d: LandmarkDataset(d.reset_index(drop=True), store, norm, cfg["cohort"]["obs_window_days"],
                                   cfg["input"]["max_visits"], *cols)

    if args.subjects:
        d = inst[inst["subject_id"].isin(args.subjects)].sort_values(["subject_id", "L"]).reset_index(drop=True)
        ds = mk(d)
        summaries = []
        for i in range(len(ds)):
            s, visits, curves = explain_instance(model, ds, i, device, grid_months=list(range(0, 61)))
            summaries.append(s)
            visits.to_csv(out / f"visits_{s['subject_id']}_{int(s['L'])}.csv", index=False)
        pd.DataFrame(summaries).to_csv(out / "case_predictions.csv", index=False)

    if args.ig:
        df = prepare_instances(inst, cfg["experiment"]["outcome"])
        df = assign_splits(df, cfg, cfg["experiment"]["split"], cfg["experiment"]["outcome"])
        ig = integrated_gradients(model, mk(df[df["split"] == "test"]), device)
        ig.to_csv(out / "integrated_gradients.csv", header=["mean_abs_attribution"])
        save_json({"process_share": float(ig["process_share"])}, out / "process_share.json")
