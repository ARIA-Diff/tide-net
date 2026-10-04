import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data.splits import assign_splits, prepare_instances
from .experiment import TABULAR, load_cohort, model_settings, parse_name, fit_deep, predict_deep
from .baselines import tabular as T
from .utils import get_device, resolve, save_json


def sample_configs(space: dict, n: int, seed: int):
    rng = np.random.default_rng(seed)
    return [{k: v[int(rng.integers(len(v)))] for k, v in space.items()} for _ in range(n)]


def search_model(cfg, scfg, name, log=print):
    ex = cfg["experiment"]
    outcome = ex["outcome"]
    cols = (f"y_{outcome}", f"p_{outcome}")
    inst, store, _ = load_cohort(cfg)
    df = assign_splits(prepare_instances(inst, outcome), cfg, "random", outcome)
    dfs = {k: df[df["split"] == k].reset_index(drop=True) for k in ("train", "val")}
    device = get_device(ex["device"])
    base, _ = parse_name(name)
    space = scfg["tidenet"] if base.startswith("tidenet") else (
        scfg["lightgbm"] if base.startswith("lightgbm") else scfg["baselines_deep"])
    results = []
    feats = None
    for i, hp in enumerate(sample_configs(space, scfg["n_configs"], scfg["seed"])):
        if base.startswith("lightgbm"):
            if feats is None:
                feats = {k: T.build_features(dfs[k], store, cfg["cohort"]["obs_window_days"]) for k in dfs}
            labels = {k: dfs[k][cols[0]].values.astype(int) for k in dfs}
            betas = {k: dfs[k]["beta"].values.astype(float) for k in dfs}
            f2 = {"train": feats["train"], "val": feats["val"], "test": feats["val"]}
            b2 = {"train": betas["train"], "val": betas["val"]}
            pred = T.run_lightgbm(cfg, f2, labels, b2, visit_features=(base == "lightgbm"), hp=hp, seed=0)
            score = average_precision_score(labels["val"], pred["prob"])
        else:
            _, _, _, score = fit_deep(cfg, name, dfs["train"], dfs["val"], store, cols, 0, device,
                                      hp_override=hp, log=lambda *_: None)
        log(f"[{name}] config {i + 1}/{scfg['n_configs']}  val AUPRC {score:.4f}  {hp}")
        results.append(dict(hp=hp, val_auprc=float(score)))
    best = max(results, key=lambda r: r["val_auprc"])
    out = resolve("outputs/hparams")
    save_json(best["hp"], out / f"{base.replace('_noproc', '').replace('_novisit', '')}.json")
    save_json(results, out / f"{base}_search_log.json")
    return best
