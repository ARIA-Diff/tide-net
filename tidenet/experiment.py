import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import tabular as T
from .data.dataset import LandmarkDataset, Normalizer
from .data.splits import assign_splits, prepare_instances
from .evaluate import (delong_test, evaluate_predictions, one_landmark_per_patient, sparse_follow_up_gap,
                       subgroup_auroc)
from .models.baselines import DEEP_BASELINES, build_baseline
from .models.tidenet import TIDENet
from .data import constants as C
from .train import fit, predict
from .utils import deep_update, get_device, load_json, resolve, save_json, seed_all

TABULAR = ("kfre", "lmm", "logistic", "lightgbm", "lightgbm_novisit")
SUBGROUP_COLS = ["visit_tertile", "g_stage", "diabetes", "sex", "age_group"]


def load_cohort(cfg):
    d = resolve(cfg["paths"]["out_dir"])
    inst = pd.read_parquet(d / "instances.parquet")
    with open(d / "store.pkl", "rb") as f:
        store = pickle.load(f)
    meta = pd.read_parquet(d / "patients_meta.parquet")
    return inst, store, meta


def parse_name(name):
    soft = name.endswith("_sl")
    return (name[:-3] if soft else name), soft


def selected_hparams(name, hp_dir="outputs/hparams"):
    key = parse_name(name)[0].replace("_noproc", "").replace("_novisit", "")
    p = resolve(hp_dir) / f"{key}.json"
    return load_json(p) if p.exists() else {}


def model_settings(cfg, name, ablation=None, hp_override=None):
    base, soft = parse_name(name)
    if base.startswith("tidenet"):
        hp = deep_update(cfg["tidenet"], selected_hparams(name))
        if base == "tidenet_noproc":
            hp["use_process_view"] = False
        hp = deep_update(hp, ablation or {})
        hp = deep_update(hp, hp_override or {})
        loss_kw = dict(soft_label=hp["soft_label"], loss_reg=hp["loss_reg"], lambda_reg=hp["lambda_reg"],
                       lambda_tpp=hp["lambda_tpp"] if hp.get("use_tpp_loss", True) else 0.0)
    else:
        hp = deep_update(cfg["baselines_deep"], selected_hparams(name))
        hp = deep_update(hp, hp_override or {})
        loss_kw = dict(soft_label=soft, loss_reg="hetero" if soft else "none", lambda_reg=hp["lambda_reg"],
                       lambda_tpp=0.0)
    return hp, loss_kw


def build_deep(name, hp, norm):
    base, _ = parse_name(name)
    if base.startswith("tidenet"):
        return TIDENet(C.D_X, C.D_C, 2, hp, slope_init=norm.slope_mean)
    if base in DEEP_BASELINES:
        return build_baseline(base, hp, slope_init=norm.slope_mean)
    raise ValueError(name)


def fit_deep(cfg, name, train_df, val_df, store, cols, seed, device, ablation=None, hp_override=None, log=print):
    seed_all(seed)
    hp, loss_kw = model_settings(cfg, name, ablation, hp_override)
    norm = Normalizer(cfg["input"]["log_vars"]).fit(store, train_df)
    mk = lambda d: LandmarkDataset(d, store, norm, cfg["cohort"]["obs_window_days"] if "cohort" in cfg else 1826,
                                   cfg["input"]["max_visits"], *cols)
    model = build_deep(name, hp, norm)
    model, hist, best = fit(model, mk(train_df), mk(val_df), loss_kw, cfg["optim"], device, seed,
                            cfg["experiment"]["num_workers"], log)
    return model, norm, hist, best


def predict_deep(cfg, model, norm, df, store, cols, device):
    ds = LandmarkDataset(df, store, norm, cfg["cohort"]["obs_window_days"] if "cohort" in cfg else 1826,
                         cfg["input"]["max_visits"], *cols)
    return predict(model, ds, device, num_workers=cfg["experiment"]["num_workers"])


def run_tabular(cfg, name, dfs, store, cols, seed, hp=None):
    obs = cfg["cohort"]["obs_window_days"]
    labels = {k: dfs[k][cols[0]].values.astype(int) for k in dfs}
    betas = {k: dfs[k]["beta"].values.astype(float) for k in dfs}
    if name in ("kfre", "logistic", "lightgbm", "lightgbm_novisit"):
        feats = {k: T.build_features(dfs[k], store, obs) for k in dfs}
    if name == "kfre":
        return T.run_kfre(cfg, feats=feats)
    if name == "logistic":
        return T.run_logistic(cfg, feats=feats, labels=labels)
    if name == "lmm":
        return T.run_lmm(cfg, dfs=dfs, store=store, labels=labels, obs_days=obs)
    return T.run_lightgbm(cfg, feats, labels, betas, visit_features=(name == "lightgbm"), hp=hp or selected_hparams(name),
                          seed=seed)


def run_dir(cfg, tag):
    ex = cfg["experiment"]
    suffix = "_deathpos" if ex.get("include_death_positive") else ""
    return resolve(ex["output_dir"]) / f"{ex['split']}_{ex['outcome']}{suffix}" / tag


def run_experiment(cfg, name, seed=0, ablation=None, tag=None, log=print):
    ex = cfg["experiment"]
    outcome = ex["outcome"]
    cols = (f"y_{outcome}", f"p_{outcome}")
    inst, store, _ = load_cohort(cfg)
    df = prepare_instances(inst, outcome, ex.get("include_death_positive", False))
    df = assign_splits(df, cfg, ex["split"], outcome)
    dfs = {k: df[df["split"] == k].reset_index(drop=True) for k in ("train", "val", "test")}
    device = get_device(ex["device"])
    out = run_dir(cfg, tag or name) / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    if name in TABULAR:
        seed_all(seed)
        pred = run_tabular(cfg, name, dfs, store, cols, seed)
    else:
        model, norm, hist, best = fit_deep(cfg, name, dfs["train"], dfs["val"], store, cols, seed, device,
                                           ablation, log=log)
        pred = predict_deep(cfg, model, norm, dfs["test"], store, cols, device)
        hist.to_csv(out / "history.csv", index=False)
        save_json(model_settings(cfg, name, ablation)[0], out / "hparams.json")
        import torch
        torch.save({"model": model.state_dict(), "norm": norm.__dict__}, out / "model.pt")

    res = dfs["test"][["subject_id", "L", "beta", "se"] + SUBGROUP_COLS].copy()
    res["y"] = dfs["test"][cols[0]].values
    res = pd.concat([res, pred], axis=1)
    res.to_parquet(out / "predictions.parquet")
    metrics = evaluate_predictions(res, cfg["evaluation"], with_ci=(seed == 0), seed=seed)
    sub = subgroup_auroc(res, SUBGROUP_COLS)
    sub.to_csv(out / "subgroups.csv", index=False)
    if len(sub):
        metrics["sparse_follow_up_gap"] = sparse_follow_up_gap(sub)
    save_json(metrics, out / "metrics.json")
    return metrics


def aggregate_runs(root):
    rows = []
    for mdir in sorted(Path(root).iterdir()):
        if not mdir.is_dir():
            continue
        ms = [json.load(open(p)) for p in sorted(mdir.glob("seed*/metrics.json"))]
        if not ms:
            continue
        df = pd.DataFrame(ms).select_dtypes("number")
        row = {"model": mdir.name, "n_seeds": len(ms)}
        for c in df.columns:
            row[c] = df[c].mean()
            row[f"{c}_sd"] = df[c].std()
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(Path(root) / "summary.csv", index=False)
    return out


def compare_auroc(root, model_a, model_b, seed=0):
    a = pd.read_parquet(Path(root) / model_a / f"seed{seed}" / "predictions.parquet")
    b = pd.read_parquet(Path(root) / model_b / f"seed{seed}" / "predictions.parquet")
    a = one_landmark_per_patient(a, seed)
    b = b.set_index(["subject_id", "L"]).loc[list(zip(a["subject_id"], a["L"]))]
    diff, p = delong_test(a["y"].values, a["prob"].values, b["prob"].values)
    return {"model_a": model_a, "model_b": model_b, "auroc_diff": diff, "p_value": p}
