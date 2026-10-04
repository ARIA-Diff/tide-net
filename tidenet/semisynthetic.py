import pickle

import numpy as np
import pandas as pd

from .baselines import tabular as T
from .data import constants as C
from .data.ckdepi import creatinine_from_egfr
from .data.landmarks import add_subgroup_columns, make_instances
from .data.splits import prepare_instances
from .evaluate import evaluate_predictions
from .experiment import TABULAR, fit_deep, load_cohort, parse_name, predict_deep, selected_hparams
from .utils import get_device, resolve, save_json, seed_all


def fit_latent(store, max_iter=200):
    import statsmodels.formula.api as smf
    rows = []
    for sid, p in store.items():
        ok = p["m"][:, C.EGFR_COL] > 0
        rows.append(pd.DataFrame({"sid": sid, "t": p["t"][ok] / 365.25, "egfr": p["x"][ok, C.EGFR_COL]}))
    data = pd.concat(rows, ignore_index=True)
    lin = smf.mixedlm("egfr ~ t", data, groups=data["sid"], re_formula="~t").fit(reml=False, method="lbfgs", maxiter=max_iter)
    quad = smf.mixedlm("egfr ~ t + I(t**2)", data, groups=data["sid"], re_formula="~t").fit(reml=False, method="lbfgs",
                                                                                          maxiter=max_iter)
    res = quad if quad.bic < lin.bic else lin
    fe = res.fe_params.values
    latent = {}
    for sid, re in res.random_effects.items():
        re = np.asarray(re)
        latent[sid] = (float(fe[0] + re[0]), float(fe[1] + re[1]), float(fe[2]) if len(fe) > 2 else 0.0)
    return latent, float(np.sqrt(res.scale)), bool(res is quad)


def latent_g(coef, t_days):
    tau = np.asarray(t_days, float) / 365.25
    a0, a1, a2 = coef
    return a0 + a1 * tau + a2 * tau ** 2, a1 + 2 * a2 * tau


def latent_window_slope_fn(coef, out_days):
    def fn(L):
        grid = np.arange(L + 1.0, L + out_days + 1.0)
        g, _ = latent_g(coef, grid)
        return float(np.polyfit(grid / 365.25, g, 1)[0])
    return fn


def median_gap_days(store):
    return float(np.median(np.concatenate([np.diff(p["t"]) for p in store.values() if len(p["t"]) > 1])))


def simulate_patient(p, coef, alpha, alpha2, cutoff, lam0, rate_mult, resid_sd, rng):
    t_real = p["t"].astype(np.float64)
    lo, hi = t_real[0], t_real[-1]
    grid = np.arange(lo, hi + 1.0)
    g, dg = latent_g(coef, grid)
    rate = np.minimum(rate_mult * lam0 * np.exp(alpha * (-dg / 5.0) + alpha2 * (g < cutoff)), 1.0)
    lam_max = rate.max()
    n_cand = rng.poisson(lam_max * (hi - lo))
    cand = rng.uniform(lo, hi, n_cand)
    keep = rng.uniform(size=n_cand) < rate[np.clip((cand - lo).astype(int), 0, len(rate) - 1)] / lam_max
    times = np.unique(np.floor(cand[keep]))
    if len(times) < 2:
        return None
    pos = np.clip(np.searchsorted(t_real, times), 1, len(t_real) - 1)
    near = np.where(np.abs(t_real[pos - 1] - times) <= np.abs(t_real[pos] - times), pos - 1, pos)
    x, m = p["x"][near].copy(), p["m"][near].copy()
    age = p["age"][near] + (times - t_real[near]) / 365.25
    egfr = np.clip(latent_g(coef, times)[0] + rng.normal(0.0, resid_sd, len(times)), 1.0, 200.0)
    female = np.full(len(times), p["female"])
    x[:, C.EGFR_COL], m[:, C.EGFR_COL] = egfr, 1
    x[:, C.CREAT_COL], m[:, C.CREAT_COL] = creatinine_from_egfr(egfr, age, female), 1
    x[:, C.EVENT_IDX], m[:, C.EVENT_IDX] = 0, 0
    return dict(t=times.astype(np.float32), x=x, m=m, c=p["c"][near], age=age.astype(np.float32), female=p["female"])


def build_simulated(store, meta, latent, resid_sd, alpha, rate_mult, cfg, scfg, seed):
    cc, th = cfg["cohort"], cfg["outcomes"]
    rng = np.random.default_rng(seed)
    lam0 = 1.0 / median_gap_days(store)
    meta = meta.set_index("subject_id")
    new_store, rows = {}, []
    for sid, p in store.items():
        if sid not in latent:
            continue
        sim = simulate_patient(p, latent[sid], alpha, scfg["alpha2"], scfg["egfr_cutoff"], lam0, rate_mult, resid_sd, rng)
        if sim is None:
            continue
        m = meta.loc[sid]
        nz = lambda v: None if v is None or pd.isna(v) else float(v)
        r, _ = make_instances(sim["t"].astype(np.float64), sim["x"][:, C.EGFR_COL], sim["age"].astype(np.float64),
                              m["elig_start"], nz(m["krt_day"]), nz(m["code_day"]), nz(m["death_day"]), cc, th,
                              latent_window_slope=latent_window_slope_fn(latent[sid], cc["out_window_days"]))
        if not r:
            continue
        new_store[sid] = sim
        for x in r:
            x.update(subject_id=sid, female=int(m["female"]), group=m["group"],
                     diabetes_L=int(sim["x"][x["k"], C.DIABETES_COL]))
        rows.extend(r)
    return add_subgroup_columns(pd.DataFrame(rows)), new_store


def patient_split(sids, seed, fracs=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    sids = np.array(sorted(sids))
    rng.shuffle(sids)
    n1, n2 = int(fracs[0] * len(sids)), int((fracs[0] + fracs[1]) * len(sids))
    out = {s: "train" for s in sids[:n1]}
    out.update({s: "val" for s in sids[n1:n2]})
    out.update({s: "test" for s in sids[n2:]})
    return out


def _sim_set(cache, key, store, meta, latent, resid_sd, alpha, rate_mult, cfg, scfg, pmap):
    d = resolve(scfg["output_dir"]) / f"alpha{alpha}_rate{rate_mult}"
    if key not in cache:
        if (d / "instances.parquet").exists():
            inst = pd.read_parquet(d / "instances.parquet")
            with open(d / "store.pkl", "rb") as f:
                st = pickle.load(f)
        else:
            inst, st = build_simulated(store, meta, latent, resid_sd, alpha, rate_mult, cfg, scfg,
                                       seed=scfg["seed"] + int(100 * alpha) + int(10 * rate_mult))
            d.mkdir(parents=True, exist_ok=True)
            inst.to_parquet(d / "instances.parquet")
            with open(d / "store.pkl", "wb") as f:
                pickle.dump(st, f, protocol=4)
        inst = prepare_instances(inst, "slope5")
        inst["split"] = inst["subject_id"].map(pmap)
        inst["y_eval"] = inst[scfg["eval_label"]]
        cache[key] = (inst.dropna(subset=["split"]), st)
    return cache[key]


def _metrics(df, pred, ecfg):
    res = pd.concat([df[["subject_id", "L", "beta", "y_eval"]].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    m = evaluate_predictions(res, ecfg, y_col="y_eval")
    return {k: m.get(k) for k in ("auroc", "auprc", "cal_slope", "cal_intercept", "brier")}


def run_semisynthetic(cfg, scfg, seeds, log=print):
    inst, store, meta = load_cohort(cfg)
    latent, resid_sd, quad = fit_latent(store, scfg["lmm_max_iter"])
    log(f"latent LMM fitted (quadratic term: {quad}); residual SD {resid_sd:.2f}")
    pmap = patient_split(store.keys(), scfg["seed"])
    device = get_device(cfg["experiment"]["device"])
    cache, rows = {}, []
    ecfg, obs = cfg["evaluation"], cfg["cohort"]["obs_window_days"]
    cols = ("y_slope5", "p_slope5")
    get = lambda a, r: _sim_set(cache, (a, r), store, meta, latent, resid_sd, a, r, cfg, scfg, pmap)

    for a_tr in scfg["train_alphas"]:
        tr_all, tr_store = get(a_tr, 1.0)
        tr, va = (tr_all[tr_all["split"] == s].reset_index(drop=True) for s in ("train", "val"))
        tests = {a: get(a, 1.0) for a in scfg["alphas"]}
        te = {a: d[d["split"] == "test"].reset_index(drop=True) for a, (d, _) in tests.items()}
        for name in scfg["models"]:
            for seed in seeds:
                seed_all(seed)
                if name in TABULAR:
                    dfs = {"train": tr, "val": va}
                    feats = {"train": T.build_features(tr, tr_store, obs), "val": T.build_features(va, tr_store, obs)}
                    feats["test"] = {a: T.build_features(te[a], tests[a][1], obs) for a in te}
                    labels = {k: v[cols[0]].values.astype(int) for k, v in dfs.items()}
                    betas = {k: v["beta"].values.astype(float) for k, v in dfs.items()}
                    preds = T.run_lightgbm(cfg, feats, labels, betas, visit_features=(name == "lightgbm"),
                                           hp=selected_hparams(name), seed=seed)
                else:
                    model, norm, _, _ = fit_deep(cfg, name, tr, va, tr_store, cols, seed, device, log=lambda *_: None)
                    preds = {a: predict_deep(cfg, model, norm, te[a], tests[a][1], cols, device) for a in te}
                for a_te in te:
                    rows.append(dict(experiment="shift", model=name, train_alpha=a_tr, test_alpha=a_te, seed=seed,
                                     **_metrics(te[a_te], preds[a_te], ecfg)))
                log(f"[shift] train alpha={a_tr} {name} seed {seed} done")
    res = pd.DataFrame(rows)

    rows2 = []
    for rate in scfg["rate_multipliers"]:
        d_all, d_store = get(scfg["soft_label_study"]["alpha"], rate)
        tr, va, te = (d_all[d_all["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
        for name in scfg["soft_label_study"]["models"]:
            for variant, abl in (("soft", {}), ("hard", {"soft_label": False})):
                for seed in seeds:
                    model, norm, _, _ = fit_deep(cfg, name, tr, va, d_store, cols, seed, device, ablation=abl,
                                                 log=lambda *_: None)
                    pred = predict_deep(cfg, model, norm, te, d_store, cols, device)
                    rows2.append(dict(experiment="soft_label", model=name, variant=variant, rate_multiplier=rate,
                                      seed=seed, **_metrics(te, pred, ecfg)))
    res2 = pd.DataFrame(rows2)

    out = resolve(scfg["output_dir"])
    res.to_csv(out / "shift_results.csv", index=False)
    res2.to_csv(out / "soft_label_results.csv", index=False)
    save_json(summarize_shift(res, scfg), out / "shift_summary.json")
    return res, res2


def summarize_shift(res, scfg):
    hi, lo = max(scfg["alphas"]), min(scfg["alphas"])
    g = res[res["experiment"] == "shift"].groupby(["model", "train_alpha", "test_alpha"])[["auroc", "cal_slope"]].mean()
    out = {}
    for model in res["model"].unique():
        try:
            matched_hi = g.loc[(model, hi, hi), "auroc"]
            matched_lo = g.loc[(model, lo, lo), "auroc"] if (model, lo, lo) in g.index else np.nan
            shifted = g.loc[(model, hi, lo), "auroc"]
            out[model] = dict(auroc_matched_alpha_hi=float(matched_hi), gain_in_distribution=float(matched_hi - matched_lo),
                              loss_under_shift=float(matched_hi - shifted),
                              cal_slope_under_shift=float(g.loc[(model, hi, lo), "cal_slope"]))
        except KeyError:
            continue
    return out
