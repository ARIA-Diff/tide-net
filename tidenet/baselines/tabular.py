import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from ..data import constants as C
from ..data.dataset import window_bounds


def instance_features(p, L, obs_days, age_L, female):
    lo, hi = window_bounds(p["t"], L, obs_days, 10 ** 9)
    t = p["t"][lo:hi].astype(np.float64)
    x, m, c = p["x"][lo:hi], p["m"][lo:hi], p["c"][lo:hi]
    f = {}
    for d in C.MEAS_IDX:
        name = C.X_VARS[d]
        ok = m[:, d] > 0
        tt, v = t[ok], x[ok, d].astype(np.float64)
        n = len(v)
        f[f"{name}_last"] = v[-1] if n else np.nan
        f[f"{name}_mean"] = v.mean() if n else np.nan
        f[f"{name}_min"] = v.min() if n else np.nan
        f[f"{name}_max"] = v.max() if n else np.nan
        f[f"{name}_std"] = v.std() if n > 1 else np.nan
        f[f"{name}_slope"] = np.polyfit(tt / 365.25, v, 1)[0] if n >= 2 and tt[-1] > tt[0] else np.nan
        f[f"{name}_n"] = n
        f[f"{name}_tsl"] = L - tt[-1] if n else np.nan
    gaps = np.diff(t)
    f["gap_mean"] = gaps.mean() if len(gaps) else np.nan
    f["gap_median"] = np.median(gaps) if len(gaps) else np.nan
    f["gap_max"] = gaps.max() if len(gaps) else np.nan
    f["gap_std"] = gaps.std() if len(gaps) > 1 else np.nan
    f["n_visits"] = len(t)
    f["frac_inpatient"] = c[:, 2].mean()
    f["frac_emergency"] = c[:, 1].mean()
    for d in C.EVENT_IDX:
        f[f"{C.X_VARS[d]}_cnt"] = float(x[:, d].sum())
    for d in C.COMORB_IDX:
        f[C.X_VARS[d]] = float(x[-1, d])
    f["age"], f["female"] = age_L, female
    return f


def is_visit_feature(name):
    return (name.endswith("_n") or name.endswith("_tsl") or name.startswith("gap_")
            or name in ("n_visits", "frac_inpatient", "frac_emergency"))


def build_features(df, store, obs_days):
    rows = [instance_features(store[s], L, obs_days, a, fm)
            for s, L, a, fm in zip(df["subject_id"].values, df["L"].values, df["age_L"].values, df["female"].values)]
    return pd.DataFrame(rows)


def _result(prob, mu=None, n=None):
    return pd.DataFrame({"prob": prob, "mu": np.nan if mu is None else mu, "sigma": np.nan})


def kfre_risk(feat, kcfg, acr_fill):
    acr = feat["acr_last"].fillna(acr_fill).clip(lower=0.1)
    ctr, cf = kcfg["centering"], kcfg["coef"]
    lp = (cf["age10"] * (feat["age"] / 10 - ctr["age10"]) + cf["male"] * ((1 - feat["female"]) - ctr["male"])
          + cf["egfr5"] * (feat["egfr_last"] / 5 - ctr["egfr5"]) + cf["ln_acr"] * (np.log(acr) - ctr["ln_acr"]))
    return 1.0 - kcfg["baseline_survival"] ** np.exp(lp)


def run_kfre(cfg, feats, **_):
    acr_fill = feats["train"]["acr_last"].median()
    return _result(kfre_risk(feats["test"], cfg["kfre"], acr_fill).values)


def _lr_matrix(feat, cols, med, scaler=None):
    X = feat[cols].fillna(med).values
    return X if scaler is None else scaler.transform(X)


def run_logistic(cfg, feats, labels, **_):
    cols = [c for c in feats["train"].columns if c.endswith("_last")] + C.COMORB_VARS + ["age", "female"]
    med = feats["train"][cols].median()
    sc = StandardScaler().fit(_lr_matrix(feats["train"], cols, med))
    Xtr, Xva, Xte = (_lr_matrix(feats[k], cols, med, sc) for k in ("train", "val", "test"))
    best, model = -1, None
    for Cval in cfg["logistic"]["C_grid"]:
        m = LogisticRegression(C=Cval, max_iter=2000).fit(Xtr, labels["train"])
        s = average_precision_score(labels["val"], m.predict_proba(Xva)[:, 1])
        if s > best:
            best, model = s, m
    return _result(model.predict_proba(Xte)[:, 1])


def fit_lmm(store, sids, max_iter=200):
    import statsmodels.formula.api as smf
    rows = []
    for s in sids:
        p = store[s]
        ok = p["m"][:, C.EGFR_COL] > 0
        rows.append(pd.DataFrame({"sid": s, "t": p["t"][ok] / 365.25, "egfr": p["x"][ok, C.EGFR_COL]}))
    data = pd.concat(rows, ignore_index=True)
    res = smf.mixedlm("egfr ~ t", data, groups=data["sid"], re_formula="~t").fit(method="lbfgs", maxiter=max_iter)
    return dict(fe=res.fe_params.values, G=res.cov_re.values, s2=float(res.scale))


def lmm_past_slope(params, t, y):
    X = np.column_stack([np.ones_like(t), t])
    V = X @ params["G"] @ X.T + params["s2"] * np.eye(len(t))
    b = params["G"] @ X.T @ np.linalg.solve(V, y - X @ params["fe"])
    return float(params["fe"][1] + b[1])


def lmm_slopes(params, df, store, obs_days):
    out = []
    for s, L in zip(df["subject_id"].values, df["L"].values):
        p = store[s]
        lo, hi = window_bounds(p["t"], L, obs_days, 10 ** 9)
        ok = p["m"][lo:hi, C.EGFR_COL] > 0
        t = p["t"][lo:hi][ok].astype(np.float64) / 365.25
        y = p["x"][lo:hi, C.EGFR_COL][ok].astype(np.float64)
        out.append(lmm_past_slope(params, t, y))
    return np.asarray(out)


def run_lmm(cfg, dfs, store, labels, obs_days, **_):
    params = fit_lmm(store, dfs["train"]["subject_id"].unique(), cfg["lmm"]["max_iter"])
    X = {k: np.column_stack([lmm_slopes(params, dfs[k], store, obs_days), dfs[k]["egfr_L"].values]) for k in dfs}
    m = LogisticRegression(max_iter=1000).fit(X["train"], labels["train"])
    return _result(m.predict_proba(X["test"])[:, 1], mu=X["test"][:, 0])


def run_lightgbm(cfg, feats, labels, betas, visit_features=True, hp=None, seed=0, **_):
    import lightgbm as lgb
    lcfg = cfg["lightgbm"]
    params = dict(lcfg["defaults"])
    params.update(hp or {})
    cols = [c for c in feats["train"].columns if visit_features or not is_visit_feature(c)]
    Xtr, Xva = feats["train"][cols], feats["val"][cols]
    multi = isinstance(feats["test"], dict)
    Xte = {k: v[cols] for k, v in feats["test"].items()} if multi else feats["test"][cols]
    clf = lgb.LGBMClassifier(n_estimators=lcfg["n_estimators"], random_state=seed, verbose=-1,
                             bagging_freq=1, bagging_fraction=0.8, **params)
    clf.fit(Xtr, labels["train"], eval_set=[(Xva, labels["val"])], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(lcfg["early_stopping_rounds"], verbose=False)])
    ok_tr, ok_va = np.isfinite(betas["train"]), np.isfinite(betas["val"])
    reg = lgb.LGBMRegressor(n_estimators=lcfg["n_estimators"], random_state=seed, verbose=-1,
                            bagging_freq=1, bagging_fraction=0.8, **params)
    reg.fit(Xtr[ok_tr], betas["train"][ok_tr], eval_set=[(Xva[ok_va], betas["val"][ok_va])],
            callbacks=[lgb.early_stopping(lcfg["early_stopping_rounds"], verbose=False)])
    one = lambda X: _result(clf.predict_proba(X)[:, 1], mu=reg.predict(X))
    return {k: one(v) for k, v in Xte.items()} if multi else one(Xte)
