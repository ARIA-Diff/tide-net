import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y, p, bins=10):
    order = np.argsort(p)
    ece = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            ece += len(idx) / len(y) * abs(y[idx].mean() - p[idx].mean())
    return float(ece)


def calibration_slope_intercept(y, p):
    import statsmodels.api as sm
    lp = _logit(p)
    try:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        inter = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception:
        return float("nan"), float("nan")
    return float(slope), float(inter)


def calibration_curve(y, p, bins=10):
    order = np.argsort(p)
    return [(float(p[i].mean()), float(y[i].mean())) for i in np.array_split(order, bins) if len(i)]


def sens_at_specificity(y, p, spec=0.90):
    thr = np.quantile(p[y == 0], spec)
    return float(np.mean(p[y == 1] > thr))


def ppv_top_fraction(y, p, frac=0.10):
    k = max(int(round(frac * len(y))), 1)
    return float(y[np.argsort(-p)[:k]].mean())


def decision_curve(y, p, thresholds):
    n, prev = len(y), y.mean()
    rows = []
    for pt in thresholds:
        pred = p >= pt
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        w = pt / (1 - pt)
        rows.append(dict(threshold=float(pt), net_benefit=float(tp / n - fp / n * w),
                         treat_all=float(prev - (1 - prev) * w), treat_none=0.0))
    return pd.DataFrame(rows)


def slope_metrics(beta, mu, sigma, z=1.96):
    ok = np.isfinite(beta)
    if not ok.any():
        return {}
    return dict(slope_mae=float(np.mean(np.abs(beta[ok] - mu[ok]))),
                interval_coverage=float(np.mean(np.abs(beta[ok] - mu[ok]) <= z * sigma[ok])))


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    out = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        out[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    res = np.empty(n)
    res[order] = out
    return res


def _fast_delong(preds, m):
    k, n = preds.shape
    nn_ = n - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(1) / m / nn_ - (m + 1.0) / 2.0 / nn_
    v01 = (tz[:, :m] - tx) / nn_
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / nn_
    return aucs, np.atleast_2d(cov)


def delong_test(y, p1, p2):
    y = np.asarray(y).astype(int)
    order = np.argsort(-y)
    preds = np.vstack([p1, p2])[:, order]
    m = int(y.sum())
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    return float(aucs[0] - aucs[1]), float(2 * stats.norm.sf(abs(z)))


def one_landmark_per_patient(df, seed=0, id_col="subject_id"):
    return df.groupby(id_col, group_keys=False).sample(1, random_state=seed)


def patient_bootstrap_ci(df, y_col, p_col, n_boot=1000, seed=0, id_col="subject_id"):
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(id_col).indices.values())
    y, p = df[y_col].values, df[p_col].values
    aur, aup = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        aur.append(roc_auc_score(y[idx], p[idx]))
        aup.append(average_precision_score(y[idx], p[idx]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"auroc_ci": ci(aur), "auprc_ci": ci(aup)}


def evaluate_predictions(df, ecfg, y_col="y", p_col="prob", with_ci=False, seed=0):
    y, p = df[y_col].values.astype(int), df[p_col].values.astype(float)
    m = dict(n=int(len(y)), n_pos=int(y.sum()), prevalence=float(y.mean()))
    if y.min() == y.max():
        return m
    m.update(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
             sens_at_90=sens_at_specificity(y, p, ecfg["sens_at_specificity"]),
             ppv_top_decile=ppv_top_fraction(y, p, ecfg["ppv_top_fraction"]),
             brier=brier(y, p), ece=expected_calibration_error(y, p, ecfg["ece_bins"]))
    m["cal_slope"], m["cal_intercept"] = calibration_slope_intercept(y, p)
    if "mu" in df and df["mu"].notna().any():
        m.update(slope_metrics(df["beta"].values.astype(float), df["mu"].values.astype(float),
                               df["sigma"].values.astype(float), ecfg["interval_z"]))
    if with_ci:
        m.update(patient_bootstrap_ci(df, y_col, p_col, ecfg["n_bootstrap"], seed))
    return m


def subgroup_auroc(df, group_cols, y_col="y", p_col="prob"):
    rows = []
    for col in group_cols:
        for level, g in df.groupby(col):
            if g[y_col].nunique() < 2:
                continue
            rows.append(dict(subgroup=col, level=level, n=len(g), auroc=float(roc_auc_score(g[y_col], g[p_col]))))
    return pd.DataFrame(rows)


def sparse_follow_up_gap(sub: pd.DataFrame):
    t = sub[sub["subgroup"] == "visit_tertile"].set_index("level")["auroc"]
    return float(t.get("dense", np.nan) - t.get("sparse", np.nan))
