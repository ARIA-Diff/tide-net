import numpy as np
from scipy.stats import t as student_t

LABEL_DEFS = ("slope5", "slope3", "rel30", "composite")


def ols_slope(t_years, g):
    t_years = np.asarray(t_years, float)
    g = np.asarray(g, float)
    K = len(t_years)
    if K < 3:
        return np.nan, np.nan, K
    tc = t_years - t_years.mean()
    sxx = float(np.sum(tc ** 2))
    if sxx <= 0:
        return np.nan, np.nan, K
    beta = float(np.sum(tc * (g - g.mean())) / sxx)
    resid = g - g.mean() - beta * tc
    sigma2 = float(np.sum(resid ** 2) / (K - 2))
    return beta, float(np.sqrt(sigma2 / sxx)), K


def soft_label(beta, se, K, tau):
    if not np.isfinite(beta):
        return np.nan
    if not np.isfinite(se) or se <= 1e-8:
        return float(beta <= tau)
    return float(student_t.cdf((tau - beta) / se, df=max(K - 2, 1)))


def window_outcomes(days, egfr, k, L, egfr_L, krt_day, death_day, cfg):
    out_days = cfg["out_window_days"]
    i0 = k + 1
    i1 = int(np.searchsorted(days, L + out_days, side="right"))
    dw, gw = days[i0:i1], egfr[i0:i1]
    K = len(dw)
    span = float(dw[-1] - dw[0]) if K > 0 else 0.0
    evaluable = K >= cfg["min_out_values"] and span >= cfg["min_out_span_days"]
    beta = se = np.nan
    if evaluable:
        beta, se, _ = ols_slope((dw - L) / 365.25, gw)
    kf = krt_day is not None and L < krt_day <= L + out_days
    kf_egfr15 = bool(K > 0 and np.any(gw < cfg["kf_egfr"]))
    low = gw <= (1.0 - cfg["rel_decline"]) * egfr_L if K > 0 else np.zeros(0, bool)
    rel30 = bool(low.any() and (dw[low].max() - dw[low].min()) >= cfg["rel_decline_confirm_days"])
    died = death_day is not None and L < death_day <= L + out_days
    death_pos = bool(died and low.any())
    return dict(K=K, span=span, evaluable=bool(evaluable), beta=beta, se=se, kf=bool(kf),
                kf_egfr15=kf_egfr15, rel30_obs=rel30, death_pos=death_pos)


def label_columns(o, thresholds):
    row = {}
    kf, beta, se, K = o["kf"], o["beta"], o["se"], o["K"]
    for name in ("slope5", "slope3"):
        tau = thresholds[name]
        if kf:
            y, p = 1, 1.0
        elif np.isfinite(beta):
            y, p = int(beta <= tau), soft_label(beta, se, K, tau)
        else:
            y, p = 0, np.nan
        row[f"y_{name}"], row[f"p_{name}"] = y, p
    tau = thresholds["slope5"]
    if kf or o["kf_egfr15"]:
        row["y_composite"], row["p_composite"] = 1, 1.0
    elif np.isfinite(beta):
        row["y_composite"], row["p_composite"] = int(beta <= tau), soft_label(beta, se, K, tau)
    else:
        row["y_composite"], row["p_composite"] = 0, np.nan
    y = int(kf or o["rel30_obs"])
    row["y_rel30"], row["p_rel30"] = y, float(y)
    return row
