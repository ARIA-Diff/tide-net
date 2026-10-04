import numpy as np
import pandas as pd

from .outcomes import window_outcomes, label_columns


def add_subgroup_columns(inst: pd.DataFrame) -> pd.DataFrame:
    inst = inst.copy()
    inst["visit_tertile"] = pd.qcut(inst["n_hist"].rank(method="first"), 3,
                                    labels=["sparse", "mid", "dense"]).astype(str)
    inst["g_stage"] = pd.cut(inst["egfr_L"], [0, 30, 45, 60, 1e9], right=False,
                             labels=["G4", "G3b", "G3a", "G1-2"]).astype(str)
    inst["age_group"] = pd.cut(inst["age_L"], [0, 65, 80, 200], right=False,
                               labels=["<65", "65-79", ">=80"]).astype(str)
    inst["sex"] = np.where(inst["female"] == 1, "F", "M")
    inst["diabetes"] = np.where(inst["diabetes_L"] == 1, "yes", "no")
    return inst.reset_index(drop=True)


def make_instances(days, egfr, age, elig_start, krt_day, code_day, death_day, cfg, thresholds,
                   latent_window_slope=None):
    days = np.asarray(days)
    n = len(days)
    obs_days = cfg["obs_window_days"]
    lo, hi = cfg["landmark_egfr_range"]
    rows, last_kept = [], -np.inf
    flags = {"candidate": False, "evaluable": False}
    for k in range(n):
        L = float(days[k])
        if L < elig_start or age[k] < cfg["min_age"]:
            continue
        if krt_day is not None and krt_day <= L:
            continue
        e = float(egfr[k])
        if not (lo <= e <= hi):
            continue
        if e >= cfg["ckd_egfr_threshold"] and (code_day is None or code_day > L):
            continue
        j0 = int(np.searchsorted(days, L - obs_days, side="left"))
        n_hist = k + 1 - j0
        if n_hist < cfg["min_hist_values"] or days[k] - days[j0] < cfg["min_hist_span_days"]:
            continue
        flags["candidate"] = True
        o = window_outcomes(days, egfr, k, L, e, krt_day, death_day, cfg)
        eligible_primary = o["evaluable"] or o["kf"]
        if not (eligible_primary or o["death_pos"]):
            continue
        if eligible_primary:
            flags["evaluable"] = True
        if L - last_kept < cfg["min_instance_gap_days"]:
            continue
        last_kept = L
        row = dict(L=L, k=k, egfr_L=e, age_L=float(age[k]), n_hist=n_hist, K=o["K"], beta=o["beta"],
                   se=o["se"], kf=int(o["kf"]), kf_egfr15=int(o["kf_egfr15"]), death_pos=int(o["death_pos"]),
                   eligible_primary=int(eligible_primary))
        row.update(label_columns(o, thresholds))
        if latent_window_slope is not None:
            s = latent_window_slope(L)
            row["latent_slope"] = s
            row["y_star"] = int(o["kf"] or s <= thresholds["slope5"])
        rows.append(row)
    return rows, flags
