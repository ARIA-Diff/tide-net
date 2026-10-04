import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import constants as C

MONTH = 30.4375
YEAR = 365.25


class Normalizer:

    def __init__(self, log_vars):
        self.log_idx = [C.VAR_INDEX[v] for v in log_vars]
        self.mean = np.zeros(C.D_X, np.float32)
        self.std = np.ones(C.D_X, np.float32)
        self.age_mean, self.age_std, self.slope_mean = 60.0, 15.0, 0.0

    def _log(self, x):
        x = x.copy()
        x[:, self.log_idx] = np.log1p(np.clip(x[:, self.log_idx], 0, None))
        return x

    def fit(self, store, train_instances: pd.DataFrame):
        s = np.zeros(C.D_X)
        ss = np.zeros(C.D_X)
        cnt = np.zeros(C.D_X)
        for sid in train_instances["subject_id"].unique():
            p = store[sid]
            x, m = self._log(p["x"]), p["m"].astype(np.float64)
            s += (x * m).sum(0)
            ss += (x ** 2 * m).sum(0)
            cnt += m.sum(0)
        cnt = np.maximum(cnt, 1)
        self.mean = (s / cnt).astype(np.float32)
        self.std = np.sqrt(np.maximum(ss / cnt - (s / cnt) ** 2, 1e-6)).astype(np.float32)
        self.age_mean = float(train_instances["age_L"].mean())
        self.age_std = float(train_instances["age_L"].std() + 1e-6)
        self.slope_mean = float(train_instances["beta"].mean(skipna=True))
        return self

    def transform(self, x, m):
        return ((self._log(x) - self.mean) / self.std) * m


def window_bounds(t, L, obs_days, max_visits):
    hi = int(np.searchsorted(t, L, side="right"))
    lo = int(np.searchsorted(t, L - obs_days, side="left"))
    return max(lo, hi - max_visits), hi


def last_and_delta(x, m, t, t_start):
    V = x.shape[0]
    ar = np.arange(V)[:, None]
    li = np.maximum.accumulate(np.where(m > 0, ar, -1), axis=0)
    has = li >= 0
    li0 = np.maximum(li, 0)
    last = np.where(has, np.take_along_axis(x, li0, axis=0), 0.0)
    tl = np.where(has, t[li0], t_start)
    delta = (t[:, None] - tl) / MONTH
    return last.astype(np.float32), delta.astype(np.float32)


class LandmarkDataset(Dataset):
    def __init__(self, instances: pd.DataFrame, store, norm: Normalizer, obs_days: int, max_visits: int,
                 y_col: str, p_col: str):
        self.inst = instances.reset_index(drop=True)
        self.store, self.norm = store, norm
        self.obs_days, self.max_visits = obs_days, max_visits
        self.sid = self.inst["subject_id"].values
        self.L = self.inst["L"].values.astype(np.float64)
        self.y = self.inst[y_col].values.astype(np.float32)
        self.p = self.inst[p_col].values.astype(np.float32)
        self.beta = self.inst["beta"].values.astype(np.float32)
        self.se = self.inst["se"].values.astype(np.float32)
        self.age = self.inst["age_L"].values.astype(np.float32)
        self.female = self.inst["female"].values.astype(np.float32)

    def __len__(self):
        return len(self.inst)

    def __getitem__(self, i):
        p = self.store[self.sid[i]]
        L = self.L[i]
        lo, hi = window_bounds(p["t"], L, self.obs_days, self.max_visits)
        t = p["t"][lo:hi].astype(np.float64)
        m = p["m"][lo:hi].astype(np.float32)
        x = self.norm.transform(p["x"][lo:hi], m)
        t_start = L - self.obs_days
        last, delta = last_and_delta(x, m, t, t_start)
        dt = np.diff(t, prepend=t[0]) / MONTH
        has_beta = np.isfinite(self.beta[i])
        return dict(
            x=x.astype(np.float32), m=m, last=last, delta=delta, c=p["c"][lo:hi].astype(np.float32),
            t=((t - t_start) / YEAR).astype(np.float32), dt=dt.astype(np.float32),
            lmt=((L - t) / YEAR).astype(np.float32),
            gap=np.float32((L - t[-1]) / MONTH),
            s=np.array([(self.age[i] - self.norm.age_mean) / self.norm.age_std, self.female[i]], np.float32),
            y=self.y[i], p=self.p[i],
            beta=np.float32(self.beta[i] if has_beta else 0.0), se=np.float32(self.se[i] if has_beta else 1.0),
            has_beta=np.float32(has_beta), idx=i)


def collate(items):
    B = len(items)
    V = max(len(it["t"]) for it in items)
    out = {}
    for key in ("x", "m", "last", "delta", "c"):
        D = items[0][key].shape[1]
        a = np.zeros((B, V, D), np.float32)
        for b, it in enumerate(items):
            a[b, :len(it["t"])] = it[key]
        out[key] = torch.from_numpy(a)
    for key in ("t", "dt", "lmt"):
        a = np.zeros((B, V), np.float32)
        for b, it in enumerate(items):
            a[b, :len(it["t"])] = it[key]
        out[key] = torch.from_numpy(a)
    vm = np.zeros((B, V), bool)
    for b, it in enumerate(items):
        vm[b, :len(it["t"])] = True
    out["vmask"] = torch.from_numpy(vm)
    out["length"] = torch.tensor([len(it["t"]) for it in items])
    out["s"] = torch.from_numpy(np.stack([it["s"] for it in items]))
    for key in ("gap", "y", "p", "beta", "se", "has_beta"):
        out[key] = torch.tensor(np.array([it[key] for it in items], np.float32))
    out["idx"] = torch.tensor([it["idx"] for it in items])
    return out


def epoch_batches(instances: pd.DataFrame, R: int, batch_size: int, rng: np.random.Generator):
    chosen = []
    for _, idx in instances.groupby("subject_id").indices.items():
        idx = np.asarray(idx)
        chosen.extend(rng.choice(idx, size=min(R, len(idx)), replace=False).tolist())
    chosen = np.asarray(chosen)
    rng.shuffle(chosen)
    return [chosen[i:i + batch_size].tolist() for i in range(0, len(chosen), batch_size)]


def sequential_batches(n: int, batch_size: int):
    return [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]
