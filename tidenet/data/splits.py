import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_instances(inst: pd.DataFrame, outcome: str, include_death_positive: bool = False) -> pd.DataFrame:
    d = inst.copy()
    ycol, pcol = f"y_{outcome}", f"p_{outcome}"
    if include_death_positive:
        d = d[(d["eligible_primary"] == 1) | (d["death_pos"] == 1)].copy()
        dp = d["death_pos"] == 1
        d.loc[dp, ycol] = 1
        d.loc[dp, pcol] = 1.0
    else:
        d = d[d["eligible_primary"] == 1].copy()
    return d.dropna(subset=[pcol]).reset_index(drop=True)


def assign_splits(inst: pd.DataFrame, cfg: dict, mode: str, outcome: str = "slope5") -> pd.DataFrame:
    sc = cfg["splits"]
    seed = sc["split_seed"]
    pat = inst.groupby("subject_id").agg(pos=(f"y_{outcome}", "max"), group=("group", "first")).reset_index()
    if mode == "random":
        r = sc["random"]
        trainval, test = train_test_split(pat, test_size=r["test"], stratify=pat["pos"], random_state=seed)
        train, val = train_test_split(trainval, test_size=r["val"] / (r["train"] + r["val"]),
                                      stratify=trainval["pos"], random_state=seed)
    elif mode == "temporal":
        trainval = pat[pat["group"].isin(sc["temporal_train_groups"])]
        test = pat[pat["group"].isin(sc["temporal_test_groups"])]
        train, val = train_test_split(trainval, test_size=sc["temporal_val_fraction"],
                                      stratify=trainval["pos"], random_state=seed)
    else:
        raise ValueError(mode)
    assign = {}
    for name, df in (("train", train), ("val", val), ("test", test)):
        assign.update({s: name for s in df["subject_id"]})
    out = inst.copy()
    out["split"] = out["subject_id"].map(assign)
    return out.dropna(subset=["split"]).reset_index(drop=True)
