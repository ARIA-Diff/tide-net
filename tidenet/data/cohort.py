import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import constants as C
from .ckdepi import ckd_epi_2021
from .landmarks import add_subgroup_columns, make_instances
from ..utils import resolve, save_json


def _view(con, mimic, module, table, types=None):
    path = (Path(mimic) / module / f"{table}.csv.gz").as_posix()
    opt = ""
    if types:
        opt = ", types={" + ", ".join(f"'{k}': '{v}'" for k, v in types.items()) + "}"
    con.execute(f"CREATE OR REPLACE VIEW {table} AS "
                f"SELECT * FROM read_csv('{path}', header=true{opt}, sample_size=100000)")


def _prefix_match(codes: pd.Series, versions: pd.Series, p9, p10):
    codes = codes.astype(str).str.strip()
    m9 = (versions == 9) & codes.str.startswith(tuple(p9)) if p9 else pd.Series(False, index=codes.index)
    m10 = (versions == 10) & codes.str.startswith(tuple(p10)) if p10 else pd.Series(False, index=codes.index)
    return m9 | m10


def ckd_onset_from_egfr(days, egfr, thr=60.0, confirm=90):
    low = egfr < thr
    n, i = len(days), 0
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and low[j + 1]:
            j += 1
        if days[j] - days[i] >= confirm:
            kk = int(np.searchsorted(days[i:j + 1], days[i] + confirm, side="left")) + i
            return int(days[kk])
        i = j + 1
    return None


def _attach(meas, vkeys, tol):
    left = meas.sort_values("day")
    right = vkeys.rename(columns={"day": "visit_day"}).assign(day=vkeys["day"]).sort_values("day")
    m = pd.merge_asof(left, right, on="day", by="subject_id", direction="forward", tolerance=tol)
    m = m.dropna(subset=["visit_day"])
    m["visit_day"] = m["visit_day"].astype(np.int64)
    return m


def build_cohort(cfg):
    cc, th = cfg["cohort"], cfg["outcomes"]
    pc = cfg["paths"]
    out_dir = resolve(pc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mimic = resolve(pc["mimic_dir"])
    con = duckdb.connect()
    _view(con, mimic, "hosp", "patients")
    _view(con, mimic, "hosp", "admissions")
    _view(con, mimic, "hosp", "labevents",
          {"subject_id": "BIGINT", "itemid": "INTEGER", "charttime": "TIMESTAMP", "valuenum": "DOUBLE"})
    _view(con, mimic, "hosp", "diagnoses_icd")
    _view(con, mimic, "hosp", "procedures_icd")
    _view(con, mimic, "hosp", "prescriptions")
    _view(con, mimic, "hosp", "omr")
    _view(con, mimic, "icu", "icustays")
    if pc.get("use_chartevents", True):
        _view(con, mimic, "icu", "chartevents",
              {"subject_id": "BIGINT", "itemid": "INTEGER", "charttime": "TIMESTAMP", "valuenum": "DOUBLE"})

    groups = ", ".join(f"'{g}'" for g in cc["anchor_year_groups"])
    pat = con.execute(
        f"SELECT subject_id, gender, anchor_age, anchor_year, anchor_year_group, CAST(dod AS DATE) AS dod "
        f"FROM patients WHERE anchor_age >= {cc['min_age']} AND anchor_year_group IN ({groups})").df()
    pat["subject_id"] = pat["subject_id"].astype(np.int64)
    con.register("cp", pat[["subject_id"]])

    lo, hi = cc["creatinine_range"]
    cr = con.execute(f"""
        WITH cr AS (
          SELECT l.subject_id, CAST(l.charttime AS DATE) AS day, median(l.valuenum) AS creatinine
          FROM labevents l JOIN cp ON cp.subject_id = l.subject_id
          WHERE l.itemid = {cc['creatinine_itemid']} AND l.valuenum BETWEEN {lo} AND {hi}
          GROUP BY 1, 2)
        SELECT cr.subject_id, cr.day, cr.creatinine,
               COALESCE(MAX(CASE WHEN cr.day >= CAST(a.admittime AS DATE) THEN 2 ELSE 1 END), 0) AS setting
        FROM cr LEFT JOIN admissions a
          ON a.subject_id = cr.subject_id
         AND cr.day BETWEEN CAST(COALESCE(a.edregtime, a.admittime) AS DATE) AND CAST(a.dischtime AS DATE)
        GROUP BY cr.subject_id, cr.day, cr.creatinine""").df()
    cr["subject_id"] = cr["subject_id"].astype(np.int64)
    cr["date"] = pd.to_datetime(cr["day"])
    cr = cr.merge(pat, on="subject_id", how="inner").sort_values(["subject_id", "date"]).reset_index(drop=True)
    t0 = cr.groupby("subject_id")["date"].min()
    cr["day"] = (cr["date"] - cr["subject_id"].map(t0)).dt.days.astype(np.int64)
    cr["age"] = cr["anchor_age"] + cr["date"].dt.year - cr["anchor_year"]
    cr["female"] = (cr["gender"] == "F").astype(int)
    cr["egfr"] = ckd_epi_2021(cr["creatinine"], cr["age"], cr["female"])
    flow = {"adults_with_creatinine": int(cr["subject_id"].nunique())}

    code = con.execute(f"""
        SELECT d.subject_id, MIN(CAST(a.admittime AS DATE)) AS code_date
        FROM diagnoses_icd d JOIN admissions a ON a.hadm_id = d.hadm_id JOIN cp ON cp.subject_id = d.subject_id
        WHERE regexp_matches(trim(d.icd_code), '{C.CKD_REGEX}') GROUP BY 1""").df()
    code["code_date"] = pd.to_datetime(code["code_date"])
    code_map = dict(zip(code["subject_id"].astype(np.int64), code["code_date"]))

    meta = []
    for sid, g in cr.groupby("subject_id", sort=False):
        days, egfr = g["day"].values, g["egfr"].values
        onset_e = ckd_onset_from_egfr(days, egfr, cc["ckd_egfr_threshold"], cc["ckd_confirm_days"])
        cdate = code_map.get(sid)
        code_day = int((cdate - t0[sid]).days) if cdate is not None and pd.notna(cdate) else None
        onsets = [d for d in (onset_e, code_day) if d is not None]
        if not onsets:
            continue
        adult = np.flatnonzero(g["age"].values >= cc["min_age"])
        if len(adult) == 0:
            continue
        elig = max(min(onsets), int(days[adult[0]]))
        dod = g["dod"].iloc[0]
        meta.append(dict(subject_id=sid, female=int(g["female"].iloc[0]), group=g["anchor_year_group"].iloc[0],
                         elig_start=elig, code_day=code_day,
                         death_day=int((pd.Timestamp(dod) - t0[sid]).days) if pd.notna(dod) else None,
                         t0=t0[sid]))
    meta = pd.DataFrame(meta)
    flow["ckd_criteria_met"] = int(len(meta))
    cand = meta[["subject_id"]].copy()
    con.register("cand", cand)

    dx = con.execute("""
        SELECT d.subject_id, trim(d.icd_code) AS icd_code, d.icd_version, CAST(a.admittime AS DATE) AS date
        FROM diagnoses_icd d JOIN admissions a ON a.hadm_id = d.hadm_id JOIN cand ON cand.subject_id = d.subject_id""").df()
    dx["subject_id"] = dx["subject_id"].astype(np.int64)
    dx["date"] = pd.to_datetime(dx["date"])
    px = con.execute("""
        SELECT p.subject_id, trim(p.icd_code) AS icd_code, p.icd_version, CAST(p.chartdate AS DATE) AS date
        FROM procedures_icd p JOIN cand ON cand.subject_id = p.subject_id""").df()
    px["subject_id"] = px["subject_id"].astype(np.int64)
    px["date"] = pd.to_datetime(px["date"])
    t0_map = t0.to_dict()

    def to_day(df):
        return (df["date"] - df["subject_id"].map(t0_map)).dt.days.astype(np.int64)

    krt_dates = pd.concat([
        dx.loc[_prefix_match(dx["icd_code"], dx["icd_version"], C.KRT_ICD9_DX, C.KRT_ICD10_DX), ["subject_id", "date"]],
        px.loc[_prefix_match(px["icd_code"], px["icd_version"], C.KRT_ICD9_PROC, C.KRT_ICD10_PROC), ["subject_id", "date"]]])
    krt_dates["day"] = to_day(krt_dates)
    krt_first = krt_dates.groupby("subject_id")["day"].min()
    meta["krt_day"] = meta["subject_id"].map(krt_first)
    excl_krt = meta["krt_day"].notna() & (meta["krt_day"] <= meta["elig_start"])
    flow["excluded_previous_krt"] = int(excl_krt.sum())
    meta = meta[~excl_krt].reset_index(drop=True)
    keep_ids = set(meta["subject_id"])
    con.register("cand", meta[["subject_id"]])

    vis = cr[cr["subject_id"].isin(keep_ids)][["subject_id", "date", "day", "creatinine", "egfr", "setting",
                                                 "age", "female"]].copy()
    vkeys = vis[["subject_id", "day"]]
    tol = cc["lab_tolerance_days"]

    id2var = {i: v for v, ids in C.LAB_ITEMS.items() for i in ids}
    ids = ", ".join(str(i) for i in id2var)
    lab = con.execute(f"""
        SELECT l.subject_id, CAST(l.charttime AS DATE) AS date, l.itemid, median(l.valuenum) AS val
        FROM labevents l JOIN cand ON cand.subject_id = l.subject_id
        WHERE l.itemid IN ({ids}) AND l.valuenum IS NOT NULL GROUP BY 1, 2, 3""").df()
    lab["var"] = lab["itemid"].map(id2var)
    dip = con.execute(f"""
        SELECT l.subject_id, CAST(l.charttime AS DATE) AS date, upper(trim(l.value)) AS v
        FROM labevents l JOIN cand ON cand.subject_id = l.subject_id WHERE l.itemid = {C.DIPSTICK_ITEM}""").df()
    dip["val"] = dip["v"].map(C.DIPSTICK_MAP)
    dip["var"] = "urine_dipstick"
    meas = [lab[["subject_id", "date", "var", "val"]], dip.dropna(subset=["val"])[["subject_id", "date", "var", "val"]]]

    names = ", ".join(f"'{n}'" for n in C.OMR_BP_NAMES + [C.OMR_BMI_NAME])
    omr = con.execute(f"""
        SELECT o.subject_id, CAST(o.chartdate AS DATE) AS date, o.result_name, o.result_value
        FROM omr o JOIN cand ON cand.subject_id = o.subject_id WHERE o.result_name IN ({names})""").df()
    bp = omr[omr["result_name"].isin(C.OMR_BP_NAMES)]
    sd = bp["result_value"].astype(str).str.split("/", expand=True)
    if sd.shape[1] >= 2:
        for col, var in ((0, "sbp"), (1, "dbp")):
            meas.append(pd.DataFrame({"subject_id": bp["subject_id"], "date": bp["date"], "var": var,
                                      "val": pd.to_numeric(sd[col], errors="coerce")}).dropna())
    bmi = omr[omr["result_name"] == C.OMR_BMI_NAME]
    meas.append(pd.DataFrame({"subject_id": bmi["subject_id"], "date": bmi["date"], "var": "bmi",
                              "val": pd.to_numeric(bmi["result_value"], errors="coerce")}).dropna())
    if pc.get("use_chartevents", True):
        cid = {i: "sbp" for i in C.CHART_SBP} | {i: "dbp" for i in C.CHART_DBP}
        ce = con.execute(f"""
            SELECT c.subject_id, CAST(c.charttime AS DATE) AS date, c.itemid, median(c.valuenum) AS val
            FROM chartevents c JOIN cand ON cand.subject_id = c.subject_id
            WHERE c.itemid IN ({", ".join(str(i) for i in cid)}) AND c.valuenum IS NOT NULL GROUP BY 1, 2, 3""").df()
        ce["var"] = ce["itemid"].map(cid)
        meas.append(ce[["subject_id", "date", "var", "val"]])

    meas = pd.concat(meas, ignore_index=True)
    meas["subject_id"] = meas["subject_id"].astype(np.int64)
    meas["date"] = pd.to_datetime(meas["date"])
    for var, (a, b) in C.VALUE_RANGES.items():
        sel = meas["var"] == var
        meas = meas[~sel | meas["val"].between(a, b)]
    meas["day"] = to_day(meas)
    ma = _attach(meas, vkeys, tol)
    wide = ma.groupby(["subject_id", "visit_day", "var"])["val"].median().unstack("var").reset_index()
    wide = wide.rename(columns={"visit_day": "day"})
    vis = vis.merge(wide, on=["subject_id", "day"], how="left")

    ev = []
    adm = con.execute("""
        SELECT a.subject_id, CAST(a.admittime AS DATE) AS adm_date, CAST(a.edregtime AS DATE) AS ed_date
        FROM admissions a JOIN cand ON cand.subject_id = a.subject_id""").df()
    ev.append(pd.DataFrame({"subject_id": adm["subject_id"], "date": adm["adm_date"], "var": "admission"}))
    ev.append(pd.DataFrame({"subject_id": adm["subject_id"], "date": adm["ed_date"], "var": "ed_visit"}).dropna())
    icu = con.execute("""
        SELECT s.subject_id, CAST(s.intime AS DATE) AS date FROM icustays s
        JOIN cand ON cand.subject_id = s.subject_id""").df()
    ev.append(pd.DataFrame({"subject_id": icu["subject_id"], "date": icu["date"], "var": "icu"}))
    aki = dx[_prefix_match(dx["icd_code"], dx["icd_version"], C.AKI_ICD9, C.AKI_ICD10)]
    ev.append(pd.DataFrame({"subject_id": aki["subject_id"], "date": aki["date"], "var": "aki"}))
    all_drugs = "|".join(C.DRUG_CLASSES.values())
    rx = con.execute(f"""
        SELECT r.subject_id, CAST(r.starttime AS DATE) AS date, lower(r.drug) AS drug
        FROM prescriptions r JOIN cand ON cand.subject_id = r.subject_id
        WHERE r.starttime IS NOT NULL AND regexp_matches(lower(r.drug), '{all_drugs}')""").df()
    rx["date"] = pd.to_datetime(rx["date"])
    for var, rgx in C.DRUG_CLASSES.items():
        d = rx[rx["drug"].str.contains(rgx, regex=True)].sort_values(["subject_id", "date"]).copy()
        prev = d.groupby("subject_id")["date"].shift()
        d = d[prev.isna() | ((d["date"] - prev).dt.days > cc["med_washout_days"])]
        ev.append(pd.DataFrame({"subject_id": d["subject_id"], "date": d["date"], "var": var}))
    ev = pd.concat(ev, ignore_index=True)
    ev["subject_id"] = ev["subject_id"].astype(np.int64)
    ev["date"] = pd.to_datetime(ev["date"])
    ev["day"] = to_day(ev)
    ea = _attach(ev, vkeys, None)
    ewide = (ea.groupby(["subject_id", "visit_day", "var"]).size().unstack("var") > 0).reset_index()
    ewide = ewide.rename(columns={"visit_day": "day"})
    vis = vis.merge(ewide, on=["subject_id", "day"], how="left", suffixes=("", "_ev"))

    dx["day"] = to_day(dx)
    for name, (p9, p10) in C.COMORBIDITY_CODES.items():
        first = dx.loc[_prefix_match(dx["icd_code"], dx["icd_version"], p9, p10)].groupby("subject_id")["day"].min()
        fd = vis["subject_id"].map(first)
        vis[name] = (vis["day"].values >= fd.values) & fd.notna().values
    vis = vis.sort_values(["subject_id", "day"]).reset_index(drop=True)

    n = len(vis)
    X = np.zeros((n, C.D_X), np.float32)
    M = np.zeros((n, C.D_X), np.uint8)
    for j, var in enumerate(C.X_VARS):
        if var in C.EVENT_VARS:
            flag = vis[var].fillna(False).astype(bool).values if var in vis else np.zeros(n, bool)
            X[:, j], M[:, j] = flag.astype(np.float32), flag.astype(np.uint8)
        elif var in C.COMORB_VARS:
            X[:, j], M[:, j] = vis[var].astype(np.float32).values, 1
        else:
            vals = vis[var].values.astype(np.float64) if var in vis else np.full(n, np.nan)
            ok = np.isfinite(vals)
            X[ok, j], M[ok, j] = vals[ok], 1
    setting = vis["setting"].values.astype(int)
    Cc = np.eye(C.D_C, dtype=np.uint8)[setting]

    meta = meta.set_index("subject_id", drop=False)
    sids = vis["subject_id"].values
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(sids)) + 1, [n]])
    days_all, egfr_all, age_all = vis["day"].values.astype(np.float64), vis["egfr"].values, vis["age"].values.astype(np.float64)

    store, rows, n_cand, n_eval = {}, [], 0, 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        sid = int(sids[a])
        m = meta.loc[sid]
        krt = None if pd.isna(m["krt_day"]) else float(m["krt_day"])
        cde = None if m["code_day"] is None or pd.isna(m["code_day"]) else float(m["code_day"])
        dth = None if m["death_day"] is None or pd.isna(m["death_day"]) else float(m["death_day"])
        r, fl = make_instances(days_all[a:b], egfr_all[a:b], age_all[a:b], m["elig_start"], krt, cde, dth, cc, th)
        n_cand += fl["candidate"]
        n_eval += fl["evaluable"]
        if not r:
            continue
        store[sid] = dict(t=days_all[a:b].astype(np.float32), x=X[a:b], m=M[a:b], c=Cc[a:b],
                          age=age_all[a:b].astype(np.float32), female=int(m["female"]))
        for x in r:
            x.update(subject_id=sid, female=int(m["female"]), group=m["group"],
                     diabetes_L=int(X[a + x["k"], C.DIABETES_COL]))
        rows.extend(r)
    inst = pd.DataFrame(rows)
    flow["no_eligible_landmark"] = int(len(meta) - n_cand)
    flow["no_evaluable_outcome_window"] = int(n_cand - n_eval)
    flow["patients_final"] = int(inst["subject_id"].nunique())
    flow["instances_final"] = int(len(inst))

    inst = add_subgroup_columns(inst)
    flow["rkfd_prevalence_primary"] = float(inst.loc[inst["eligible_primary"] == 1, "y_slope5"].mean())

    inst.to_parquet(out_dir / "instances.parquet")
    meta.reset_index(drop=True).to_parquet(out_dir / "patients_meta.parquet")
    with open(out_dir / "store.pkl", "wb") as f:
        pickle.dump(store, f, protocol=4)
    save_json(flow, out_dir / "flow.json")
    return inst, store, meta, flow
