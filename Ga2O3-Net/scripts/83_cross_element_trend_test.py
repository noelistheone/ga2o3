"""V62 / research C1 — CROSS-ELEMENT trend breakthrough test (leave-one-ELEMENT-out CV).

The research (Part C1) says the biggest documented cross-element payoff is replacing one-hot dopant tokens
with PHYSICAL/CHEMICAL encodings (one-hot degrades ~39.6% when an element is withheld vs ~8.7-10.6% for
learned/physical embeddings), and that the mandatory bar to beat is a tuned tree (RF/XGBoost). This script
runs exactly that, leakage-free, on the project's own data.

Question: given process features (T, log c, log pO2, method), does the ELEMENT representation let a model
extrapolate a dopant's property to an element held out of training? We compare:
  - process_only         : no element info (lower bound)
  - onehot               : one-hot element id (information-less for an UNSEEN element)
  - phys5                : 5-vec physical descriptors (electronegativity, ionic radius, carrier sign, dv, dperiod)
  - mace256              : frozen MACE-MP-0 dopant-site features (cached), the "decoupled backbone" route
each with Ridge (decoupled linear), RandomForest and XGBoost (the baseline GATE).

Protocol: leave-one-element-out over doped single-element rows with >= MIN_ROWS; pooled OOD R^2 / Spearman /
MAE on held-out predictions. TabPFN intentionally EXCLUDED (project licensing non-goal); RF/XGBoost serve as
the required tree gate. Writes results/phase62_codoping/cross_element_trend.json. CPU, deterministic.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.data.dcwm_transitions import _parse_conc_frac, atmosphere_to_key, method_to_idx, ATM_PO2  # noqa: E402
from src.data.element_descriptors import ELEMENT_DESCRIPTORS  # noqa: E402

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb

CSV = PROJ / "data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v56.csv"
MIN_ROWS = 4
_MACE = np.load(PROJ / "data/processed/mace_mp_features.npz", allow_pickle=True)
_MACE_EL = {str(e): _MACE["mean_dop"][i] for i, e in enumerate(_MACE["elements"])}
_PHYS_DIM = len(next(iter(ELEMENT_DESCRIPTORS.values())))


def load(target):
    df = pd.read_csv(CSV)
    y = pd.to_numeric(df[target], errors="coerce")
    df = df[y.notna()].reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        e = str(r.get("element") or "undoped").strip()
        if e in ("—", "-", "nan", ""):
            e = "undoped"
        T_raw = pd.to_numeric(r.get("temperature_C"), errors="coerce")
        rows.append(dict(
            elem=e, y=float(pd.to_numeric(r[target], errors="coerce")),
            c=_parse_conc_frac(r),
            T=float(T_raw) if pd.notna(T_raw) else 25.0,
            lpO2=math.log10(ATM_PO2[atmosphere_to_key(str(r.get("atmosphere")))]),
            method=method_to_idx(str(r.get("method")))))
    return pd.DataFrame(rows)


def proc_feats(d):
    m = np.eye(6)[np.clip(d["method"].to_numpy(), 0, 5)]
    base = np.column_stack([d["T"].to_numpy(),
                            np.log10(np.clip(d["c"].to_numpy(), 1e-6, None)),
                            d["lpO2"].to_numpy()])
    return np.hstack([base, m])


def elem_feats(d, rep):
    els = d["elem"].tolist()
    if rep == "process_only":
        return np.zeros((len(els), 0))
    if rep == "onehot":
        uniq = sorted(set(els)); idx = {e: i for i, e in enumerate(uniq)}
        M = np.zeros((len(els), len(uniq)))
        for i, e in enumerate(els):
            M[i, idx[e]] = 1.0
        return M
    if rep == "phys5":
        return np.array([ELEMENT_DESCRIPTORS.get(e, (0.0,) * _PHYS_DIM) for e in els])
    if rep == "mace256":
        dim = _MACE["mean_dop"].shape[1]
        return np.array([_MACE_EL[e] if e in _MACE_EL else np.zeros(dim) for e in els])
    raise ValueError(rep)


def models():
    return {
        "ridge": lambda: Ridge(alpha=1.0),
        "rf": lambda: RandomForestRegressor(n_estimators=400, max_depth=6, random_state=0, n_jobs=-1),
        "xgb": lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1),
    }


def run_target(target):
    d = load(target)
    # single-element doped rows eligible to be held out (exclude '+' multi-element + undoped)
    doped = d[(~d["elem"].str.contains(r"\+")) & (d["elem"] != "undoped")]
    counts = doped["elem"].value_counts()
    hold_elems = [e for e in counts.index if counts[e] >= MIN_ROWS]
    Xp = proc_feats(d); y = d["y"].to_numpy(); els = d["elem"].to_numpy()
    out = {"target": target, "n_rows": len(d), "n_hold_elements": len(hold_elems),
           "hold_elements": hold_elems, "MIN_ROWS": MIN_ROWS, "reps": {}}
    mace_cov = [e for e in hold_elems if e in _MACE_EL]
    out["mace_covered_hold_elements"] = mace_cov
    for rep in ["process_only", "onehot", "phys5", "mace256"]:
        Xe = elem_feats(d, rep)
        X = np.hstack([Xp, Xe])
        for mname, mk in models().items():
            preds, truth = [], []
            for he in hold_elems:
                if rep == "mace256" and he not in _MACE_EL:
                    continue  # MACE can't represent this element (e.g. N, H) -> skip fold for fairness
                tr = els != he
                te = (els == he)
                sc = StandardScaler().fit(X[tr])
                Xtr = np.nan_to_num(sc.transform(X[tr])); Xte = np.nan_to_num(sc.transform(X[te]))
                mdl = mk(); mdl.fit(Xtr, y[tr])
                preds.extend(mdl.predict(Xte).tolist()); truth.extend(y[te].tolist())
            if len(truth) < 3:
                continue
            preds = np.array(preds); truth = np.array(truth)
            ss_res = float(np.sum((truth - preds) ** 2))
            ss_tot = float(np.sum((truth - truth.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rho = float(spearmanr(truth, preds).correlation) if len(truth) > 2 else float("nan")
            mae = float(np.mean(np.abs(truth - preds)))
            out["reps"].setdefault(rep, {})[mname] = dict(
                OOD_R2=round(r2, 4), OOD_spearman=round(rho, 4), OOD_MAE=round(mae, 4), n=len(truth))
    return out


def main():
    rep = {t: run_target(t) for t in ["photo_dark_ratio", "vacancy_concentration"]}
    outp = PROJ / "results/phase62_codoping/cross_element_trend.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, indent=2))
    for t, r in rep.items():
        print(f"\n=== {t}  (leave-one-element-out; {r['n_hold_elements']} held elements: {r['hold_elements']}) ===")
        print(f"{'representation':14s} {'model':6s} {'OOD_R2':>8s} {'spearman':>9s} {'MAE':>7s} {'n':>4s}")
        for repname, mm in r["reps"].items():
            for mname, v in mm.items():
                print(f"{repname:14s} {mname:6s} {v['OOD_R2']:8.3f} {v['OOD_spearman']:9.3f} {v['OOD_MAE']:7.3f} {v['n']:4d}")
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
