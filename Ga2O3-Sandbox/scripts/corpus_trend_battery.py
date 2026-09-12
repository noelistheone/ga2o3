"""Corpus trend battery — the DISCRIMINATING statistical validation the n=3 lab test cannot give.

Tests the forward chain against the curated corpus on WITHIN-condition contrasts only (never
pooled absolutes — the lab-confound the sandbox claims to sidestep), scored against permutation
-null floors (Phase-101/104 discipline). The discriminating power is in the transport data:
donors raise n, acceptors lower it — a real sign test the chain's donor/acceptor physics must pass.

Tests (each vs a permutation floor of the same statistic on shuffled predictions):
  T1  cross-dopant carrier-density RANKING: Spearman(chain predicted n vs corpus median n) over
      dopants with transport data. Tests the donor/acceptor/deep classification directly.
  T2  within-paper doped-vs-reference n SIGN test (transport corpus).
  T3  within-paper doped-vs-undoped PDR SIGN test (device corpus) — with the upward-bias caveat.
Only dopants in KROGER's 19 are testable (others need Tier-4 energetics).
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward as ff, kroger_db  # noqa: E402

db = kroger_db.load()
KROGER = set(db.elements)  # Ga O + 17 dopants
PROC = dict(T_anneal=1073.0, pO2=1e-5, EcT_fraction=0.40, film=True, E_B_eV=0.03, Nd_bg=1e17)
rng = np.random.default_rng(0)

# ── forward predictions per dopant (cache) ───────────────────────────────────
und = ff.forward("undoped", 0.0, db=db, **PROC)
cache = {"n_und": und["hall_n_cm3"]}
pred = {}
for X in [e for e in db.elements if e not in ("Ga", "O")]:
    try:
        r = ff.forward(X, 0.01, db=db, undoped_cache=cache, **PROC)
        pred[X] = {"n": r["hall_n_cm3"], "mu": r["hall_mu_cm2Vs"], "sigma": r["sigma_S_cm"],
                   "PDR_score": r["PDR_score"], "dark_act": r["dark_activation_eV"]}
    except Exception:
        pass


def perm_floor(stat_fn, n_perm=2000):
    vals = [stat_fn(shuffle=True) for _ in range(n_perm)]
    return np.array(vals)


# ── T1: cross-dopant carrier-density ranking ─────────────────────────────────
tp = pd.read_csv(NET / "results/phase65/transport_llm_extracted_v2.csv")
tp = tp[tp["carrier_cm3"].notna()]
# median corpus n per dopant (single-dopant rows in KROGER-19)
corpus_n = {}
for el, g in tp.groupby("dopant_element"):
    el = str(el)
    if el in KROGER and el not in ("Ga", "O") and el in pred:
        corpus_n[el] = float(np.median(np.log10(g["carrier_cm3"].astype(float).clip(lower=1))))
common = [e for e in corpus_n if e in pred]
T1 = {}
if len(common) >= 4:
    x = np.array([np.log10(max(pred[e]["n"], 1.0)) for e in common])
    y = np.array([corpus_n[e] for e in common])
    rho = float(spearmanr(x, y).correlation)
    null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(5000)])
    p = float((np.abs(null) >= abs(rho)).mean())
    T1 = {"dopants": common, "n_dopants": len(common), "spearman_rho": round(rho, 3),
          "perm_p": round(p, 4), "beats_floor_p05": bool(p < 0.05),
          "chain_log10n": {e: round(float(np.log10(max(pred[e]['n'], 1))), 1) for e in common},
          "corpus_median_log10n": {e: round(corpus_n[e], 1) for e in common}}

# ── T2: within-paper n sign (transport corpus, same doi, doped vs lower-n reference) ─
t2_hits = t2_tot = 0
t2_detail = []
for doi, g in tp.groupby("doi"):
    els = [str(e) for e in g["dopant_element"].dropna().unique()]
    ku = [e for e in els if e in KROGER and e not in ("Ga", "O") and e in pred]
    has_ref = g["dopant_element"].isna().any() or ("undoped" in [str(e).lower() for e in els])
    # compare each KROGER dopant's corpus n vs the paper's min n (reference proxy)
    if not ku:
        continue
    ref_n = float(np.log10(max(g["carrier_cm3"].astype(float).min(), 1)))
    for e in ku:
        ge = g[g["dopant_element"].astype(str) == e]
        obs = float(np.log10(max(ge["carrier_cm3"].astype(float).median(), 1))) - ref_n
        prd = np.log10(max(pred[e]["n"], 1.0)) - np.log10(max(und["hall_n_cm3"], 1.0))
        if abs(obs) < 0.1:
            continue
        t2_tot += 1
        t2_hits += int(np.sign(obs) == np.sign(prd))
        t2_detail.append({"doi": doi[:30], "el": e, "obs_dsign": int(np.sign(obs)),
                          "pred_dsign": int(np.sign(prd))})
T2 = {"n_contrasts": t2_tot, "correct_sign": t2_hits,
      "frac": round(t2_hits / t2_tot, 3) if t2_tot else None}
if t2_tot:
    # binomial permutation floor
    null = rng.binomial(t2_tot, 0.5, 20000) / t2_tot
    T2["perm_p"] = round(float((null >= t2_hits / t2_tot).mean()), 4)
    T2["beats_floor_p05"] = bool(T2["perm_p"] < 0.05)

# ── T3: within-paper PDR doped-vs-undoped sign (device corpus) ────────────────
dv = pd.read_csv(NET / "data/raw/experimental/ga2o3_exp.csv")
dv = dv[dv["usable_flag"].fillna(1) != 0]
t3_hits = t3_tot = 0
t3_obs_signs = []
for doi, g in dv.groupby("doi"):
    und_rows = g[g["element"].astype(str).str.lower() == "undoped"]
    if und_rows.empty or und_rows["photo_dark_ratio"].isna().all():
        continue
    pdr_und = float(und_rows["photo_dark_ratio"].astype(float).median())
    for _, row in g.iterrows():
        el = str(row["element"])
        if el.lower() == "undoped" or el not in pred or pd.isna(row["photo_dark_ratio"]):
            continue
        obs = float(row["photo_dark_ratio"]) - pdr_und
        prd = pred[el]["PDR_score"] - und["PDR_score"]
        if abs(obs) < 0.05:
            continue
        t3_tot += 1
        t3_hits += int(np.sign(obs) == np.sign(prd))
        t3_obs_signs.append(int(np.sign(obs)))
T3 = {"n_contrasts": t3_tot, "correct_sign": t3_hits,
      "frac": round(t3_hits / t3_tot, 3) if t3_tot else None}
if t3_tot:
    null = rng.binomial(t3_tot, 0.5, 20000) / t3_tot
    T3["perm_p_vs_coinflip"] = round(float((null >= t3_hits / t3_tot).mean()), 4)
    # base-rate: fraction of corpus contrasts that are 'up' → the majority-class baseline
    frac_up = float(np.mean(np.array(t3_obs_signs) > 0))
    majority = max(frac_up, 1 - frac_up)
    T3["corpus_frac_up"] = round(frac_up, 3)
    T3["majority_class_baseline"] = round(majority, 3)
    T3["beats_majority_baseline"] = bool(t3_hits / t3_tot > majority)
    T3["verdict"] = ("PDR skill CONFOUNDED by base rate — chain accuracy ≈ always-predict-"
                     "majority; not demonstrated skill" if t3_hits / t3_tot <= majority + 1e-9
                     else "PDR beats the majority-class baseline (genuine sign skill)")

report = {"process_params": PROC,
          "T1_cross_dopant_n_ranking": T1,
          "T2_within_paper_n_sign": T2,
          "T3_within_paper_PDR_sign": T3,
          "testable_dopants_in_KROGER": sorted([e for e in pred]),
          "honest_frame": "within-condition contrasts only; pooled absolutes banned (lab "
                          "confound). T1 (cross-dopant n ranking) is the strongest discriminating "
                          "test — it probes the donor/acceptor/deep classification directly."}
(PROJ / "results/tier2/corpus_trend_battery.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
