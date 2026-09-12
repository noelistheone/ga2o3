"""Freeze-in-T calibration — can ONE global effective freeze-in T give realistic ABSOLUTE n
across dopants? (The user's absolute-accuracy push.) Anti-overfitting: fit the single T on a
TRAIN split of dopants, report the held-out TEST error. If test RMS(log n) is small, absolute n
is recoverable with one calibrated knob; if it stays large, absolutes are energetics-capped even
with calibration (only per-dopant fitting would match = overfitting).
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward as ff, kroger_db  # noqa: E402

db = kroger_db.load()
KROGER = set(db.elements)

tp = pd.read_csv(NET / "results/phase65/transport_llm_extracted_v2.csv")
tp = tp[tp["carrier_cm3"].notna()]
corpus_n = {}
for el, g in tp.groupby("dopant_element"):
    el = str(el)
    if el in KROGER and el not in ("Ga", "O"):
        corpus_n[el] = float(np.median(np.log10(g["carrier_cm3"].astype(float).clip(lower=1))))

T_grid = [1073, 1200, 1350, 1500, 1650, 1800, 1950]
# predict log n per dopant per freeze-in T (pO2 fixed at 1e-5, 1 at%)
pred = {T: {} for T in T_grid}
for T in T_grid:
    proc = dict(T_anneal=float(T), pO2=1e-5, EcT_fraction=0.40, film=True, E_B_eV=0.03, Nd_bg=1e17)
    und = ff.forward("undoped", 0.0, db=db, **proc)
    cache = {"n_und": und["hall_n_cm3"]}
    for X in corpus_n:
        try:
            r = ff.forward(X, 0.01, db=db, undoped_cache=cache, **proc)
            pred[T][X] = np.log10(max(r["hall_n_cm3"], 1.0))
        except Exception:
            pass

dopants = sorted([e for e in corpus_n if all(e in pred[T] for T in T_grid)])
# train/test split (deterministic: alternate)
train = dopants[::2]
test = dopants[1::2]


def rms_logerr(Tf, subset):
    errs = [pred[Tf][e] - corpus_n[e] for e in subset]
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else None


# best global Tf on TRAIN
train_rms = {T: rms_logerr(T, train) for T in T_grid}
best_T = min(train_rms, key=train_rms.get)
test_rms = rms_logerr(best_T, test)
all_rms = rms_logerr(best_T, dopants)

# per-dopant at best T
per_dopant = {e: {"pred_log_n": round(pred[best_T][e], 2), "corpus_log_n": round(corpus_n[e], 2),
                  "err": round(pred[best_T][e] - corpus_n[e], 2)} for e in dopants}

# DONOR-SUBSET calibration: for the predictable shallow-donor subclass, how good can absolute n get?
DONORS = [e for e in ["Si", "Sn", "Ge", "Zr", "Hf", "Ta"] if e in dopants]
donor_train = DONORS[::2]
donor_test = DONORS[1::2]
donor_train_rms = {T: rms_logerr(T, donor_train) for T in T_grid}
donor_best_T = min(donor_train_rms, key=donor_train_rms.get)
donor_test_rms = rms_logerr(donor_best_T, donor_test)
donor_all_rms = rms_logerr(donor_best_T, DONORS)
donor_detail = {e: {"pred_log_n": round(pred[donor_best_T][e], 2),
                    "corpus_log_n": round(corpus_n[e], 2),
                    "err": round(pred[donor_best_T][e] - corpus_n[e], 2)} for e in DONORS}

report = {
    "donor_subset_calibration": {
        "donors": DONORS, "best_freeze_in_T": donor_best_T,
        "train_rms_by_T": {str(T): round(v, 2) for T, v in donor_train_rms.items()},
        "train_rms_log10n": round(donor_train_rms[donor_best_T], 2),
        "TEST_rms_log10n_heldout": round(donor_test_rms, 2) if donor_test_rms else None,
        "all_donor_rms_log10n": round(donor_all_rms, 2),
        "per_donor": donor_detail,
        "verdict": ("for the shallow-donor subclass, a calibrated freeze-in T gives absolute n "
                    "within ~%.1f orders (held-out) — the predictable-subclass absolute accuracy"
                    % donor_test_rms if donor_test_rms else "n/a")},
    "n_dopants": len(dopants), "dopants": dopants,
    "train_dopants": train, "test_dopants": test,
    "train_rms_by_T": {str(T): round(v, 2) for T, v in train_rms.items()},
    "best_freeze_in_T_on_train": best_T,
    "train_rms_log10n": round(train_rms[best_T], 2),
    "TEST_rms_log10n_heldout": round(test_rms, 2) if test_rms else None,
    "all_rms_log10n": round(all_rms, 2),
    "per_dopant_at_best_T": per_dopant,
    "interpretation": ("one calibrated freeze-in T gives absolute n within ~%.1f orders on "
                       "held-out dopants → absolute n IS calibratable" % test_rms
                       if test_rms and test_rms < 1.0 else
                       "held-out RMS(log n) = %.1f orders → one global freeze-in T does NOT give "
                       "absolute n; only per-dopant fitting would (overfitting). Absolutes remain "
                       "energetics-capped; the freeze-in knob sets the overall SCALE not per-dopant "
                       "values" % (test_rms if test_rms else -1)),
}
(PROJ / "results/tier2/freeze_in_calibration.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "per_dopant_at_best_T"}, indent=2))
print("\nper-dopant at best T:")
for e, d in per_dopant.items():
    print(f"  {e:3s} pred={d['pred_log_n']:+6.2f} corpus={d['corpus_log_n']:+6.2f} err={d['err']:+.2f}"
          f" {'[train]' if e in train else '[TEST]'}")
