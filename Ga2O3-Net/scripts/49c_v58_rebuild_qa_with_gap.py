"""Phase 58 — rebuild Stage-1 QA corpus, injecting the REAL per-dopant V_O charge state
(from the 82 converged gap calcs) into the 10 gap dopants' QA pairs.

Wraps scripts/41_v58_kroger_to_qa.py UNCHANGED (imported via importlib because the
filename starts with a digit) — the 4 honest sources (kroger / dft_cache / literature /
tier1c) are built exactly as before, then every pair whose dopant is one of the 10 gap
dopants {Al,B,Cr,Cu,Fe,Ni,Ti,V,W,Zn} is post-processed:

  • label.dominant_charge_state  <- the REAL per-dopant dominant V_O charge at the n-type
    host εF (4.0 eV above VBM), decided by the DFT transition levels ε(0/+1), ε(+1/+2)
    (VBM-realigned, from dft/qe_hse06/results/v58_gap_dopant_summary.csv via the same
    realignment build_dft_cache + 49b apply). e.g. if ε(+1/+2) < 4.0 the +2 state is
    stable; here all 10 sit with the +1/0 boundary (ε(0/+1)) around 3.9-4.4 eV, so the
    real dominant state is +1 (or 0 for the shallowest, Al/B) — NOT the blanket +2 the
    closed-form n-type Brouwer template assigned.
  • label.reasoning  <- a transition-level-aware clause is appended naming the real
    ε(0/+1), ε(+1/+2) and the resulting dominant charge.

The CORE label log_vo_predicted is UNCHANGED (anchor-based; absolute per-dopant ΔE_f is
NOT extractable from doped V_O cells — documented trap). log_K_eq / delta_E_f_eV are left
as the source template produced them (they derive from the native ΔE_f baseline, which we
do not claim to improve per-dopant). Only the charge-state + reasoning carry the new,
reference-independent DFT information.

Keeps the ~8200 total count and the 9:1 split (same RNG seed 58 as the original).
Outputs (overwrites; .bak_pre_gap backups already taken):
  data/processed/stage1_qa_pairs.jsonl   (train)
  data/processed/stage1_qa_pairs_val.jsonl (val)

GPU-over-CPU note: scalar/string post-processing on CPU; no GPU path applies.
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
SUMMARY_CSV = PROJ / "dft" / "qe_hse06" / "results" / "v58_gap_dopant_summary.csv"
GAP_DOPANTS = ["Al", "B", "Cr", "Cu", "Fe", "Ni", "Ti", "V", "W", "Zn"]

# Import the unchanged Stage-1 QA builder module (filename starts with a digit).
_spec = importlib.util.spec_from_file_location("qa41", PROJ / "scripts" / "41_v58_kroger_to_qa.py")
QA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(QA)

from scripts.build_dft_cache import EPS_0_1, ANCHOR_FERMI_EV  # native baseline + host εF

_CHARGE_SYM = {0: "neutral V_O^0", 1: "singly-charged V_O^+1", 2: "doubly-charged V_O^2+"}


def _load_realigned_levels() -> dict:
    """Per-dopant VBM-realigned ε(0/+1), ε(+1/+2) + dominant charge at εF=4.0.

    Realignment IDENTICAL to scripts/49b_v58_enhance_cache.py: subtract a single common
    offset so the cohort-mean raw ε(0/+1) matches the native EPS_0_1 baseline. This keeps
    the QA charge calls consistent with the cache dims 5-6.
    """
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["dopant"].isin(GAP_DOPANTS)].copy()
    raw01 = df.set_index("dopant")["eps_0_1_eV"].to_dict()
    raw12 = df.set_index("dopant")["eps_1_2_eV"].to_dict()
    dom_site = df.set_index("dopant")["dominant_site"].to_dict()
    delta_vbm = float(np.nanmean(list(raw01.values()))) - EPS_0_1
    out = {}
    for d in GAP_DOPANTS:
        e01 = raw01[d] - delta_vbm
        e12 = raw12[d] - delta_vbm
        if not np.isnan(e12) and ANCHOR_FERMI_EV < e12:
            q = 2
        elif not np.isnan(e01) and ANCHOR_FERMI_EV < e01:
            q = 1
        else:
            q = 0
        out[d] = {"eps01": e01, "eps12": e12, "q": int(q),
                  "site": dom_site.get(d, "—")}
    return out, delta_vbm


def _gap_clause(elem: str, lv: dict) -> str:
    return (f" From the V58 gap-dopant HSE/PBE+U calcs, {elem} sets V_O charge-transition "
            f"levels epsilon(0/+1) approximately {lv['eps01']:.2f} eV and epsilon(+1/+2) "
            f"approximately {lv['eps12']:.2f} eV above the valence-band maximum (dominant "
            f"V_O site {lv['site']}); with the n-type host Fermi level near {ANCHOR_FERMI_EV:.1f} eV "
            f"this makes the {_CHARGE_SYM[lv['q']]} the dominant charge state for {elem}.")


def _patch_pair(p: dict, levels: dict) -> bool:
    """Override dominant_charge_state + append transition clause for gap-dopant pairs.
    Returns True if patched."""
    elem = p["input"]["dopant"]
    if elem not in levels:
        return False
    lv = levels[elem]
    p["label"]["dominant_charge_state"] = lv["q"]
    # append the DFT transition clause (idempotent: only once)
    if "charge-transition levels" not in p["label"]["reasoning"]:
        p["label"]["reasoning"] = p["label"]["reasoning"].rstrip() + _gap_clause(elem, lv)
    p["_gap_charge_real"] = True
    return True


def main():
    import os
    os.chdir(PROJ)
    rng = np.random.default_rng(58)  # SAME seed as the original for count/split parity

    levels, delta_vbm = _load_realigned_levels()
    print("Per-dopant REAL transition levels (VBM-realigned, Δ_VBM=%.3f eV):" % delta_vbm)
    for d in GAP_DOPANTS:
        lv = levels[d]
        print(f"  {d:3s} site={lv['site']:6s} eps(0/+1)={lv['eps01']:.3f} "
              f"eps(+1/+2)={lv['eps12']:.3f}  -> dominant q@eF4.0 = {lv['q']}")

    print("\nBuilding Stage-1 QA (same 4 sources, RNG seed 58)...")
    kroger = QA.build_kroger(rng, per_dopant=240)
    print(f"  source 1 KROGER-derived:        {len(kroger)}")
    dft = QA.build_dft(rng, target=1500)
    print(f"  source 2 DFT-cache-derived:     {len(dft)}")
    lit = QA.build_literature(rng, target=750)
    print(f"  source 3 reference-literature:  {len(lit)}")
    tier1c, dropped = QA.build_tier1c(rng, per_dopant=100)
    print(f"  source 4 Tier-1C synthetic:     {len(tier1c)}  (dropped {dropped})")

    all_pairs = kroger + dft + lit + tier1c
    print(f"  TOTAL before split:             {len(all_pairs)}")

    # ---- patch gap-dopant pairs across ALL sources ----
    n_patched = 0
    patched_by_source = Counter()
    patched_by_dopant = Counter()
    charge_change = Counter()  # (old_q, new_q)
    for p in all_pairs:
        old_q = p["label"]["dominant_charge_state"]
        if _patch_pair(p, levels):
            n_patched += 1
            patched_by_source[p["_source"]] += 1
            patched_by_dopant[p["input"]["dopant"]] += 1
            charge_change[(old_q, p["label"]["dominant_charge_state"])] += 1

    print(f"\n[gap-charge injection] {n_patched} pairs got REAL per-dopant charge info")
    print("  by source: " + ", ".join(f"{s}={n}" for s, n in sorted(patched_by_source.items())))
    print("  by dopant: " + ", ".join(f"{d}={n}" for d, n in sorted(patched_by_dopant.items())))
    print("  dominant-charge changes (old->new : count):")
    for (oq, nq), n in sorted(charge_change.items()):
        tag = "  (CHANGED)" if oq != nq else ""
        print(f"    q{oq} -> q{nq} : {n}{tag}")

    # strip the internal flag before writing (keep schema clean: input/label/_source)
    for p in all_pairs:
        p.pop("_gap_charge_real", None)

    # ---- verify (reuse the original module's gate) ----
    rep = QA.verify(all_pairs)
    print("\n=== VERIFY ===")
    print(f"  log_vo range: [{rep['log_vo_min']:.2f}, {rep['log_vo_max']:.2f}]  "
          f"mean {rep['log_vo_mean']:.2f}  NaN {rep['n_nan']}")
    print(f"  JSON/range/NaN issues: {len(rep['issues'])}")
    for s in rep["issues"][:5]:
        print("     -", s)

    srcs = Counter(p["_source"] for p in all_pairs)
    print("\n  per-source breakdown:")
    for s, n in sorted(srcs.items()):
        print(f"    {s:12s} {n}")

    # ---- 9:1 split (same scheme as original) ----
    rng.shuffle(all_pairs)
    n_val = round(len(all_pairs) * 0.1)
    val = all_pairs[:n_val]
    train = all_pairs[n_val:]

    with open(QA.OUT_TRAIN, "w") as f:
        for p in train:
            f.write(json.dumps(p) + "\n")
    with open(QA.OUT_VAL, "w") as f:
        for p in val:
            f.write(json.dumps(p) + "\n")
    print(f"\n  wrote {len(train)} train -> {QA.OUT_TRAIN}")
    print(f"  wrote {len(val)} val   -> {QA.OUT_VAL}")
    print(f"  TOTAL {len(all_pairs)} pairs  (9:1 split)")

    # ---- metrics-from-disk: re-read written files, count real-charge pairs ----
    def _count_gap(path):
        n_gap = n_q = 0
        gap_q = Counter()
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d["input"]["dopant"] in levels:
                    n_gap += 1
                    gap_q[d["label"]["dominant_charge_state"]] += 1
                n_q += 1
        return n_gap, n_q, gap_q
    tg, tt, tq = _count_gap(QA.OUT_TRAIN)
    vg, vt, vq = _count_gap(QA.OUT_VAL)
    print("\n=== RE-READ from disk (metrics-from-disk) ===")
    print(f"  train: {tt} pairs, {tg} gap-dopant; gap dominant-charge dist = {dict(tq)}")
    print(f"  val:   {vt} pairs, {vg} gap-dopant; gap dominant-charge dist = {dict(vq)}")
    print(f"  TOTAL on disk: {tt + vt} pairs, {tg + vg} gap-dopant with real charge info")


if __name__ == "__main__":
    main()
