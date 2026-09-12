"""Phase 58 — enhance dft_features_cache_v58.npz dims 5-6 with PER-DOPANT V_O transition
levels from the 82 converged gap-dopant calcs (scripts/49_v58_parse_gap_calcs.py).

What changes / what STAYS (verify-correctness rule):
  • Dims 0-2 (log_vo anchor ΔE_f^{0,1,2})  → UNCHANGED. The absolute per-dopant ΔE_f is
    NOT extractable from the doped V_O cells (no doped-no-vacancy reference on disk); they
    stay the native-V_O + Reuter-Scheffler μ_O baseline. Atmosphere/T monotonicity lives
    entirely in dims 0-2 + dim 7, so it is UNAFFECTED by this enhancement.
  • Dims 5-6 (transition levels ε(0/+1), ε(+1/+2)) → REPLACED by per-dopant DFT values for
    the 10 gap dopants (Al,B,Cr,Cu,Fe,Ni,Ti,V,W,Zn); all other dopants keep the native
    EPS_0_1 / EPS_1_2 baseline.

VBM-realignment (the physically-honest correction — see docs/phase58_lite_dft_finding.md
and build_dft_cache.py docstring "≈8 eV > the 4.85 eV gap" callout):
  The RAW per-dopant transition levels from the PBE+U gap calcs come out ~8 eV above the
  nominal VBM=0 placeholder — i.e. ABOVE the 4.85 eV gap — because those calcs carry no
  band-edge reference and no Freysoldt potential-alignment ΔV_PCA (the doped-no-vacancy
  cell that would fix the VBM is the very reference we do not have). The ABSOLUTE placement
  is therefore NOT physical. What IS reference-independent and physical:
     (i)  the level SPACING  ε(0/+1) − ε(+1/+2)  (negative-U sign), and
     (ii) the PER-DOPANT relative ordering of the levels.
  We preserve both by subtracting a SINGLE common offset Δ_VBM chosen so the cohort-mean
  ε(0/+1) of the gap dopants equals the native in-gap baseline EPS_0_1. This keeps every
  per-dopant deviation and the spacing exactly as the DFT gives them, while pinning the
  absolute scale to the physical in-gap window. (A per-dopant offset would erase the
  reference-independent per-dopant structure — the only thing we trust — so we do NOT do
  that.) Both raw and realigned values are recorded in the cache meta for traceability.

Outputs: overwrites data/processed/dft_features_cache_v58.npz (backup .bak_pre_gap exists).
Re-reads it via build_dft_cache.load_cache and re-checks Mg atmosphere/T monotonicity.

GPU-over-CPU note: scalar assembly + one logsumexp check on CPU; no GPU path applies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_dft_cache import (
    load_cache, _anchor_log_vo, EPS_0_1, EPS_1_2, ANCHOR_FERMI_EV,
)
from scripts.build_dft_cache import build_cache  # ensure baseline cache exists / rebuildable

PROJ = Path(__file__).resolve().parents[1]
CACHE = PROJ / "data" / "processed" / "dft_features_cache_v58.npz"
SUMMARY_CSV = PROJ / "dft" / "qe_hse06" / "results" / "v58_gap_dopant_summary.csv"
GAP_DOPANTS = ["Al", "B", "Cr", "Cu", "Fe", "Ni", "Ti", "V", "W", "Zn"]


def dominant_charge_at_eF(eps01, eps12, eF=ANCHOR_FERMI_EV):
    if not np.isnan(eps12) and eF < eps12:
        return 2
    if not np.isnan(eps01) and eF < eps01:
        return 1
    return 0


def main():
    if not CACHE.exists():
        print("[info] baseline cache missing — building it first via build_dft_cache")
        build_cache()

    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["dopant"].isin(GAP_DOPANTS)].copy()

    raw01 = df.set_index("dopant")["eps_0_1_eV"].to_dict()
    raw12 = df.set_index("dopant")["eps_1_2_eV"].to_dict()
    dom_site = df.set_index("dopant")["dominant_site"].to_dict()

    # ---- single common VBM-realignment offset: cohort-mean ε(0/+1) → native EPS_0_1 ----
    mean_raw01 = float(np.nanmean(list(raw01.values())))
    delta_vbm = mean_raw01 - EPS_0_1   # subtract this from every raw level
    realigned01 = {d: raw01[d] - delta_vbm for d in raw01}
    realigned12 = {d: raw12[d] - delta_vbm for d in raw12}

    print(f"[realign] cohort-mean raw ε(0/+1) = {mean_raw01:.3f} eV; native EPS_0_1 = "
          f"{EPS_0_1:.3f} eV  ->  Δ_VBM = {delta_vbm:.3f} eV (subtracted from all levels)")
    print(f"[baseline] native EPS_0_1={EPS_0_1:.4f}  EPS_1_2={EPS_1_2:.4f} (kept for "
          f"non-gap dopants + dims 0-2 anchor unchanged)\n")

    # ---- load baseline cache, patch dims 5-6 for gap dopants ----
    idx, feats, meta = load_cache()
    feats = feats.copy()
    keys = list(idx.keys())

    per_dopant_meta = {}
    n_rows_patched = 0
    for key in keys:
        dop = key.split("|")[0]
        if dop in realigned01:
            feats[idx[key]][5] = realigned01[dop]
            feats[idx[key]][6] = realigned12[dop]
            n_rows_patched += 1

    for d in GAP_DOPANTS:
        per_dopant_meta[d] = {
            "dominant_site": dom_site.get(d, "—"),
            "eps_0_1_raw_eV": round(float(raw01[d]), 4),
            "eps_1_2_raw_eV": round(float(raw12[d]), 4),
            "eps_0_1_realigned_eV": round(float(realigned01[d]), 4),
            "eps_1_2_realigned_eV": round(float(realigned12[d]), 4),
            "dominant_charge_eF4.0": dominant_charge_at_eF(realigned01[d], realigned12[d]),
            "negative_U_spacing_eV": round(float(raw01[d] - raw12[d]), 4),
        }

    # ---- update meta (document the limitation + provenance) ----
    meta["gap_dopant_transitions"] = {
        "source": "scripts/49_v58_parse_gap_calcs.py (82/90 converged PBEU_v58 gap calcs)",
        "dopants": GAP_DOPANTS,
        "dims_5_6_per_dopant": True,
        "vbm_realignment_eV": round(delta_vbm, 4),
        "realignment_note": (
            "Dims 5-6 for these 10 dopants are the DFT charge-transition levels ε(0/+1), "
            "ε(+1/+2), VBM-realigned by a single common offset so the cohort-mean ε(0/+1) "
            "matches the native EPS_0_1 baseline. Reference-independent content (per-dopant "
            "relative ordering + level spacing / negative-U sign) is preserved exactly; the "
            "absolute VBM placement of the raw PBE+U levels (~8 eV, above the 4.85 eV gap) "
            "is NOT physical because the gap calcs carry no band-edge reference / no "
            "Freysoldt ΔV_PCA — the doped-no-vacancy reference cell is not on disk."),
        "ABSOLUTE_dEf_not_extractable": (
            "Per-dopant ABSOLUTE V_O formation energy ΔE_f is NOT extractable from these "
            "doped V_O cells (needs E[doped, no-vacancy]); dims 0-2 stay the native + "
            "Reuter-Scheffler baseline. Same trap that gave v_o_ef.csv ~1490 eV garbage."),
        "per_dopant": per_dopant_meta,
        "n_cache_rows_patched": n_rows_patched,
    }
    meta["dims"] = (meta.get("dims", "") +
                    "  [dims 5-6 now PER-DOPANT (VBM-realigned DFT) for 10 gap dopants; "
                    "native EPS for others; dims 0-2 unchanged native baseline]")

    np.savez_compressed(CACHE, keys=np.array(keys), features=feats,
                        meta_json=json.dumps(meta))
    print(f"[saved] {CACHE}  ({n_rows_patched} of {len(keys)} rows patched in dims 5-6)\n")

    # ---- metrics-from-disk: RE-READ and confirm the per-dopant ε values landed ----
    idx2, feats2, meta2 = load_cache()
    print("=== RE-READ from disk: per-dopant dims 5-6 (gap dopants) ===")
    print(f"{'dop':4s} {'dom_site':9s} {'eps(0/+1)':>10s} {'eps(+1/+2)':>11s} "
          f"{'rawe01':>8s} {'q@eF4.0':>8s}")
    for d in GAP_DOPANTS:
        row = feats2[idx2[f"{d}|Ar|700"]]
        pm = meta2["gap_dopant_transitions"]["per_dopant"][d]
        print(f"{d:4s} {pm['dominant_site']:9s} {row[5]:10.3f} {row[6]:11.3f} "
              f"{pm['eps_0_1_raw_eV']:8.3f} {pm['dominant_charge_eF4.0']:8d}")

    # sanity: a non-gap dopant must STILL carry the native EPS baseline
    mg = feats2[idx2["Mg|Ar|700"]]
    assert abs(mg[5] - EPS_0_1) < 1e-9 and abs(mg[6] - EPS_1_2) < 1e-9, \
        "Mg (non-gap) dims 5-6 must stay native baseline!"
    print(f"\n[check] non-gap Mg dims 5-6 = ({mg[5]:.4f},{mg[6]:.4f}) == native baseline OK")

    # ---- monotonicity: dims 0-2 + dim 7 unchanged, so Mg ordering must STILL hold ----
    print("\n=== MONOTONICITY (Mg, dims 0-2 unchanged) ===")
    print("  Atmosphere (Mg, 700C): expect log[V_O] Ar > Ar:O2=4:1 > Ar:O2=1:1 > O2")
    prev = None
    atm_ok = True
    for atm in ["Ar", "Ar_O2_4_1", "Ar_O2_1_1", "O2"]:
        row = feats2[idx2[f"Mg|{atm}|700"]]
        lv = _anchor_log_vo(row, 973.15)
        print(f"    {atm:11s} muO={row[7]:.3f}  log[V_O]={lv:.3f}")
        if prev is not None and lv > prev + 1e-9:
            atm_ok = False
        prev = lv
    print(f"  -> atmosphere monotonic (Ar>O2): {'OK' if atm_ok else 'FAIL'}")

    print("  Temperature (Mg, Ar): expect higher T -> more V_O")
    prevT = None
    T_ok = True
    for T_C in [500, 700, 900, 1100]:
        row = feats2[idx2[f"Mg|Ar|{T_C}"]]
        lv = _anchor_log_vo(row, T_C + 273.15)
        print(f"    T={T_C}C  log[V_O]={lv:.3f}")
        if prevT is not None and lv < prevT - 1e-9:
            T_ok = False
        prevT = lv
    print(f"  -> temperature monotonic (high-T>low-T): {'OK' if T_ok else 'FAIL'}")

    assert atm_ok and T_ok, "Monotonicity broke — dims 0-2 should be untouched!"
    print("\n[done] cache enhanced; dims 0-2 monotonicity preserved; dims 5-6 per-dopant.")


if __name__ == "__main__":
    import os
    os.chdir(PROJ)
    main()
