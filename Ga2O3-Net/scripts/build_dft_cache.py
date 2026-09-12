"""V58 — build data/processed/dft_features_cache_v58.npz (DFT-anchored V_O features).

Design ref: phase58_v58_dft_llm_hybrid_design.md §3.4, §4.1, §7.12.

⚠ ANCHOR-SOURCE DECISION (2026-05-27, recorded in design-doc §7 callout + audit log):
The raw QE outputs `dft/qe_hse06/results/v_o_ef.csv` CANNOT be used as a clean V_O
formation-energy source: those supercells are DOPED (e.g. `Bi_O_I_q0` = V_O in a
Bi-doped 79-atom cell) and `parse_outputs.py` referenced them to PRISTINE Ga2O3 adding
only +μ_O — never the Ga→dopant substitution term (+μ_Ga − μ_M). The result conflates
the dopant atomic energy (≈1200-1500 eV per-dopant offsets) and the simplified Freysoldt
charge correction yields unphysical transition levels (≈8 eV > the 4.85 eV gap). Building
the V58 anchor from those numbers gives log[V_O] ≈ −7000 (verified, rejected).

Instead the anchor uses the repo's CLEAN HSE06-literature parametrization in
`src/data/kroger_synthetic.py` — `NATIVE_VO_EF = {0:+3.5, 1:+1.8, 2:+0.3} eV`
(native V_O formation at εF=midgap, μ_O=0; HSE06 Lyons 2022 = the doc's KROGER/Lyons §6.2
reference; the +3.5 eV neutral value matches Varley 2010 / doc §7.11). This IS HSE06 DFT
formation energy, satisfying §3.4's intent and H3 traceability.

εF treatment: midgap (already baked into NATIVE_VO_EF), NOT the doc §4.1 per-class εF.
Rationale: a fixed per-class εF inside a Boltzmann-over-charges sum drives E_f(V_O^2+)
negative for acceptors → [V_O] explodes past N_site, AND inverts the empirical dopant→V_O
direction (charge-neutrality feedback — the measured V_O includes compensation, the bare
fixed-εF formation does not). So the DFT anchor here fixes the ATMOSPHERE (μ_O), TEMPERATURE
and CHARGE-STATE physics + the absolute V_O scale (the Tier-1C magnitude fix); the dopant
DIRECTION stays with the composition stream + δ residual + within-DOI loss, exactly as in
V55-Ext. Dopant-resolved DFT anchoring (per-dopant ΔE_f via self-consistent Brouwer εF +
M_Ga substitution calcs) is the documented P1/P2 refinement.

Atmosphere/T dependence (the physical content of the anchor):
    ΔE_f^q(T, p_O2) = NATIVE_VO_EF[q] + Δμ_O(T, p_O2),   Δμ_O = μ_O(T,p) − μ_O^{O-rich} ≤ 0
For V_O one O is removed → +μ_O; lower μ_O (Ar / high T) ⇒ lower ΔE_f ⇒ more V_O. Correct.
The cache is per (dopant, atm, T): identical ΔE_f baseline across dopants (native V_O),
differing only via dim 8 (Δμ_M_max) and the row's composition/process downstream.

16-d vector (dims per §3.4): 0-2 ΔE_f(V_O^{0,1,2}); 3-4 M_Ga (NaN, no QE sub calc → M4
imputes); 5-6 V_O transition levels ε(0/+1),ε(+1/+2); 7 μ_O(T,p); 8 Δμ_M_max; 9 log10
N_site (=log_pref), 10-12 = 0 (entropy comps, P2); 13 Eg; 14 ε_0; 15 ε_∞.

GPU-over-CPU: the logsumexp / Boltzmann reductions use torch; the per-(dopant,atm,T)
assembly loop is one-shot startup (rule-exempt).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.data.kroger_synthetic import (
    NATIVE_VO_EF, DOPANT_OFFSET, EG_GA2O3, LOG_PREFACTOR, KB_EV, LN10,
)
from src.data.mu_o_table import lookup as mu_o_lookup
from src.data.defect_chemical_potentials import fermi_level_eV, delta_mu_m_max

PROJ = Path(__file__).resolve().parents[1]

ATM_P_O2 = {"Ar": 1e-5, "Ar_O2_4_1": 0.05, "Ar_O2_1_1": 0.2, "O2": 1.0}
T_C_GRID = [500, 700, 900, 1100]
EPS0 = 11.2
EPS_INF = 3.5
# Anchor Fermi level: β-Ga2O3 is intrinsically/unintentionally n-type, εF near CBM.
# Using εF≈4.0 eV (vs NATIVE_VO_EF's midgap 2.425) keeps the donor V_O^{+1,+2} formation
# energies POSITIVE so the Boltzmann-over-charges sum does not saturate past N_site, and
# gives the physically correct suppression of V_O in oxidizing/low-T conditions. A single
# realistic host εF avoids both the saturation (midgap) and the sign-inversion (per-class
# εF) failure modes — see module docstring.
ANCHOR_FERMI_EV = 4.0
# V_O transition levels from NATIVE_VO_EF (εF=midgap). ε(q/q') = [E_f^q(midgap) −
# E_f^q'(midgap)]/(q'−q) + (q'−q)·(Eg/2)/(q'−q)... computed at εF=0: E_f^q(0)=NATIVE[q]−q·Eg/2.
_EF0 = {q: NATIVE_VO_EF[q] - q * (EG_GA2O3 / 2.0) for q in (0, 1, 2)}
EPS_0_1 = _EF0[0] - _EF0[1]   # ε(0/+1) eV above VBM
EPS_1_2 = _EF0[1] - _EF0[2]   # ε(+1/+2)


def build_cache(
    mu_o_npz: str | Path = "data/processed/mu_O_table_v58.npz",
    chem_pot_json: str | Path = "dft/qe_hse06/results/chemical_potentials.json",
    out_path: str | Path = "data/processed/dft_features_cache_v58.npz",
) -> dict:
    chem = json.loads((PROJ / chem_pot_json).read_text())
    mu_O_Orich = float(chem["limits"]["O_rich"]["mu_O"])
    dopants = sorted(DOPANT_OFFSET.keys())

    keys, feats, provenance = [], [], {}
    for dop in dopants:
        for atm, p in ATM_P_O2.items():
            for T_C in T_C_GRID:
                T_K = T_C + 273.15
                mu_O = mu_o_lookup(PROJ / mu_o_npz, T_K, p)
                d_mu_O = mu_O - mu_O_Orich                     # ≤ 0
                # E_f^q(εF) = NATIVE_VO_EF[q] + q·(εF − Eg/2), then atmosphere shift Δμ_O.
                dE = [NATIVE_VO_EF[q] + q * (ANCHOR_FERMI_EV - EG_GA2O3 / 2.0) + d_mu_O
                      for q in (0, 1, 2)]
                d_mu_M = float(delta_mu_m_max(dop, mu_O, mu_O_Orich))
                vec = [dE[0], dE[1], dE[2],
                       float("nan"), float("nan"),             # M_Ga dims (no QE calc)
                       EPS_0_1, EPS_1_2,
                       mu_O, d_mu_M,
                       LOG_PREFACTOR, 0.0, 0.0, 0.0,
                       EG_GA2O3, EPS0, EPS_INF]
                keys.append(f"{dop}|{atm}|{T_C}")
                feats.append(vec)
        provenance[dop] = {
            # NOTE: the applied anchor εF is the single n-type host value ANCHOR_FERMI_EV
            # (see meta.eps_F_eV); the per-class value below is informational ONLY (it is
            # NOT applied to the anchor — applying it would saturate/invert, see docstring).
            "valence_class_fermi_eV_informational_not_applied": fermi_level_eV(dop),
            "dopant_offset_empirical": DOPANT_OFFSET[dop]}

    feats_arr = np.array(feats, dtype=np.float64)
    meta = dict(
        source="kroger_synthetic NATIVE_VO_EF (HSE06 Lyons 2022) + Reuter-Scheffler mu_O",
        anchor_note="native V_O formation (dopant-independent baseline) + atmosphere/T via mu_O; "
                    "dopant direction left to composition stream + delta + within-DOI loss",
        rejected_source="v_o_ef.csv (doped supercells, pristine ref, conflated dopant energy)",
        native_vo_ef=NATIVE_VO_EF, eps_F_eV=ANCHOR_FERMI_EV,
        eps_F_rationale="n-type host εF near CBM; avoids midgap saturation + per-class sign-inversion",
        eps_0_1_eV=round(EPS_0_1, 4), eps_1_2_eV=round(EPS_1_2, 4),
        mu_O_Orich_eV=mu_O_Orich, atm_p_O2=ATM_P_O2, T_C_grid=T_C_GRID,
        log10_N_site=LOG_PREFACTOR, Eg_eV=EG_GA2O3, eps0=EPS0, eps_inf=EPS_INF,
        dims="0-2 dEf(V_O); 3-4 M_Ga(NaN); 5-6 transitions; 7 muO; 8 dmuM; 9-12 log_pref; 13 Eg; 14 eps0; 15 eps_inf",
        dopants_covered=dopants, n_entries=len(keys), provenance=provenance,
    )
    out_path = PROJ / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, keys=np.array(keys), features=feats_arr,
                        meta_json=json.dumps(meta))
    return meta


def load_cache(npz_path: str | Path = "data/processed/dft_features_cache_v58.npz"):
    p = npz_path if Path(npz_path).is_absolute() else PROJ / npz_path
    d = np.load(p, allow_pickle=False)
    keys = [str(k) for k in d["keys"]]
    return {k: i for i, k in enumerate(keys)}, d["features"], json.loads(str(d["meta_json"]))


def _anchor_log_vo(feat_row, T_K: float) -> float:
    import torch
    ef = torch.tensor(feat_row[0:3], dtype=torch.float64)
    log_pref = float(np.sum(feat_row[9:13]))
    log_vo = log_pref + float(torch.logsumexp(-ef / (KB_EV * T_K), 0)) / LN10
    return min(log_vo, LOG_PREFACTOR)  # clamp ≤ N_site (physical)


if __name__ == "__main__":
    import os
    os.chdir(PROJ)
    meta = build_cache()
    print(f"Built dft_features_cache_v58.npz: {meta['n_entries']} entries, "
          f"{len(meta['dopants_covered'])} dopants")
    print(f"  transitions: eps(0/+1)={meta['eps_0_1_eV']} eV, eps(+1/+2)={meta['eps_1_2_eV']} eV")
    idx, feats, _ = load_cache()
    # atmosphere ordering: log[V_O] must DECREASE Ar -> O2 (oxidizing suppresses V_O)
    print("  Atmosphere ordering (Mg, T=700C): expect Ar > Ar:O2 > O2")
    for atm in ["Ar", "Ar_O2_4_1", "Ar_O2_1_1", "O2"]:
        row = feats[idx[f"Mg|{atm}|700"]]
        print(f"    {atm:11s} muO={row[7]:.3f}  dEf={row[0]:.2f}/{row[1]:.2f}/{row[2]:.2f}  "
              f"log[V_O]={_anchor_log_vo(row, 973.15):.2f}")
    print("  Temperature ordering (Mg, Ar): expect higher T -> more V_O")
    for T_C in T_C_GRID:
        row = feats[idx[f"Mg|Ar|{T_C}"]]
        print(f"    T={T_C}C  log[V_O]={_anchor_log_vo(row, T_C+273.15):.2f}")
