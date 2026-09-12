"""Phase 53 Stage 2 — Synthetic KROGER-style dataset for encoder_kroger pretrain.

Simplified Brouwer-form V_O equation specialized for β-Ga2O3 + 19 dopants.
Goal: produce smooth, physical synthetic (T, p_O2, dopant, [c]) → log10[V_O^q]
tuples for the encoder_kroger MLP pretrain. Used as auxiliary signal in V53-α.

Approach: V5-style closed-form Brouwer (matches BrouwerHeadVC math) with element
offsets from LIT_RHO_VC convention:
  log10[V_O^q] = log_prefactor[q]
                  - E_f^q / (kT · ln10)
                  - 0.5 · log10(p_O2)
                  + dopant_term(elem, c, q)
                  + log_charge_offset(q, εF_assumed)

E_f and offsets parametrized from HSE06 literature averages (Lyons 2022, Varley
2017). Concentrations are clamped to physical bounds [10⁹, N_O_sites].
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# Constants
KB_EV = 8.617333262e-5
LN10 = math.log(10.0)

# β-Ga2O3 host
EG_GA2O3 = 4.85           # bandgap (HSE06)
N_O_SITES = 2.85e22       # anion site density (cm⁻³, 1.5 × 1.9e22 cation)
LOG_PREFACTOR = math.log10(N_O_SITES)  # ~22.45

# Native V_O formation energies at εF = midgap, μ_O = 0 (HSE06 Lyons 2022)
NATIVE_VO_EF = {0: +3.5, 1: +1.8, 2: +0.3}

# Element-specific dopant_term offset (multiplies log10([dopant]/c_ref))
# Sign convention: + → high dopant raises V_O; − → suppresses V_O (LIT_RHO_VC)
# Source: eval_physics_diag_table.py LIT_RHO_VC + extrapolations
DOPANT_OFFSET = {
    # acceptors (v=2): more dopant → less V_O (Mg/Zn/Cu, p-type compensation)
    "Mg": -0.95, "Zn": -0.90, "Cu": -0.85, "Ni": -0.85,
    # isovalent (v=3): negligible direct V_O effect
    "Al": +0.05, "Fe": +0.10, "B": +0.05, "V": +0.10, "Er": +0.05,
    "Eu": +0.05, "Cr": +0.10,
    # lone-pair isovalent (Bi/Sb relabel, V52)
    "Sb": +0.10, "Bi": +0.10,
    # donors (v=4): more dopant → more V_O (Sn/Si/Ti/Ge)
    "Si": +0.95, "Sn": +0.95, "Ti": +0.80, "Ge": +0.85,
    # super-donors (v≥5): strongest V_O enhancement
    "Ta": +0.95, "W": +0.95,
}
ELEMENTS = list(DOPANT_OFFSET.keys())

C_REF = 1e-2  # 1 at% as concentration reference


def kroger_predict(elem: str, c_total: float, T: float, log_pO2: float,
                   q: int = 0) -> float:
    """Closed-form V_O concentration prediction for given inputs.

    Returns log10([V_O^q] / cm⁻³) clamped to [9, log10(N_O_sites)].

    Args:
        elem:     dopant cation symbol
        c_total:  total dopant atomic fraction [0, 1]
        T:        absolute temperature [K]
        log_pO2:  log10 of O₂ partial pressure (atm). 0 = 1 atm, -20 = vacuum
        q:        V_O charge state {0, 1, 2}
    """
    if elem not in DOPANT_OFFSET:
        offset = 0.0
    else:
        offset = DOPANT_OFFSET[elem]

    # Concentration term: a · log10(c / c_ref)
    log_c = math.log10(max(c_total, 1e-6) / C_REF)
    dopant_term = offset * log_c

    # Boltzmann term for charge q
    E_f_q = NATIVE_VO_EF.get(q, NATIVE_VO_EF[0])
    boltz = -E_f_q / (KB_EV * T * LN10)

    # Charge-state εF dependence (assume εF = midgap)
    eF_assumed = EG_GA2O3 / 2.0
    charge_term = q * eF_assumed / (KB_EV * T * LN10)

    # Combine (using 0.5 * log10(p_O2) factor from O₂ equilibrium)
    log_VO = LOG_PREFACTOR + boltz - 0.5 * log_pO2 + dopant_term + charge_term

    # Clamp to physical bounds (above thermal noise, below site density)
    return max(9.0, min(LOG_PREFACTOR, log_VO))


def generate_synthetic_dataset(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate (dopant, c, T, p_O2, q) → log10[V_O^q] tuples."""
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n_samples):
        elem = rng.choice(ELEMENTS)
        # Concentrations: log-uniform in [10⁻⁴, 0.1] at-fraction
        c_total = float(10 ** rng.uniform(-4, -1))
        T = float(rng.uniform(600, 1500))
        log_pO2 = float(rng.uniform(-20, 0))
        # All 3 charge states for each (elem, c, T, p_O2)
        for q in (0, 1, 2):
            log_VO = kroger_predict(elem, c_total, T, log_pO2, q)
            rows.append({
                "elem": elem,
                "c_total": c_total,
                "T_K": T,
                "log_pO2": log_pO2,
                "q": q,
                "log_VO": log_VO,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "kroger_synthetic.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic KROGER for β-Ga2O3 / {len(ELEMENTS)} dopants × 3 charges...")
    df = generate_synthetic_dataset(n_samples=10000, seed=42)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print()
    print("Per-element / charge log_VO range:")
    print(df.groupby(["elem", "q"])["log_VO"].agg(["mean", "std", "min", "max"]).round(2).to_string())
