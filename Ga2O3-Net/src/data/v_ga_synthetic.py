"""Phase 53 Stage 3 — V_Ga^q formation energies for KKT-Hardnet Brouwer head.

**Stage 3 完整 design**: prefers REAL DFT-derived V_Ga^q E_f from CP2K/QE Tier 3
calcs (saved to dft/qe_hse06/results/v_ga_ef.csv after user runs the QE calcs).
Falls back to literature synthetic values (Lyons 2018, Varley 2017 HSE06) when
the DFT CSV doesn't exist yet — this lets the architecture + training pipeline
be ready before DFT completes.

CSV format expected (when DFT done):
  dopant,defect_type,charge,E_f_eV,site_label
  Mg,V_Ga,0,6.34,Ga_I
  Mg,V_Ga,-1,5.17,Ga_I
  Mg,V_Ga,-2,3.59,Ga_I
  ...
  Sn,Sn-V_Ga_complex,-1,-2.12,—
  ...

If `dft/qe_hse06/results/v_ga_ef.csv` exists, use those values; else use literature.

Convention:
  E_f^q(εF, μ_Ga, μ_O) = E_f^q(εF=VBM, μ_Ga=0) - q*εF
  At εF=midgap (~2.4eV), V_Ga^-3 has E_f = +2 + 3*2.4 = +9.2 eV (very unstable)
  At εF=VBM, V_Ga^-3 has E_f = +2 eV (most stable in p-type, fully ionized)
"""
from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

# ── Synthetic V_Ga^q E_f at εF=VBM (HSE06 literature averages, μ_Ga=0) ──
# Lyons 2018 J. Appl. Phys.; Varley 2017 — used as fallback when DFT CSV missing.
NATIVE_V_GA_EF_VBM = {
    0:  +6.5,    # V_Ga neutral (Ga-rich limit)
    -1: +5.0,    # transition 0/-1 at εF≈1.5 eV above VBM
    -2: +3.5,    # transition -1/-2 at εF≈2.5 eV
    -3: +2.0,    # transition -2/-3 at εF≈3.5 eV (deepest acceptor in n-type)
}

# Dopant effective charge on Ga^3+ site
# Acceptor (M^2+): -1, Isovalent (M^3+): 0, Donor (M^4+): +1, Super-donor (M^5+): +2, etc.
DOPANT_CHARGE = {
    "Mg": -1, "Zn": -1, "Cu": -1, "Ni": -1,
    "Al": 0, "Fe": 0, "B": 0, "V": 0, "Er": 0, "Eu": 0, "Cr": 0,
    "Sb": 0, "Bi": 0,    # V52 lone-pair relabel
    "Si": +1, "Sn": +1, "Ti": +1, "Ge": +1,
    "Ta": +2, "W": +3,
}


def load_v_ga_ef_table(dft_csv_path: str | None = None) -> dict:
    """Load V_Ga^q E_f table (DFT CSV preferred, synthetic fallback).

    Returns:
        Dict {(dopant, q): E_f_at_εF=VBM} for each (dopant, charge) tuple.
        Includes 'native' (no dopant) entries: ('native', q).
    """
    PROJ = Path(__file__).resolve().parents[2]
    if dft_csv_path is None:
        dft_csv_path = PROJ / "dft" / "qe_hse06" / "results" / "v_ga_ef.csv"
    else:
        dft_csv_path = Path(dft_csv_path)

    table = {}
    # Always include native V_Ga^q (no dopant)
    for q, ef in NATIVE_V_GA_EF_VBM.items():
        table[("native", q)] = ef

    # Layer DFT data on top if available
    if dft_csv_path.exists():
        df = pd.read_csv(dft_csv_path)
        for _, row in df.iterrows():
            key = (row["dopant"], int(row["charge"]))
            table[key] = float(row["E_f_eV"])
        print(f"[v_ga_ef] Loaded {len(df)} DFT V_Ga entries from {dft_csv_path}")
    else:
        # Synthetic fallback: each dopant inherits native E_f (no dopant-specific shift)
        for elem in DOPANT_CHARGE:
            for q in NATIVE_V_GA_EF_VBM:
                table[(elem, q)] = NATIVE_V_GA_EF_VBM[q]

    return table


def v_ga_ef_at_eF(table: dict, dopant: str, q: int, eF: float) -> float:
    """V_Ga^q E_f at given εF for given dopant (returns synthetic if no DFT entry)."""
    key = (dopant, q)
    if key not in table:
        key = ("native", q)
    if key not in table:
        return float("inf")  # invalid charge
    return table[key] - q * eF
