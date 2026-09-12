"""V58 — dopant & host chemical potentials for defect formation energies.

Design ref: phase58_v58_dft_llm_hybrid_design.md §7.8 (+ §4.1, §4.2).

Provides, as a function of (T, p_O2):
  - μ_Ga(T, p_O2): coupled to μ_O via the β-Ga2O3 equilibrium
        2 μ_Ga + 3 μ_O = μ(Ga2O3) = E_per_FU_Ga2O3   (QE-consistent, exact)
    so  μ_Ga(T,p) = ½ ( E_per_FU_Ga2O3 − 3 μ_O(T,p) ).
  - Δμ_M^max(M; T, p_O2): the dopant chemical-potential UPPER BOUND set by the most
    stable competing oxide M_aO_b (competing-phase analysis, §7.8):
        a μ_M + b μ_O = a μ_M^0 + b μ_O^0 + ΔG_f(M_aO_b)
      ⇒ Δμ_M^max = μ_M − μ_M^0 = [ ΔG_f(M_aO_b) − b·Δμ_O(T,p) ] / a
    where Δμ_O(T,p) = μ_O(T,p) − μ_O^{O-rich} ≤ 0 is the O-chemical-potential deviation
    from the O-rich (elemental O2) reference at which ΔG_f is defined.

Δμ_M is expressed RELATIVE to the elemental metal reference (μ_M^0 ≡ 0), which is the
standard competing-phase formalism and the form that enters M_Ga substitution energies
E_f(M_Ga) = E_def − E_bulk − μ_M + μ_Ga + … . This avoids needing a QE elemental-metal
calc per dopant (which we do not have) while keeping the physically meaningful
atmosphere dependence exact.

APPROXIMATION (documented, not silent): ΔG_f(M_aO_b) uses tabulated EXPERIMENTAL
standard formation enthalpies (298 K, kJ/mol), not QE-computed oxide energies. This is
a common and defensible choice (many defect papers use experimental competing-phase
energies). It affects ONLY the M_Ga dopant-substitution feature dims (3-4) and the
Stage-1 synthetic-QA μ_M values; it does NOT enter V_O formation (the V58-lite-DFT
critical path), which depends on μ_O alone. A QE-consistent μ_M (elemental + oxide QE
calcs) is a P2 refinement noted in §7.8.

GPU-over-CPU (feedback_gpu_over_cpu 2026-05-27): grid evaluations vectorize over the
T×p grid on cuda via mu_o_table.mu_o; scalar per-dopant lookups stay on host.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .mu_o_table import KB_EV, mu_o  # noqa: F401  (KB_EV re-export convenience)

_KJMOL_PER_EV = 96.485332

# Most-stable competing oxide per dopant: (formula, ΔH_f° kJ/mol per f.u., n_M, n_O).
# Experimental standard formation enthalpies (298 K), CRC / NIST / Kubaschewski.
# F has NO stable oxide → bounded by ½ E[F2(g)] (elemental gas); handled specially.
_COMPETING_OXIDE = {
    "Mg": ("MgO",    -601.6,  1, 1),
    "Zn": ("ZnO",    -350.5,  1, 1),
    "Si": ("SiO2",   -910.7,  1, 2),
    "Sn": ("SnO2",   -577.6,  1, 2),
    "Ge": ("GeO2",   -580.0,  1, 2),
    "Ti": ("TiO2",   -944.0,  1, 2),
    "Ta": ("Ta2O5", -2046.0,  2, 5),
    "Cu": ("CuO",    -157.3,  1, 1),
    "Fe": ("Fe2O3",  -824.2,  2, 3),
    "Cr": ("Cr2O3", -1139.7,  2, 3),
    "In": ("In2O3",  -925.8,  2, 3),
    "Sb": ("Sb2O3",  -708.8,  2, 3),
    "Bi": ("Bi2O3",  -573.9,  2, 3),
    "W":  ("WO3",    -842.9,  1, 3),
    "V":  ("V2O5",  -1550.6,  2, 5),
    "La": ("La2O3", -1793.7,  2, 3),
    "Eu": ("Eu2O3", -1651.4,  2, 3),
    "Er": ("Er2O3", -1897.9,  2, 3),
    "B":  ("B2O3",  -1273.5,  2, 3),
    # F: no oxide; μ_F^max = ½ E[F2]. Δμ_F^max = 0 at the elemental F2 reference,
    # i.e. F is bounded by its own gas, independent of μ_O. Stored as 0.0 (relative).
    "F":  ("F2(g)",     0.0,  1, 0),
}

# Typical Fermi level per dopant valence class (eV above VBM), doc §4.1.
_FERMI_BY_CLASS = {"acceptor": 0.5, "donor": 4.0, "super-donor": 4.0,
                   "isovalent": 2.5, "anion": 0.5}

# Coarse valence-class map for the dopants used (for the §4.1 Fermi-level prior).
_DOPANT_CLASS = {
    "Mg": "acceptor", "Zn": "acceptor", "Cu": "acceptor", "Fe": "acceptor",
    "Cr": "isovalent", "Ni": "acceptor", "B": "acceptor",
    "Si": "donor", "Sn": "donor", "Ge": "donor", "Ti": "donor", "Ta": "donor",
    "W": "donor", "V": "donor", "Sb": "super-donor", "Bi": "isovalent",
    "In": "isovalent", "La": "isovalent", "Eu": "isovalent", "Er": "isovalent",
    "F": "anion", "H": "donor",
}


def fermi_level_eV(dopant: str) -> float:
    return _FERMI_BY_CLASS[_DOPANT_CLASS.get(dopant, "isovalent")]


def mu_ga(mu_O_eV, e_per_FU_Ga2O3_eV: float):
    """μ_Ga(T,p) = ½ (E_per_FU[Ga2O3] − 3 μ_O).  Works on scalar or tensor μ_O."""
    return 0.5 * (e_per_FU_Ga2O3_eV - 3.0 * mu_O_eV)


def delta_mu_m_max(dopant: str, mu_O_eV, mu_O_Orich_eV: float):
    """Δμ_M^max(T,p) relative to elemental reference, from competing oxide.

    Δμ_M^max = [ ΔG_f(M_aO_b) − b·Δμ_O ] / a ,  Δμ_O = μ_O − μ_O^{O-rich}.
    Returns same shape as mu_O_eV (eV). For F (no oxide) returns 0 (elemental F2 bound).
    """
    if dopant not in _COMPETING_OXIDE:
        # default: treat as isovalent with a generic sesquioxide-like bound
        formula, dHf_kJ, nM, nO = ("M2O3", -1000.0, 2, 3)
    else:
        formula, dHf_kJ, nM, nO = _COMPETING_OXIDE[dopant]
    dGf_eV = dHf_kJ / _KJMOL_PER_EV  # eV per f.u. (≈ ΔG_f using ΔH_f, entropy small)
    d_mu_O = mu_O_eV - mu_O_Orich_eV  # ≤ 0
    if nM == 0 or nO == 0:  # F-like, no oxide constraint vs μ_O
        if isinstance(mu_O_eV, torch.Tensor):
            return torch.zeros_like(mu_O_eV)
        return 0.0 * mu_O_eV if hasattr(mu_O_eV, "__len__") else 0.0
    return (dGf_eV - nO * d_mu_O) / nM


def build_dft_chemical_potentials(
    chem_pot_json: str | Path = "dft/qe_hse06/results/chemical_potentials.json",
    out_path: str | Path = "data/processed/dft_chemical_potentials.json",
    T_C_grid=(500, 700, 900, 1100),
    atm_p_O2=None,
) -> dict:
    """Emit dft_chemical_potentials.json (§10.1): μ_Ga + Δμ_M^max per (dopant, atm, T).

    atm_p_O2: dict atm_name -> p_O2 (atm). Default mirrors the 4 sputter atmosphere bins.
    """
    if atm_p_O2 is None:
        atm_p_O2 = {"Ar": 1e-5, "Ar_O2_4_1": 0.05, "Ar_O2_1_1": 0.2, "O2": 1.0}
    chem = json.loads(Path(chem_pot_json).read_text())
    e_per_FU = float(chem["e_per_FU_eV"])
    mu_O_Orich = float(chem["limits"]["O_rich"]["mu_O"])
    mu_O_mid = float(chem["limits"]["mid"]["mu_O"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {
        "metadata": {
            "source": "defect_chemical_potentials.py (V58)",
            "mu_Ga": "0.5*(E_per_FU[Ga2O3] - 3 mu_O), QE-consistent",
            "delta_mu_M_max": "competing-oxide bound, EXPERIMENTAL dHf (relative to elemental ref)",
            "mu_O_Orich_eV": mu_O_Orich,
            "mu_O_mid_eV": mu_O_mid,
            "e_per_FU_Ga2O3_eV": e_per_FU,
            "fermi_level_by_class_eV": _FERMI_BY_CLASS,
            "competing_oxides": {k: v[0] for k, v in _COMPETING_OXIDE.items()},
            "note": "mu_M is relative-to-elemental (Delta mu); experimental dHf approximation, "
                    "QE-consistent refinement is P2 (does not affect V_O dims).",
        },
        "entries": {},
    }
    for dop in _COMPETING_OXIDE:
        out["entries"][dop] = {"valence_class": _DOPANT_CLASS.get(dop, "isovalent"),
                               "fermi_level_eV": fermi_level_eV(dop), "by_atm_T": {}}
        for atm, p in atm_p_O2.items():
            for T_C in T_C_grid:
                T_K = T_C + 273.15
                muO = float(mu_o(torch.tensor(T_K, dtype=torch.float64),
                                 torch.tensor(p, dtype=torch.float64),
                                 mu_O_Orich, device=device).item())
                out["entries"][dop]["by_atm_T"][f"{atm}|{T_C}C"] = {
                    "mu_O_eV": round(muO, 5),
                    "mu_Ga_eV": round(float(mu_ga(muO, e_per_FU)), 5),
                    "delta_mu_M_max_eV": round(float(delta_mu_m_max(dop, muO, mu_O_Orich)), 5),
                }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parents[2])
    res = build_dft_chemical_potentials()
    n = len(res["entries"])
    print(f"Wrote data/processed/dft_chemical_potentials.json ({n} dopants)")
    # Sanity: Δμ_M^max should DECREASE as atmosphere goes O-rich (oxide more favorable)
    for dop in ["Mg", "Sn", "Sb", "F"]:
        bt = res["entries"][dop]["by_atm_T"]
        ar = bt["Ar|700C"]["delta_mu_M_max_eV"]
        o2 = bt["O2|700C"]["delta_mu_M_max_eV"]
        print(f"  {dop:3s} ({res['entries'][dop]['valence_class']:11s})  "
              f"Δμ_M^max[Ar,700C]={ar:8.3f}  [O2,700C]={o2:8.3f}  "
              f"(O2<Ar? {'OK' if o2 <= ar + 1e-6 else 'FLIP'})  "
              f"E_F={res['entries'][dop]['fermi_level_eV']}eV")
