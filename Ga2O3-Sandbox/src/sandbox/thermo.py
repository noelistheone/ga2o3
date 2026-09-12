"""Thermochemistry + bandstructure for the KROGER-class β-Ga2O3 defect engine.

CLEAN-ROOM PORT of the SGTE Gibbs-energy polynomials and material conditions used by
KROGER (Arnab et al., PCCP 2025, DOI 10.1039/D4CP04817B; github.com/mikescarpulla/KROGER,
UNLICENSED — these are the underlying published thermochemical facts, re-implemented from the
formulas, not copied code). Porting KROGER's OWN thermochemistry (rather than our independent
Shomate module) is deliberate: the stored defect formation energies cs_dHo are referenced to
THIS chemical-potential scale, so pairing them with any other μ reference would introduce a
constant offset that exponentiates into defect concentrations. src/sandbox/mu_o.py stays as an
independent cross-check.

Sources of each formula (KROGER repo files):
  G0_Ga2O3_ls : G_thermo_files/G_Ga2O3_ls.m — Zinkevich & Aldinger JACS (2004), J/mol.
  G0_O2_gv    : G_thermo_files/G_O2_gv.m — SGTE O2(g), two T ranges, J/mol.
  Eg(T)/Ec/Ev/Nc/Nv : Ga2O3/KROGER_Set_Ga2O3_Material_Conditions.m (Varshni + T^1.5 DOS).
  dSvib       : defect_equilibrium_dark.m dSvib_quantum_per_mode (Bose per-mode entropy).
"""
from __future__ import annotations

import numpy as np

# physical constants (KROGER values)
Q = 1.602176634e-19            # J/eV
AVO = 6.0221409e+23            # 1/mol
KB_EV = 8.617333262e-5         # eV/K
J_PER_MOL_TO_EV = Q * AVO      # 96485.33 J/mol per eV

# ── bandstructure / DOS (KROGER_Set_Ga2O3_Material_Conditions.m) ──────────────
EG0 = 5.0                       # 0 K gap [eV]
VARSHNI_A = 0.00105248259206626
VARSHNI_B = 676.975385507958
TREF = 300.0
NC_REF = 1e19                   # cm^-3 at 300 K
NV_REF = 5e20                   # cm^-3 at 300 K
N_SITE = 1.91e22                # cm^-3 per lattice-site type (all 9 identical in KROGER)
VIBENT_T0 = 870.0               # characteristic phonon T0 for the vibrational-entropy model [K]
# self-trapped-hole relaxation energies (E above VBM), KROGER default sth_flag=1
E_RELAX_STH1 = 0.52
E_RELAX_STH2 = 0.53


def _asarray(x):
    return np.asarray(x, dtype=np.float64)


def G0_Ga2O3_ls(T):
    """Gibbs energy of β-Ga2O3(s), eV per formula unit. T in K (array ok). X_i=1 (pure)."""
    T = _asarray(T)
    g_jmol = (-1127917.0 + 684.8332 * T - 112.3935 * T * np.log(T)
              - 0.00796268819 * T**2 + 1080114.0 / T)
    return g_jmol / J_PER_MOL_TO_EV


def G0_O2_gv(T, pO2_atm=1.0, P_tot_atm=1.0):
    """Gibbs energy of O2(g), eV per O2 molecule, incl. the k_B T ln(pO2) pressure term.

    KROGER's G0_O2_gv adds k_B T (ln(P_tot/P_ref) + ln(X_i)) with X_i = pO2/P_tot, P_ref=1 atm,
    which reduces to k_B T ln(pO2) for the O2 partial pressure.
    """
    T = _asarray(T)
    pO2_atm = _asarray(pO2_atm)
    lo = (-6960.6927 - 51.1831467 * T - 22.25862 * T * np.log(T)
          - 0.01023867 * T**2 + 1.339947e-6 * T**3 - 76749.55 / T)
    hi = (-13136.0174 + 24.7432966 * T - 33.55726 * T * np.log(T)
          - 0.0012348985 * T**2 + 1.66943333e-8 * T**3 - 539886.0 / T)
    g_jmol = np.where(T <= 900.0, lo, hi)
    g = g_jmol / J_PER_MOL_TO_EV
    return g + KB_EV * T * np.log(pO2_atm / 1.0)   # X_i term (P_tot=P_ref=1 atm cancels)


def mu_Ga_O(T, pO2_atm=1.0):
    """Host chemical potentials (μ_Ga, μ_O) in eV on KROGER's scale, from pO2-variable eq.

    μ_O = G0_O2/2 ;  μ_Ga = (G0_Ga2O3 - 3 μ_O)/2  (Ga2O3 = 2Ga + 3O equilibrium).
    """
    T = _asarray(T)
    mu_O = G0_O2_gv(T, pO2_atm) / 2.0
    mu_Ga = (G0_Ga2O3_ls(T) - 3.0 * mu_O) / 2.0
    return mu_Ga, mu_O


def delta_Eg(T):
    T = _asarray(T)
    return (VARSHNI_A * T**2) / (T + VARSHNI_B)


def band_edges(T, EcT_fraction=0.40):
    """Return (Eg(T), Ev(T), Ec(T)) with E=0 at the 0 K VBM.

    Ev(T) = (1-f)·ΔEg  rises;  Ec(T) = Eg0 - f·ΔEg  falls;  f = EcT_fraction (paper best-fit 0.40).
    """
    T = _asarray(T)
    dEg = delta_Eg(T)
    Ev = (1.0 - EcT_fraction) * dEg
    Ec = EG0 - EcT_fraction * dEg
    Eg = EG0 - dEg
    return Eg, Ev, Ec


def Nc(T):
    return NC_REF * (_asarray(T) / TREF) ** 1.5


def Nv(T):
    return NV_REF * (_asarray(T) / TREF) ** 1.5


def dSvib_norm(T, T0=VIBENT_T0):
    """Per-mode quantum vibrational entropy factor (dimensionless), KROGER definition:
    n = 1/(exp(T0/T)-1);  s = (1+n)ln(1+n) - n ln(n) + ln 2.
    The defect free-energy term is  dG_vib = 3 k_B T · s(T) · Σ_element cs_dm  (vacancy Σdm=-1).
    """
    T = _asarray(T)
    n = 1.0 / (np.expm1(T0 / T))
    return (1.0 + n) * np.log1p(n) - n * np.log(n) + np.log(2.0)


if __name__ == "__main__":
    import json
    out = {}
    for T in [300.0, 1000.0, 1400.0, 1950.0, 2068.0]:
        muGa, muO = mu_Ga_O(T, 1.0)
        muGa_lo, muO_lo = mu_Ga_O(T, 0.02)
        Eg, Ev, Ec = band_edges(T, 0.40)
        out[f"T={T:.0f}K"] = {
            "G0_Ga2O3_eV_FU": round(float(G0_Ga2O3_ls(T)), 4),
            "mu_O_pO2=1": round(float(muO), 4), "mu_Ga_pO2=1": round(float(muGa), 4),
            "mu_O_pO2=0.02": round(float(muO_lo), 4),
            "Eg": round(float(Eg), 4), "Ev": round(float(Ev), 4), "Ec": round(float(Ec), 4),
            "Nc": f"{float(Nc(T)):.3e}", "Nv": f"{float(Nv(T)):.3e}",
            "dSvib_norm": round(float(dSvib_norm(T)), 4),
            "TSvib_per_vacancy_eV": round(float(3 * KB_EV * T * dSvib_norm(T)), 4),
        }
    print(json.dumps(out, indent=2))
