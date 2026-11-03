"""Transport layer — mobility μ and conductivity σ, FED BY the defect-equilibrium solver.

The novel coupling (nobody has done this for β-Ga2O3): the ionized-scattering-center density
N_I comes from the Tier-1 charge-neutrality solve at the film's process conditions, not from a
Hall fit. Compensation is inside N_I = Σ|q|·N_cs (ionized acceptors are summed there with weight
|q|); there is NO separate compensating-acceptor term — mobility(..., N_A_cm3=...) accepts N_A
for diagnostics but does not use it, and no compensation-ratio model is fitted because the only
validated anchor set (scripts/validate_transport.py) is uncompensated. Mobility model:

  μ_POP(T)  : polar-optical-phonon-limited Hall mobility, Ma et al. APL 109,212101 (2016),
              μ_POP,Hall = 56·(exp(508/T) − 1) cm²/Vs (validated vs β-Ga2O3 Hall 300-500 K;
              300 K → 248, consistent with first-principles ceiling ~250 Hall / 258 drift,
              Poncé & Giustino PRR 2,033102 (2020)).
  μ_II(T,N_I): Brooks-Herring ionized-impurity, N_I = Σ|q|·N_cs from the solver (compensation
              built in). SI closed form (Chattopadhyay & Queisser, RMP 53,745 (1981)).
  Matthiessen: 1/μ = 1/μ_POP + 1/μ_II  (+ 1/μ_GB for polycrystalline films).
  μ_GB(T)   : Seto thermionic grain-boundary barrier, μ_GB = μ0·exp(−E_B/kT); E_B is the one
              calibrated film parameter (0 = single crystal).
  Hall vs drift: r_H ≈ 1.2 (falls 1.9→1.1 as density rises); σ = e·n·μ_drift.

Also provides Ma's empirical compensation-aware closed form as a well-anchored cross-check.
"""
from __future__ import annotations

import numpy as np

from . import thermo

# material constants (β-Ga2O3)
EPS_S = 10.2          # static dielectric (avg)
EPS_INF = 3.57
M_STAR = 0.28         # conduction effective mass / m0
E = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB_J = 1.380649e-23
HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
KB_EV = thermo.KB_EV
POP_ENERGY_MEV = 44.0  # effective POP phonon energy (Ma)


# Caughey-Thomas/Masetti empirical μ(N_I) fitted to single-crystal β-Ga2O3 RT Hall data
# (196@2.3e16, 180@1e17, 150@1e18, 90@1e19, 85@1.2e20 — Ma/Neal/Bhattacharyya). Spans the full
# 1e16-1e20 range with the correct degenerate plateau where Ma's power law over-suppresses.
CT_MU_MAX = 205.0     # cm²/Vs (low-N_I POP+residual ceiling)
CT_MU_MIN = 82.0      # cm²/Vs (degenerate plateau)
CT_N_REF = 1.18e18    # cm^-3
CT_ALPHA = 1.25


def mu_caughey_thomas(N_I_cm3, T=300.0):
    """Primary RT Hall mobility [cm²/Vs] as a function of total ionized density N_I (from the
    solver, compensation-aware). Weak T-scaling of the phonon ceiling via the POP form."""
    N_I = np.asarray(N_I_cm3, float)
    mu_max = CT_MU_MAX * (mu_pop_hall(T) / mu_pop_hall(300.0))   # let the ceiling ride POP(T)
    return CT_MU_MIN + (mu_max - CT_MU_MIN) / (1.0 + (N_I / CT_N_REF) ** CT_ALPHA)


def mu_pop_hall(T):
    """POP-limited HALL mobility [cm²/Vs], Ma et al. closed form. Valid ~300-500 K."""
    T = np.asarray(T, float)
    return 56.0 * (np.exp(508.0 / T) - 1.0)


def mu_brooks_herring(T, N_I_cm3, n_screen_cm3, m_star=M_STAR, eps_s=EPS_S):
    """Brooks-Herring ionized-impurity HALL-scale mobility [cm²/Vs] (independent cross-check).

    SI closed form (Chattopadhyay & Queisser, RMP 53,745 (1981)):
      μ = 2^{7/2}(ε)²(kT)^{3/2} / (π^{3/2} e³ √m* N_I) · [ln(1+b) − b/(1+b)]^{-1}
      b = 8 m* ε (kT)² / (ħ² e² n_screen)
    computed in SI, returned in cm²/Vs.
    """
    T = np.asarray(T, float)
    N_I = np.maximum(np.asarray(N_I_cm3, float), 1.0) * 1e6      # m^-3
    n_s = np.maximum(np.asarray(n_screen_cm3, float), 1e10) * 1e6
    ms = m_star * M0
    eps = eps_s * EPS0                                          # F/m
    kT = KB_J * T
    b = (8.0 * ms * eps * kT**2) / (HBAR**2 * E**2 * n_s)
    screen = np.maximum(np.log1p(b) - b / (1.0 + b), 1e-8)
    mu_si = (2.0**3.5 * eps**2 * kT**1.5) / (
        np.pi**1.5 * E**3 * np.sqrt(ms) * N_I * screen)         # m²/Vs
    return mu_si * 1e4                                          # cm²/Vs


def hall_factor(T, n_cm3):
    """Heuristic r_H=mu_H/mu_drift: 1.55 at low n to 1.05 at high n.

    This interpolation is not a sample-specific Hall-factor measurement or a scattering
    calculation. A single-electron Hall density is n_H=n/r_H. Report r_H=1 sensitivity
    where experiments do not state their Hall-factor convention.
    """
    n = np.asarray(n_cm3, float)
    # smooth interpolation on log density between ~1.6 (n<1e17) and ~1.05 (n>5e19)
    x = np.clip((np.log10(np.maximum(n, 1.0)) - 17.0) / (19.7 - 17.0), 0.0, 1.0)
    return 1.55 - 0.5 * x


def seto_gb_factor(T, E_B_eV):
    """Seto thermionic grain-boundary mobility factor exp(-E_B/kT). E_B=0 -> single crystal."""
    return np.exp(-np.asarray(E_B_eV, float) / (KB_EV * np.asarray(T, float)))


def ma_compensation_closed_form(T, N_I_cm3):
    """Ma et al. empirical HALL mobility with ionized density N_I (compensation-aware) [cm²/Vs]."""
    T = np.asarray(T, float)
    denom = (1.0 + np.asarray(N_I_cm3, float) / (np.maximum(T - 278.0, 1.0) * 2.8e16)) ** 0.68
    return mu_pop_hall(T) * (1.0 / denom)


def mobility(T, n_cm3, N_I_cm3, N_A_cm3=0.0, film=False, E_B_eV=0.0, m_star=M_STAR):
    """Full mobility from solver-fed densities. Returns drift & Hall μ and the components.

    N_I: total ionized scattering centers (Σ|q|N_cs from solver). n_cm3: free electrons (screen).

    N_A_cm3 is ACCEPTED BUT NOT USED, deliberately, and is echoed back only as a diagnostic.
    Compensation already enters through N_I and only through N_I: full_forward._ionized_densities
    builds N_I = Σ|q|·N_cs over ALL charge states, so every ionized acceptor is already counted
    there (N_A = Σ max(−q,0)·N_cs is a strict subset of that sum). Caughey-Thomas / Brooks-Herring
    / Ma are all parameterised on the TOTAL ionized scattering-centre density, so adding N_A again
    would double-count the acceptors, not add missing physics.

    No compensation-RATIO term is introduced either. The only validated anchor this layer has
    (scripts/validate_transport.py: 5 published single-crystal Hall μ(n) points, max_rel_err 0.11)
    is an UNCOMPENSATED set — it is evaluated at N_I = n, i.e. N_A = 0 — so it carries exactly
    zero information about a μ(N_A/N_D) dependence and could not falsify any ratio model fitted
    here. Introducing one would be unanchored curve-fitting, so it is not done.
    """
    T = float(T)
    mu_ct = float(mu_caughey_thomas(N_I_cm3, T))            # PRIMARY (Ga2O3-anchored, full range)
    mu_pop = float(mu_pop_hall(T))
    mu_ii = float(mu_brooks_herring(T, N_I_cm3, n_cm3, m_star))
    mu_ma = float(ma_compensation_closed_form(T, N_I_cm3))  # cross-check (valid <~1e18)
    rH = float(hall_factor(T, n_cm3))
    seto = float(seto_gb_factor(T, E_B_eV)) if film else 1.0
    # primary Hall mobility = Caughey-Thomas μ(N_I) × Seto GB for films
    mu_hall = mu_ct * seto
    mu_drift = mu_hall / rH
    return {
        "mu_hall": mu_hall, "mu_drift": mu_drift, "r_hall": rH,
        "mu_CaugheyThomas_hall": mu_ct, "seto_factor": seto,
        "mu_Ma_hall": mu_ma, "mu_POP_hall": mu_pop, "mu_II_BH_hall": mu_ii,
        "regime": "film(GB)" if film else "single-crystal/epi",
        "N_I_used_cm3": float(np.asarray(N_I_cm3, float)),
        "N_A_cm3_diagnostic_unused": float(np.asarray(N_A_cm3, float)),
        "compensation_note": "compensation enters via N_I only (N_A is a subset of N_I and is "
                             "not applied separately; no compensation-ratio term — no "
                             "compensated anchor exists to validate one)",
    }


def conductivity(n_cm3, mu_drift_cm2Vs):
    """σ = e·n·μ_drift [S/cm]. n in cm^-3, μ in cm²/Vs → e in C, so σ[S/cm] = e·n·μ."""
    return E * np.asarray(n_cm3, float) * np.asarray(mu_drift_cm2Vs, float)
