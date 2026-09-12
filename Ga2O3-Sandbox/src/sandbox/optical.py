"""Optical layer — optical gap Eg and doping-induced shift dEg, FED BY the equilibrium n.

Closes the chain Peelaers & Van de Walle (APL 110, 082101 (2017)) left open: they computed the
Burstein-Moss shift + free-carrier absorption at an ASSUMED carrier density; here n comes from
the Tier-1 defect-equilibrium solve, so Eg/dEg become functions of (dopant, process).

  Burstein-Moss (band filling, widens gap): ΔE_BM = (ħ²/2 m_bm)(3π² n)^{2/3}
      m_bm ≈ 0.34 m0 (nonparabolicity-corrected conduction mass; reproduces Peelaers' E_F−CBM =
      0.23 eV at n=1e20). Optical widening ≈ ΔE_BM since m_v ≫ m_c.
  Band-gap renormalization (narrows gap): ΔE_BGR = −k_bgr · n^{1/3}, small in Ga2O3 so the net
      degenerate-donor shift is a WIDENING — calibrated so net ≈ +0.19 eV at degenerate n
      (Si-doped films 4.80→4.99 eV, Cell Rep. Phys. Sci. 3, 100907 (2022)).

  Eg_optical(n)  = Eg(300 K) + ΔE_BM − |ΔE_BGR|
  dEg(doped)     = Eg_optical(n_doped) − Eg_optical(n_undoped)  (within-paper differential)

Honest scope: quantitative for the degenerate-DONOR Burstein-Moss channel; deep-acceptor /
isovalent-dopant film dEg is process/disorder-dominated and NOT captured here (direction only).
"""
from __future__ import annotations

import numpy as np

from . import thermo

HBAR2_2M0 = 3.80998            # ħ²/(2 m0) [eV·Å²]
M_BM = 0.34                    # effective conduction mass for band filling (nonparabolic)
K_BGR = 8.6e-9                 # eV·cm, BGR coefficient (calibrated, see module docstring)
EG_300 = float(thermo.band_edges(300.0, 0.40)[0])   # optical Eg at 300 K from Varshni


def burstein_moss(n_cm3, m_bm=M_BM):
    """ΔE_BM [eV] band-filling widening from carrier density n [cm^-3]."""
    n = np.maximum(np.asarray(n_cm3, float), 0.0)
    n_A3 = n * 1e-24                                    # Å^-3
    return HBAR2_2M0 / m_bm * (3.0 * np.pi**2 * n_A3) ** (2.0 / 3.0)


def bgr(n_cm3, k_bgr=K_BGR):
    """ΔE_BGR [eV] (negative = narrowing) from many-body renormalization."""
    n = np.maximum(np.asarray(n_cm3, float), 0.0)
    return -k_bgr * n ** (1.0 / 3.0)


def optical_eg(n_cm3, Eg_ref=EG_300):
    """Optical (Tauc) gap [eV] for a film with free-carrier density n."""
    return Eg_ref + burstein_moss(n_cm3) + bgr(n_cm3)


# Structural-disorder gap narrowing of sputter films (Tier-3 first-of-kind computation,
# results/tier3/disorder_deg.json): 20-cell MACE melt-quench a-Ga2O3 ensemble, QE-PBE DOS ->
# crystal->amorphous narrowing −0.23 eV (ensemble) / −0.55 eV (genuinely-amorphous cells).
# CROSS-VALIDATED by an INDEPENDENT MLIP (2026-07-08): MatterSim-v1 (finite-T-trained, correct
# 5.95 g/cc density, no recrystallization) gives −0.584 eV, matching the MACE amorphous-subset
# −0.551 and confirming the 0.55 constant; round-1's −0.23 ensemble was a conservative under-
# estimate (diluted by 12/20 recrystallized cells). results/tier3/disorder_deg_mattersim.json.
# Comparable-to/larger-than the BM widening (~+0.23) -> the two compete; in a well-amorphized film
# disorder narrowing DOMINATES -> net dEg small/negative & disorder-dominated, BM anti-correlated (ρ=−0.41).
DISORDER_NARROWING_AMORPH = 0.55      # eV, fully-amorphous film (upper magnitude)
DISORDER_NARROWING_ENSEMBLE = 0.23    # eV, partially-crystalline ensemble average


def disorder_narrowing(film_disorder_frac=0.5):
    """Gap narrowing [eV, positive number] from sputter-film structural disorder.
    film_disorder_frac in [0,1]: 0 = crystalline (no narrowing), 1 = fully amorphous
    (−0.55 eV). Real as-deposited films are partially amorphous/nanocrystalline."""
    return float(DISORDER_NARROWING_AMORPH * max(0.0, min(1.0, film_disorder_frac)))


def dEg(n_doped_cm3, n_undoped_cm3, disorder_doped=0.0, disorder_undoped=0.0):
    """Within-paper doping-induced gap shift [eV] = Eg_optical(doped) − Eg_optical(undoped).
    net dEg = BM widening + BGR − (disorder narrowing difference). The disorder term captures
    that heavier doping tends to increase film disorder (more narrowing) — the mechanism behind
    the disorder-dominated dEg. disorder_{doped,undoped} in [0,1] = film amorphous fraction."""
    bm = float(burstein_moss(n_doped_cm3) - burstein_moss(n_undoped_cm3))
    bg = float(bgr(n_doped_cm3) - bgr(n_undoped_cm3))
    dis = -(disorder_narrowing(disorder_doped) - disorder_narrowing(disorder_undoped))
    return {"dEg": bm + bg + dis, "dEg_BM": bm, "dEg_BGR": bg, "dEg_disorder": dis}


if __name__ == "__main__":
    import json
    out = {"Eg_300K_ref": round(EG_300, 3)}
    for n in [1e17, 1e18, 1e19, 1e20]:
        out[f"n={n:.0e}"] = {"BM": round(float(burstein_moss(n)), 4),
                             "BGR": round(float(bgr(n)), 4),
                             "Eg_optical": round(float(optical_eg(n)), 4)}
    print(json.dumps(out, indent=2))
