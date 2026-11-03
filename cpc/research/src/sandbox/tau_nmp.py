"""τ from the configuration-coordinate (CC) diagram — nonradiative-multiphonon (NMP) capture.

Turns the DFT CC-diagram quantities (from dft/qe_cc/) into a carrier-capture barrier and an
SRH-style lifetime, in the 1-D effective-mode (Marcus/Alkauskas) picture:

  configuration coordinate ΔQ  [amu^½ Å]   — mass-weighted displacement between the two
                                              charge-state minima
  Franck-Condon relaxation energy λ [eV]    — E of the OTHER charge state evaluated at this
                                              minimum, minus its own minimum (the two dS_FC)
  level offset ΔE [eV]                       — thermodynamic transition-level position vs the band
  effective phonon ħω [eV]                   — curvature of the CC parabola (ω = sqrt(2λ)/ΔQ · ...)

Classical (Marcus) capture barrier:  ΔE_b = (λ + ΔE)² / (4λ)    (crossing-point energy)
Capture coefficient (thermally activated):  C ≈ C0 · exp(−ΔE_b / kT)
SRH lifetime for that trap:  τ = 1 / (C · N_trap)  ;  or via σ = C / v_th.

C0 folds the electron-phonon coupling and attempt frequency; without the full Alkauskas matrix
element we bound C0 by the phonon attempt rate × capture volume (order-of-magnitude, honest per
the τ fidelity grade). Absolute τ is order-of-magnitude / class-ranking only.
"""
from __future__ import annotations

import numpy as np

KB_EV = 8.617333262e-5
HBAR_EVS = 6.582119569e-16      # eV·s
V_TH_300 = 2.0e7               # cm/s, electron thermal velocity in Ga2O3 (~m*=0.28)


def marcus_barrier(lam, dE):
    """Classical CC crossing-point (capture) barrier [eV]. lam=reorganization energy, dE=driving
    force. ΔE_b = (λ+ΔE)²/(4λ). VALID ONLY in the strong-coupling regime (λ ≳ 0.2 eV): for small
    λ (weak coupling, e.g. localized d-d transitions like Fe_Ga where the lattice barely relaxes)
    the classical formula diverges and is unphysical — the quantum weak-coupling multiphonon rate
    is required instead. Callers should check `strong_coupling_valid(lam)`."""
    lam = max(float(lam), 1e-6)
    return max((lam + float(dE)) ** 2 / (4.0 * lam), 0.0)


def strong_coupling_valid(lam, threshold=0.2):
    """The classical Marcus/NMP barrier is only trustworthy for λ ≳ 0.2 eV (strong e-phonon
    coupling). Below that, this 1D model breaks down (Fe_Ga-type weak-coupling centers)."""
    return float(lam) >= threshold


def effective_phonon(lam, dQ):
    """Effective phonon energy ħω [eV] from λ and ΔQ (harmonic CC): λ = ½ Mω²ΔQ² → but with ΔQ
    in amu^½·Å, ħω[eV] ≈ sqrt(2λ/ (ΔQ² )) × unit factor. Use the standard: ħω = ħ·sqrt(2λ/(ΔQ²·u))."""
    dQ = max(float(dQ), 1e-6)
    # 2λ / ΔQ² has units eV / (amu Å²); convert to (rad/s)² then eV:
    # ω² = (2λ[J]) / (ΔQ²[kg m²]); ħω[eV] below. amu=1.6605e-27 kg, Å=1e-10 m, eV=1.602e-19 J
    two_lam_J = 2.0 * float(lam) * 1.602176634e-19
    dQ2_kgm2 = (dQ ** 2) * 1.66053907e-27 * (1e-10) ** 2
    omega = np.sqrt(two_lam_J / dQ2_kgm2)             # rad/s
    return float(HBAR_EVS * omega)                     # eV


def capture_coefficient(barrier_eV, T=300.0, C0_cm3_s=1e-7):
    """C = C0·exp(−ΔE_b/kT) [cm³/s]. C0 ~ v_th·σ_geometric ~ 1e-7 cm³/s (order-of-magnitude)."""
    return C0_cm3_s * np.exp(-barrier_eV / (KB_EV * T))


def srh_tau(C_cm3_s, N_trap_cm3):
    """Minority-carrier SRH lifetime τ = 1/(C·N_trap) [s]."""
    return 1.0 / max(C_cm3_s * max(N_trap_cm3, 1.0), 1e-99)


def tau_from_cc(dQ, lam, dE, N_trap_cm3, T=300.0, C0=1e-7):
    """Full CC→τ chain. Returns barrier, ħω, capture coeff, cross-section, τ (all order-of-mag)."""
    barrier = marcus_barrier(lam, dE)
    hw = effective_phonon(lam, dQ)
    C = capture_coefficient(barrier, T, C0)
    sigma = C / V_TH_300                                # cm²
    tau = srh_tau(C, N_trap_cm3)
    valid = strong_coupling_valid(lam)
    return {
        "strong_coupling_valid": valid,
        "model_note": ("" if valid else "λ<0.2 eV: WEAK COUPLING — classical Marcus barrier "
                       "unreliable (formula diverges); τ below is NOT trustworthy for this defect"),
        "capture_barrier_eV": round(barrier, 3),
        "effective_phonon_meV": round(hw * 1000, 1),
        "capture_coeff_cm3_s": C,
        "capture_cross_section_cm2": sigma,
        "SRH_tau_s": tau,
        "fidelity": "order-of-magnitude / class-ranking only (C0 bounded, not full Alkauskas ME)",
    }


if __name__ == "__main__":
    import json
    # illustrative: V_O-like deep trap, ΔQ~2 amu^½Å, λ~0.5 eV, dE~0.3 eV, N_trap~1e17
    print(json.dumps(tau_from_cc(dQ=2.0, lam=0.5, dE=0.3, N_trap_cm3=1e17), indent=2, default=str))
