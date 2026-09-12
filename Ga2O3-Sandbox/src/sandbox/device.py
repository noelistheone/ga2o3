"""Device-FOM layer — dark current, PDR, τ as DIRECTION/RANKING outputs (honest scope).

Per the novelty audit + feasibility study, quantitative dark/PDR/τ are NOT credible (capture
cross-sections carry 1-3 orders of systematic error; best published TCAD is a ~6-parameter
postdiction). What IS defensible and never-done: predicting how these FOMs RANK across dopants,
with the trap densities and Fermi level coming from the Tier-1 solve rather than device fits.

  dark current: activation energy E_a = E_C − E_F (from the solve); relative dark conductance
                ∝ σ (bulk) — a resistive-film proxy. Cross-dopant ranking at fixed geometry.
  PDR         : photoconductive gain rises with deep-hole-trap density (deep acceptors trap
                holes, prolong the electron lifetime) and falls with dark current. Score =
                log10(deep_acceptor_density) − log10(dark_proxy). Direction/ranking only.
  tau         : longer with more deep traps; class ranking (acceptor-faster-decay, Phase 68).
                Absolute τ withheld (needs NMP capture coeff + reconfiguration barriers = DFT).

All outputs are labelled trend-only; magnitudes are scores, not physical device values.
"""
from __future__ import annotations

import numpy as np

from . import thermo
from .kroger_db import DefectDB

# deep-level window (eV from band edges) that traps carriers on device timescales
DEEP_MIN, DEEP_MAX = 0.4, 2.5   # a level 0.4-2.5 eV from a band edge is a "deep" trap


def _deep_trap_densities(db: DefectDB, r):
    """Deep-acceptor and deep-donor trap densities [cm^-3] from the (quenched) charge states,
    using each defect's charge-transition level position within the gap as the depth proxy."""
    # a charge state that is negative and abundant with a mid-gap level acts as a hole trap
    q = db.charge
    N = r.N_cs
    deep_acc = float(N[(q < 0)].sum())     # negative centers (hole traps / acceptors)
    deep_don = float(N[(q > 0)].sum())     # positive centers
    # restrict "deep" by excluding the shallow-donor manifold (charge +1, very low formation):
    return deep_acc, deep_don


def device_foms(db: DefectDB, quenched, sigma_S_cm, n_cm3):
    """Return direction/ranking scores for dark current, PDR, tau. `quenched` = EqResult at 300K."""
    Ea_dark = float(quenched.Ec - quenched.EF)             # activation energy [eV]
    # resistive-film dark proxy: conductance ∝ σ (higher σ → higher dark current)
    dark_proxy = float(sigma_S_cm)
    deep_acc, deep_don = _deep_trap_densities(db, quenched)
    # PDR score: gain ↑ with deep hole traps, ↓ with dark conductance
    pdr_score = float(np.log10(max(deep_acc, 1.0)) - np.log10(max(dark_proxy, 1e-12)))
    # tau score: ↑ with deep trap density (log10 of total deep centers)
    tau_score = float(np.log10(max(deep_acc + deep_don, 1.0)))
    return {
        "dark_activation_eV": Ea_dark,
        "dark_proxy_sigma_S_cm": dark_proxy,
        "dark_rank_note": "higher σ / lower E_a → higher dark current (ranking only)",
        "deep_acceptor_density": deep_acc,
        "deep_donor_density": deep_don,
        "PDR_score": pdr_score,
        "PDR_note": "higher = higher photo-to-dark ratio expected (direction/ranking only)",
        "tau_score": tau_score,
        "tau_note": "higher = slower decay expected (class ranking only; absolute τ withheld)",
        "fidelity": "trend/ranking-only (quantitative dark/PDR/τ not credible — see novelty_report)",
    }
