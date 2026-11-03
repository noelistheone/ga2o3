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
from .levels import trap_depth

# deep-level window (eV from band edges) that traps carriers on device timescales
DEEP_MIN, DEEP_MAX = 0.4, 2.5   # a level 0.4-2.5 eV from a band edge is a "deep" trap
# Declared-in-advance robustness variant (stage3_prereg_acceptor_pdr.md §B2): Eg(300 K)/2.
DEEP_MAX_EG_HALF = float(thermo.band_edges(300.0, 0.40)[0]) / 2.0     # 2.4515 eV

# ── STAGE-3 CORRECTNESS FIX (results/accuracy_baseline/stage3_prereg_acceptor_pdr.md, PART B) ──
# DEEP_MIN/DEEP_MAX were defined at this line and read NOWHERE: `_deep_trap_densities` returned
# N[q<0].sum() and N[q>0].sum(), i.e. EVERY negative and EVERY positive charge state.  Shallow,
# always-ionised dopants were therefore counted as deep traps — measured consequence
# (stage3_diag.json, 1 at%): deep_donor_density for Si-doped = 2.547e20 cm^-3, essentially the
# ionised shallow-Si count itself.
#
# The window is now applied to the ADJACENT charge-transition level that governs each state
# (sandbox.levels.trap_depth):
#     q < 0  hole trap,      needs partner q+1, depth = eps(q+1/q) - Ev(300)
#     q > 0  electron trap,  needs partner q-1, depth = Ec(300)   - eps(q/q-1)
#     q == 0                 no charge -> excluded (unchanged behaviour)
#     no adjacent partner in the DB -> the level lies outside the gap, the state is ALWAYS
#                            IONISED = shallow -> excluded from both censuses.
# Densities are still summed UNWEIGHTED.  The only thing that changed is census MEMBERSHIP.
# The thresholds are the ones already written at device.py:26 before any gate existed, so they
# cannot have been tuned to T3; independent derivations are in §B2 of the pre-registration
# (DEEP_MIN from the 300 K emission time vs the corpus 1e-2..1e1 s photoresponse transients,
# DEEP_MAX = Eg/2 = "lower half of the gap for holes, upper half for electrons").

_DEEP_MASK_CACHE: dict = {}


def deep_masks(db: DefectDB, Ev: float, Ec: float,
               deep_min: float = DEEP_MIN, deep_max: float = DEEP_MAX):
    """Boolean masks (hole_trap, electron_trap) over db's charge states, plus the per-state depth.

    Pure function of (db, Ev, Ec, window) — cached, because it is identical for every forward()
    call at a given quench temperature and costs an O(nCS) Python loop.
    """
    key = (id(db), db.nCS, round(float(Ev), 9), round(float(Ec), 9),
           float(deep_min), float(deep_max))
    hit = _DEEP_MASK_CACHE.get(key)
    if hit is not None:
        return hit
    charge = db.charge
    dHo = db.dHo
    cs_ID = db.cs_ID
    # {defect_id: {q: dHo}} — the adjacent-partner lookup
    by_defect: dict = {}
    for i in range(db.nCS):
        by_defect.setdefault(int(cs_ID[i]), {})[int(charge[i])] = float(dHo[i])
    hole = np.zeros(db.nCS, dtype=bool)
    elec = np.zeros(db.nCS, dtype=bool)
    depth = np.full(db.nCS, np.nan)
    for i in range(db.nCS):
        q = int(charge[i])
        if q == 0:
            continue
        states = by_defect[int(cs_ID[i])]
        partner_q = q + 1 if q < 0 else q - 1
        if partner_q not in states:
            continue                        # always-ionised => shallow => excluded
        kind, d = trap_depth(q, float(dHo[i]), states[partner_q], Ev, Ec)
        depth[i] = d
        if deep_min <= d <= deep_max:
            if kind == "hole":
                hole[i] = True
            else:
                elec[i] = True
    out = (hole, elec, depth)
    _DEEP_MASK_CACHE[key] = out
    return out


def _deep_trap_densities(db: DefectDB, r, deep_min: float = DEEP_MIN,
                         deep_max: float = DEEP_MAX):
    """Deep-acceptor (hole-trap) and deep-donor (electron-trap) densities [cm^-3] from the
    (quenched) charge states, using each state's ADJACENT charge-transition level as the depth."""
    hole, elec, _ = deep_masks(db, r.Ev, r.Ec, deep_min, deep_max)
    deep_acc = float(r.N_cs[hole].sum())
    deep_don = float(r.N_cs[elec].sum())
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
        "deep_window_eV": [DEEP_MIN, DEEP_MAX],
        "deep_window_note": ("adjacent charge-transition level within [0.4, 2.5] eV of its own "
                             "band edge; q=0 and always-ionised (no adjacent partner state) "
                             "charge states excluded as shallow"),
        "PDR_score": pdr_score,
        "PDR_note": "higher = higher photo-to-dark ratio expected (direction/ranking only)",
        "tau_score": tau_score,
        "tau_note": "higher = slower decay expected (class ranking only; absolute τ withheld)",
        "fidelity": "trend/ranking-only (quantitative dark/PDR/τ not credible — see novelty_report)",
    }
