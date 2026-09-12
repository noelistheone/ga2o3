"""full_forward — the integrated (dopant, process) → nine-property forward simulator.

Wires Tier-1 (charge-neutrality) + transport + optical + device layers into one call that emits
all nine corpus properties for a doped β-Ga2O3 film, each tagged with an honest fidelity grade.
This is the integrated chain the novelty audit found unprecedented (docs/novelty_report.md).

Pipeline: solve equilibrium at T_anneal(process) with fixed [dopant] → quench to 300 K →
n, p, [V_O], ionized densities → μ(N_I), σ=e·n·μ → Burstein-Moss Eg/dEg → device FOM directions.
The undoped reference is solved once at matched process for the within-paper dEg differential.
"""
from __future__ import annotations

import numpy as np

from . import engine, transport, optical, device, thermo
from .kroger_db import DefectDB, load


FIDELITY = {
    "hall_n": "quantitative(bulk/epi ~2x); order-of-mag(sputtered films)",
    "V_O": "trend/ordinal-proxy only (XPS O1s metrology disputed)",
    "hall_mu": "near-quantitative single-crystal(~1.5x); order-of-mag(films)",
    "sigma": "order-of-mag; ~2-4x for bulk Sn/Si",
    "Eg": "quantitative for degenerate-donor BM(~0.1 eV); trend otherwise",
    "dEg": "sign/ranking (headline); magnitude for degenerate-donor subset",
    "dark": "trend/ranking only",
    "PDR": "direction/ranking only",
    "tau": "class-ranking only (absolute withheld)",
}


def _ionized_densities(db, q):
    """N_I (total ionized scattering centers) and N_A (ionized acceptors) [cm^-3] from quench."""
    Q = db.charge
    N_I = float((np.abs(Q) * q.N_cs).sum())
    N_A = float((np.clip(-Q, 0, None) * q.N_cs).sum())
    return max(N_I, 1.0), N_A


def _vo_total(db, r):
    o = db.col("O")
    is_vo = (db.dm[:, o] == -1) & (np.abs(db.dm).sum(axis=1) == 1)
    return float(r.N_cs[is_vo].sum())


def forward(dopant, conc_frac, T_anneal=1073.0, pO2=1e-4, EcT_fraction=0.40,
            film=True, E_B_eV=0.02, Nd_bg=1e17, db: DefectDB | None = None, undoped_cache=None):
    """Simulate all nine properties for one (dopant, concentration, process) point.

    dopant: element symbol or 'undoped'. conc_frac: cation fraction (0 for undoped).
    Nd_bg: unintentional shallow-donor background (Si/H contamination); UID Ga2O3 ~1e17.
    Returns a dict with the nine properties + provenance + fidelity grades.
    """
    db = db or load()
    from .kroger_db import N_SITE
    N_cat = N_SITE * 2.0 / 3.0                     # cation site density [cm^-3]
    fixed = {} if dopant == "undoped" or conc_frac <= 0 else {dopant: conc_frac * N_cat}

    eq = engine.solve_single(db, T_anneal, pO2=pO2, EcT_fraction=EcT_fraction,
                             fixed_conc=fixed, fd=True, Nd_bg=Nd_bg)
    q = engine.quench(db, eq, T_quench=300.0, EcT_fraction=EcT_fraction, fd=True, Nd_bg=Nd_bg)

    n, p = float(q.n), float(q.p)
    N_I, N_A = _ionized_densities(db, q)
    N_I = N_I + Nd_bg                              # background donors also scatter
    vo_anneal = _vo_total(db, eq)          # V_O at process T (before quench units)
    vo_quench = _vo_total(db, q)

    mu = transport.mobility(300.0, n_cm3=n, N_I_cm3=N_I, N_A_cm3=N_A, film=film, E_B_eV=E_B_eV)
    sigma = float(transport.conductivity(n, mu["mu_drift"]))

    # undoped reference at matched process (for within-paper dEg)
    if undoped_cache is None and dopant != "undoped":
        und = forward("undoped", 0.0, T_anneal=T_anneal, pO2=pO2, EcT_fraction=EcT_fraction,
                      film=film, E_B_eV=E_B_eV, Nd_bg=Nd_bg, db=db, undoped_cache={})
        n_und = und["hall_n_cm3"]
    elif dopant == "undoped":
        n_und = n
    else:
        n_und = undoped_cache.get("n_und", n)

    eg_opt = float(optical.optical_eg(n))
    deg = optical.dEg(n, max(n_und, 1.0))
    dev = device.device_foms(db, q, sigma, n)

    return {
        "input": {"dopant": dopant, "conc_frac": conc_frac, "T_anneal": T_anneal,
                  "pO2": pO2, "film": film, "E_B_eV": E_B_eV, "EcT_fraction": EcT_fraction},
        "hall_n_cm3": n,
        "hall_p_cm3": p,
        "V_O_cm3_anneal": vo_anneal,
        "V_O_cm3_quench": vo_quench,
        "hall_mu_cm2Vs": mu["mu_hall"],
        "mu_drift_cm2Vs": mu["mu_drift"],
        "sigma_S_cm": sigma,
        "Eg_optical_eV": eg_opt,
        "dEg_eV": deg["dEg"],
        "dark_activation_eV": dev["dark_activation_eV"],
        "PDR_score": dev["PDR_score"],
        "tau_score": dev["tau_score"],
        "N_I_ionized_cm3": N_I,
        "N_A_ionized_cm3": N_A,
        "EF_300K_below_Ec": float(q.Ec - q.EF),
        "solubility_limited": eq.meta.get("solubility_limited", False),
        "mobility_components": mu,
        "dEg_components": deg,
        "device_detail": dev,
        "fidelity": FIDELITY,
    }
