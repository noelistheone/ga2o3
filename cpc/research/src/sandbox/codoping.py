"""codoping — co-doped forward simulator + thin re-export of the multi-element equilibrium solve.

HISTORY (2026-07-28 correctness fix): the nested multi-element solver used to live here and
engine.solve_single kept its own broken joint scipy.fsolve branch, so there were TWO
implementations. The solver has been MOVED INTO src/sandbox/engine.py (engine.solve_codoped) and
engine.solve_single now delegates to it for len(fixed_conc) >= 2. This module keeps the public
names `solve_codoped` / `_Ctx` / `_bisect_mu` / ... as re-exports so existing scripts keep
working, and still owns the co-doped nine-property forward simulator.

Import direction: engine (numerics, dependency-free) <- codoping (forward stack). The reverse
(engine importing this module) would be circular, because this module pulls in full_forward,
which imports engine.

Solver summary (now in engine.py):
  inner : E_F by MONOTONE BISECTION on a wide fixed window [-40, +45] eV.
  outer : damped Newton on log10-concentration residuals + Gauss-Seidel bisection fallback.
  start : CONTINUATION from each element's converged single-dopant mu.
  gate  : explicit post-solve validity check (engine._validity); fsolve's `ier` was never a
          residual test and was never read.
Site blocking: BOTH branches now locate the 10%-site-occupancy cap by the same bisection
primitive (engine._site_blocking_cap) and report it per element.

Single-dopant equivalence: forward_codoped({X: c}) and full_forward.forward(X, c) go through the
SAME solver (engine.solve_single, len==1 branch), the SAME cation-site density
(full_forward.N_CAT_SITE, imported not recomputed) and the SAME V_O definition, so their nine
properties agree bit-for-bit.  This held before 2026-07-28, was silently broken for one day by a
half-applied N_cat correction (this module kept N_SITE*2/3 while full_forward moved to 2*N_SITE
= a factor 3 in the dissolved dopant density), and is restored + re-checked by
scripts/probe_codoping_module.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import engine, thermo, transport, optical, device
from .engine import (MU_OFF, MU_HI, EF_LO, EF_HI, OCC_CAP, DefectEquilibriumError,
                     _Ctx, _state, _g_vec, _mu_cap, _bisect_mu, _single_dopant_mu,
                     solve_codoped)
from . import full_forward as _ff
from .full_forward import (FIDELITY, N_CAT_SITE, VO_DEFINITION, _ionized_densities,
                           _vo_native, _vo_total_oxygen_deficiency,
                           # STAGE-3: the abstention / carrier-regime / incorporation-ceiling
                           # blocks are IMPORTED from full_forward, never re-implemented here, so
                           # forward_codoped({X: c}) stays bit-identical to forward(X, c).
                           apply_incorporation_ceiling, compensation_block,
                           carrier_regime_block, dominant_compensating_family,
                           _defect_only_margin)
from .kroger_db import DefectDB, N_SITE, load

# full_forward owns BOTH the cation-site density (N_CAT_SITE) and the V_O definition; they are
# IMPORTED, never recomputed here, so `conc_frac` means exactly the same thing in
# forward_codoped as in full_forward.forward and the two can be compared number for number.
#
# BUG FIXED 2026-07-28 (second pass): the 2026-07-28 N_cat correction (N_SITE*2/3 -> 2*N_SITE)
# landed in full_forward.py only; this module kept `N_cat = N_SITE * 2.0 / 3.0`, so the two
# forward simulators in one package dissolved 3x different dopant densities for the same
# nominal at% (measured at 1 at%, 1350 K: Fe hall_n 1.9687e2 via forward vs 5.9091e2 via
# forward_codoped).  Likewise `_vo_total` was resolved by a getattr chain that silently
# selected native-only or total-oxygen-deficiency depending on which full_forward was loaded;
# it is now an explicit import that fails loudly under a partial rollback.
_vo_total = _vo_total_oxygen_deficiency          # headline V_O (see full_forward.FIDELITY)


# ────────────────────────────────────────────────────────────────────────────
# hardened quench (falls back only when engine.quench's bracket fails)
# ────────────────────────────────────────────────────────────────────────────
def quench_hardened(db: DefectDB, anneal: engine.EqResult, T_quench: float = 300.0,
                    EcT_fraction: float = 0.40, fd: bool = False, sth_flag: float = 1.0,
                    frozen_defect_totals=None, Nd_bg: float = 0.0, Na_bg: float = 0.0):
    """engine.quench first (bit-identical when it works); on a bracket failure, redo the E_F
    root find by monotone bisection over [-40, +45] eV and CLAMP rather than raise."""
    try:
        r = engine.quench(db, anneal, T_quench=T_quench, EcT_fraction=EcT_fraction, fd=fd,
                          sth_flag=sth_flag, frozen_defect_totals=frozen_defect_totals,
                          Nd_bg=Nd_bg, Na_bg=Na_bg)
        r.meta["quench_bracket_failed"] = False
        return r
    except (ValueError, DefectEquilibriumError):
        # engine.quench now scans-and-clamps its own bracket and raises the NAMED
        # DefectEquilibriumError (with the offending frozen totals) only when the neutrality
        # condition genuinely has no root in [-40, +45] eV; the clamp below is kept so that
        # legacy callers of this wrapper still get a flagged result instead of a crash.
        pass
    T = float(T_quench)
    Eg, Ev, Ec = (float(x) for x in thermo.band_edges(T, EcT_fraction))
    Nc, Nv = float(thermo.Nc(T)), float(thermo.Nv(T))
    kT = thermo.KB_EV * T
    totals = anneal.N_defect if frozen_defect_totals is None else frozen_defect_totals
    s = float(thermo.dSvib_norm(T))
    dG_rel0 = db.dHo - 3.0 * kT * s * db.sum_dm
    groups = [np.where(db.cs_ID - 1 == d)[0] for d in range(db.nD)]

    def N_cs_of(EF):
        dG_rel = dG_rel0 + db.charge * EF
        N_cs = np.zeros(db.nCS)
        for d, idx in enumerate(groups):
            if totals[d] <= 0 or idx.size == 0:
                continue
            w = np.exp(np.clip(-(dG_rel[idx] - dG_rel[idx].min()) / kT, -700, 300))
            N_cs[idx] = totals[d] * w / w.sum()
        return N_cs

    def Q(EF):
        N_cs = N_cs_of(EF)
        n, p, s1, s2 = engine._carriers(EF, kT, Ec, Ev, Nc, Nv, sth_flag, fd)
        return float((db.charge * N_cs).sum() + p + s1 + s2 - n + Nd_bg - Na_bg)

    lo, hi, clamped = EF_LO, EF_HI, False
    if not (Q(lo) > 0):
        EF, clamped = lo, True
    elif not (Q(hi) < 0):
        EF, clamped = hi, True
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if Q(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        EF = 0.5 * (lo + hi)
    N_cs = N_cs_of(EF)
    n, p, sth1, sth2 = engine._carriers(EF, kT, Ec, Ev, Nc, Nv, sth_flag, fd)
    N_defect = np.zeros(db.nD)
    np.add.at(N_defect, db.cs_ID - 1, N_cs)
    return engine.EqResult(T=T, EF=float(EF), n=float(n), p=float(p), sth1=float(sth1),
                           sth2=float(sth2), N_cs=N_cs, N_defect=N_defect,
                           mu_vec=anneal.mu_vec, Ev=Ev, Ec=Ec, Eg=Eg, charge_residual=Q(EF),
                           meta={"quench_from_T": anneal.T, "fd": fd,
                                 "quench_bracket_failed": True,
                                 "quench_EF_clamped": bool(clamped)})


# ────────────────────────────────────────────────────────────────────────────
# co-doped forward simulator — same nine properties as full_forward.forward
# ────────────────────────────────────────────────────────────────────────────
def forward_codoped(dopants: dict, T_anneal=1073.0, pO2=1e-4, EcT_fraction=0.40,
                    film=True, E_B_eV=0.02, Nd_bg=1e17, db: DefectDB | None = None,
                    undoped_cache=None):
    """Nine-property forward simulation for an arbitrary dopant MIXTURE.

    dopants: {element: cation_fraction}. {} (or all-zero) = undoped.

    KEY CONTRACT (verified by scripts/probe_codoping_module.py): emits every key
    full_forward.forward emits — including the V_O_native_* / V_O_total_incl_complexes_* /
    V_O_definition provenance — except that the provenance block "input" carries "dopants"
    (a dict) in place of "dopant"/"conc_frac", and one EXTRA key "solver_meta" (the co-doped
    solver diagnostics, which have no single-dopant counterpart).
    conc_frac is interpreted with full_forward.N_CAT_SITE, the same constant forward() uses.
    transport / optical / device are reused VERBATIM.
    """
    db = db or load()
    N_cat = N_CAT_SITE                             # cation site density [cm^-3]; ONE definition,
                                                   # imported from full_forward (see module head)
    dopants = {k: float(v) for k, v in (dopants or {}).items()}
    fixed = {el: c * N_cat for el, c in dopants.items() if c > 0}
    # STAGE-3 DESIGN 4: same sourced incorporation ceiling as full_forward.forward, applied
    # per-element (Sn only has an entry), so the single-dopant equivalence contract holds.
    fixed, ceiling_info = apply_incorporation_ceiling(fixed)

    eq = solve_codoped(db, T_anneal, pO2=pO2, EcT_fraction=EcT_fraction,
                       fixed_conc=fixed, fd=True, Nd_bg=Nd_bg)
    if len(fixed) <= 1:
        q = engine.quench(db, eq, T_quench=300.0, EcT_fraction=EcT_fraction, fd=True,
                          Nd_bg=Nd_bg)
    else:
        q = quench_hardened(db, eq, T_quench=300.0, EcT_fraction=EcT_fraction, fd=True,
                            Nd_bg=Nd_bg)

    n, p = float(q.n), float(q.p)
    N_I, N_A = _ionized_densities(db, q)
    N_I = N_I + Nd_bg                              # background donors also scatter
    # Same V_O bookkeeping as full_forward.forward: headline = total oxygen deficiency,
    # isolated-native count retained alongside.
    vo_nat_anneal = _vo_native(db, eq)
    vo_nat_quench = _vo_native(db, q)
    vo_tot_anneal = _vo_total_oxygen_deficiency(db, eq)
    vo_tot_quench = _vo_total_oxygen_deficiency(db, q)
    vo_anneal = vo_tot_anneal
    vo_quench = vo_tot_quench

    mu = transport.mobility(300.0, n_cm3=n, N_I_cm3=N_I, N_A_cm3=N_A, film=film, E_B_eV=E_B_eV)
    sigma = float(transport.conductivity(n, mu["mu_drift"]))

    # undoped reference at matched process (for within-paper dEg)
    if undoped_cache is None and fixed:
        und = forward_codoped({}, T_anneal=T_anneal, pO2=pO2, EcT_fraction=EcT_fraction,
                              film=film, E_B_eV=E_B_eV, Nd_bg=Nd_bg, db=db, undoped_cache={})
        n_und = und["hall_n_cm3"]
    elif not fixed:
        n_und = n
    else:
        n_und = undoped_cache.get("n_und", n)

    eg_opt = float(optical.optical_eg(n))
    deg = optical.dEg(n, max(n_und, 1.0))
    dev = device.device_foms(db, q, sigma, n)

    # ── STAGE-3 honesty blocks — the SAME functions full_forward.forward calls ──
    comp = compensation_block(db, q, T_anneal, Nd_bg=Nd_bg, Na_bg=0.0, n_anneal=float(eq.n))
    creg = carrier_regime_block(n, p, sigma)
    _els = list(fixed)
    _one = _els[0] if len(_els) == 1 else None

    return {
        "input": {"dopants": dopants, "T_anneal": T_anneal, "pO2": pO2, "film": film,
                  "E_B_eV": E_B_eV, "EcT_fraction": EcT_fraction,
                  "N_cat_site_cm3": N_cat,
                  "dopant_loading_cm3": {el: float(v) for el, v in fixed.items()},
                  "dopant_achieved_cm3": {el: float(v) for el, v
                                          in eq.meta.get("achieved_cm3", {}).items()},
                  "dopant_target_hit_dex": {el: float(v) for el, v
                                            in eq.meta.get("target_hit_dex", {}).items()}},
        "hall_n_cm3": n,  # deprecated legacy name; retained for numerical compatibility
        "free_electron_cm3": n,
        "hall_density_cm3": n / mu["r_hall"],
        "hall_factor_rH": mu["r_hall"],
        "measurement_temperature_K": 300.0,
        "carrier_definition": "single-electron approximation: n_H=n/r_H; legacy hall_n_cm3=n",
        "hall_p_cm3": p,
        "V_O_cm3_anneal": vo_anneal,          # == V_O_total_incl_complexes_cm3_anneal
        "V_O_cm3_quench": vo_quench,          # == V_O_total_incl_complexes_cm3_quench
        "V_O_native_cm3_anneal": vo_nat_anneal,
        "V_O_native_cm3_quench": vo_nat_quench,
        "V_O_total_incl_complexes_cm3_anneal": vo_tot_anneal,
        "V_O_total_incl_complexes_cm3_quench": vo_tot_quench,
        "V_O_definition": VO_DEFINITION,
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
        "site_blocking_limited": eq.meta.get("site_blocking_limited", False),
        "solubility_limited": eq.meta.get("solubility_limited", False),
        "mobility_components": mu,
        "dEg_components": deg,
        "device_detail": dev,
        "fidelity": FIDELITY,
        # ── STAGE-3 DESIGN 1 — compensation-margin abstention ──
        **comp,
        # ── STAGE-3 DESIGN 3 — acceptor-floor honesty ──
        **creg,
        # ── STAGE-3 DESIGN 4 — sourced incorporation ceiling.  For 0 or 1 dopant the three
        #    fields are SCALARS with exactly the shapes full_forward.forward emits (that is the
        #    single-dopant bit-identity contract); for a true mixture they are per-element dicts,
        #    which has no single-dopant counterpart to be identical to.
        "above_reported_incorporation_ceiling": ceiling_info[
            "above_reported_incorporation_ceiling"],
        "requested_dopant_cm3": (float(ceiling_info["requested_dopant_cm3"].get(_one, 0.0))
                                 if len(_els) <= 1 else
                                 dict(ceiling_info["requested_dopant_cm3"])),
        "ceiling_cm3": (ceiling_info["ceiling_cm3"].get(_one)
                        if len(_els) <= 1 else dict(ceiling_info["ceiling_cm3"])),
        "ceiling_source_doi": (ceiling_info["ceiling_source_doi"].get(_one)
                               if len(_els) <= 1 else dict(ceiling_info["ceiling_source_doi"])),
        # ── STAGE-3 DESIGN 4 rank-2 — report-only compensation diagnostics ──
        "charge_cancellation_margin_defect_only": _defect_only_margin(db, q),
        "dominant_compensating_family": dominant_compensating_family(db, q),
        # anneal solver diagnostics + the QUENCH honesty flags.  quench_hardened() downgrades a
        # DefectEquilibriumError to a CLAMPED E_F; without these keys a clamped quench would
        # reach the nine properties with no trace (the nine are all read off the quench).
        "solver_meta": dict({k: v for k, v in eq.meta.items() if k != "outer_hist"},
                            quench_bracket_failed=bool(q.meta.get("quench_bracket_failed",
                                                                  False)),
                            quench_EF_clamped=bool(q.meta.get("quench_EF_clamped", False))),
    }
