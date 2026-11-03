"""ratio_codoping — a fixed stoichiometric RATIO as a FIRST-CLASS constraint on the nested
multi-element defect-equilibrium solve.

*** WHAT THIS ADDS ***
Before this module, "co-doping" was expressible only as two INDEPENDENT ABSOLUTE targets:
`fixed_conc = {"Si": 3.8e20, "Mg": 3.8e20}` pins both totals. That is the right description of a
sample whose composition has been MEASURED (SIMS on the finished film). It is the wrong
description of a PROCESS in which the two species are supplied at a fixed stoichiometric ratio
(a 9:1 co-sputter, a mixed-precursor MOCVD line, a compound target) and the ABSOLUTE level is
whatever the thermodynamics delivers.

    ABSOLUTE mode  : [X] = C_X and [Y] = C_Y            -> 2 unknowns, 2 absolute rows
    RATIO    mode  : [X]/[Y] = r, level set by an anchor -> the level FLOATS

The anchor is either
    (a) one ABSOLUTE target on some element of the system, or
    (b) a RESERVOIR: an element's mu PINNED by a compound tie (sandbox.compound_reservoir),
        in which case that element is not an unknown at all and the whole system's absolute
        level is set by the compound. See the SYSTEMATIC warning below — (b) is not free.

*** ADDITIVITY — STRUCTURAL, NOT ASSERTED ***
`src/sandbox/engine.py` is NOT edited by this module and was not opened for edit. This file
IMPORTS engine's primitives (`_Ctx`, `_state`, `_bisect_mu`, `_mu_cap`, `_single_dopant_mu`,
`_validity`, `solve_EF`) and reuses them verbatim: the inner E_F monotone bisection, the
outer damped-Newton/Gauss-Seidel machinery, the site-blocking cap and the post-solve validity
gate are the SAME code paths `engine.solve_codoped` runs. Only the RESIDUAL VECTOR differs —
a ratio row replaces an absolute row. Existing calls are therefore bit-identical by
construction, not by inspection (verified anyway: results/mathwork/_round20_ratio_bitprobe.py).

*** STAGE TAGS ARE MANDATORY ***
Every carrier density this module emits carries `stage`. A carrier density without its stage is
not a claim (Rounds 14/15). The default is `stage="anneal"`, the well-conditioned stage
(Rounds 8d/9); the quenched value is emitted alongside, explicitly tagged, and is never the
default.

*** UNGATED — READ THIS BEFORE USING ANY NUMBER FROM HERE ***
The ratio constraint has NOT been confronted with any measurement. GATE-RATIO-1
(results/mathwork/PREREG_round20_ratio.md §4) censused BOTH corpora and found 0 prediction
blocks against a pre-registered floor of 5 => UNTESTABLE-ON-THIS-CORPUS => SILENT. Call
`ungated_marker()` for the exact measurement that would gate it. Treat every number from this
module as a MECHANISM DEMONSTRATION, not a claim.

*** TWO PHYSICS LIMITS THAT ARE NOT NUMERICAL AND CANNOT BE SOLVED AWAY ***
 1. RESERVOIR-ANCHORED ratios inherit the FULL Round-12e 1.300 eV energetics systematic with no
    self-consistent damping (Round-20 Task 1 measured d log10[X]/d eV = -4.70 = -1/(kT ln10) at
    1073 K, i.e. +-6.1 dex on the incorporated concentration at +-1.3 eV). Absolute-anchored
    ratios do NOT: the solved mu absorbs the shift (measured 2.7e-15 dex/eV). Carried here as
    the named field `reservoir_reference_systematic_eV`.
 2. The KROGER DB carries 76 two-dopant charge states over 9 pairs, and 7 of those 9 pairs
    contain H. It has NO donor-acceptor pair for Si-Mg, Sn-Mg, Zn-Mg or Si-Sn — the pairs a
    lab actually co-doping beta-Ga2O3 cares about. Round-20 re-verified the cost of that
    omission: <= 0.181 dex on n at FIXED TOTAL CONTENT, but up to 109.08x (2.038 dex) on the
    INCORPORATED TOTAL at FIXED CHEMICAL POTENTIAL. Fixed chemical potential is exactly what
    reservoir mode is. See `pair_omission_warning()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import engine, thermo
from .engine import (MU_OFF, MU_HI, OCC_CAP, TOL_DEX, OCC_AT_CAP_RTOL,
                     DefectEquilibriumError, EqResult,
                     _Ctx, _state, _bisect_mu, _mu_cap, _single_dopant_mu, _validity,
                     _raise_if_invalid)
from .kroger_db import DefectDB

# Round 12e (docs/RESEARCH_LOG.md §12e): the MEASURED inter-setup systematic on Ga-containing
# formation energies. Same constant compound_reservoir.py carries; imported there, restated here
# so this module has one source of truth even if imported alone.
SIGMA_ENERGETICS_MEASURED_EV = 1.300

# The four pairs a lab co-doping beta-Ga2O3 actually asks about, and which the DB does not have.
LAB_PAIRS_OF_INTEREST = [("Si", "Mg"), ("Sn", "Mg"), ("Zn", "Mg"), ("Si", "Sn")]


# ════════════════════════════════════════════════════════════════════════════════════════════
# constraint rows
# ════════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class AbsRow:
    """log10([element]) - log10(target_cm3) = 0."""
    element: str
    target_cm3: float

    def elements(self):
        return (self.element,)

    def label(self):
        return f"[{self.element}] = {self.target_cm3:.6g} cm^-3"


@dataclass(frozen=True)
class RatioRow:
    """log10([num]) - log10([den]) - log10(r) = 0,  i.e.  [num]/[den] = r."""
    num: str
    den: str
    r: float

    def elements(self):
        return (self.num, self.den)

    def label(self):
        return f"[{self.num}]/[{self.den}] = {self.r:.6g}"


def _g_rows(rows, tot: dict) -> np.ndarray:
    """Residual vector, all rows in log10 so they are dimensionless and equally scaled."""
    g = np.empty(len(rows), dtype=float)
    for i, row in enumerate(rows):
        if isinstance(row, AbsRow):
            g[i] = (np.log10(max(tot[row.element], 1e-30))
                    - np.log10(max(float(row.target_cm3), 1e-30)))
        else:
            g[i] = (np.log10(max(tot[row.num], 1e-30))
                    - np.log10(max(tot[row.den], 1e-30))
                    - np.log10(float(row.r)))
    return g


# ════════════════════════════════════════════════════════════════════════════════════════════
# implied targets — used ONLY for the continuation seed and the site-blocking cap
# ════════════════════════════════════════════════════════════════════════════════════════════
def _implied_targets(rows, free, pinned_tot: dict) -> dict:
    """Propagate absolute targets through the ratio rows.

    These are SEEDS and CAP references, never constraints: the solve is driven by `_g_rows`.
    A ratio row whose denominator's level is still unknown simply does not propagate on this
    pass; the loop runs until nothing new appears.
    """
    tgt = dict(pinned_tot)
    for row in rows:
        if isinstance(row, AbsRow):
            tgt[row.element] = float(row.target_cm3)
    for _ in range(len(rows) + 2):
        moved = False
        for row in rows:
            if isinstance(row, RatioRow):
                if row.den in tgt and row.num not in tgt:
                    tgt[row.num] = float(row.r) * tgt[row.den]
                    moved = True
                elif row.num in tgt and row.den not in tgt:
                    tgt[row.den] = tgt[row.num] / float(row.r)
                    moved = True
        if not moved:
            break
    return {e: v for e, v in tgt.items() if e in free}


def _row_assignment(rows, free):
    """Bijection rows -> free elements, for the Gauss-Seidel fallback only.

    Greedy augmenting-path matching (rows are few). Returns None when no perfect matching
    exists, in which case the Gauss-Seidel fallback is DISABLED and recorded as disabled —
    never silently skipped.
    """
    free_idx = {e: j for j, e in enumerate(free)}
    adj = [[free_idx[e] for e in row.elements() if e in free_idx] for row in rows]
    match_of_free = [-1] * len(free)

    def try_assign(i, seen):
        for j in adj[i]:
            if seen[j]:
                continue
            seen[j] = True
            if match_of_free[j] == -1 or try_assign(match_of_free[j], seen):
                match_of_free[j] = i
                return True
        return False

    for i in range(len(rows)):
        if not try_assign(i, [False] * len(free)):
            return None
    out = {}
    for j, i in enumerate(match_of_free):
        if i == -1:
            return None
        out[i] = free[j]
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════
# result container
# ════════════════════════════════════════════════════════════════════════════════════════════
@dataclass
class RatioResult:
    eq: EqResult                      # the anneal-stage equilibrium state
    rows: list
    free_elements: list
    pinned_mu_eV: dict
    mu_solved_eV: dict
    achieved_cm3: dict
    achieved_ratios: dict
    residual_dex: dict
    converged: bool
    stage: str = "anneal"
    n_cm3: float = float("nan")       # stage-tagged (see `stage`)
    n_quench_cm3: float = float("nan")
    p_cm3: float = float("nan")
    EF_eV: float = float("nan")
    max_site_occupancy: float = float("nan")
    site_blocking_limited_per_element: dict = field(default_factory=dict)
    anchor_mode: str = "absolute"
    reservoir_reference_systematic_eV: float | None = None
    UNGATED: bool = True
    ungated_reason: str = ""
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "n_cm3": float(self.n_cm3),
            "n_cm3_stage": self.stage,
            "n_quench_cm3": float(self.n_quench_cm3),
            "n_quench_cm3_stage": "quench",
            "p_cm3": float(self.p_cm3),
            "EF_eV": float(self.EF_eV),
            "rows": [r.label() for r in self.rows],
            "free_elements": list(self.free_elements),
            "pinned_mu_eV": {k: float(v) for k, v in self.pinned_mu_eV.items()},
            "mu_solved_eV": {k: float(v) for k, v in self.mu_solved_eV.items()},
            "achieved_cm3": {k: float(v) for k, v in self.achieved_cm3.items()},
            "achieved_ratios": {k: float(v) for k, v in self.achieved_ratios.items()},
            "residual_dex": {k: float(v) for k, v in self.residual_dex.items()},
            "converged": bool(self.converged),
            "max_site_occupancy": float(self.max_site_occupancy),
            "site_blocking_limited_per_element": dict(self.site_blocking_limited_per_element),
            "anchor_mode": self.anchor_mode,
            "reservoir_reference_systematic_eV": self.reservoir_reference_systematic_eV,
            "UNGATED": bool(self.UNGATED),
            "ungated_reason": self.ungated_reason,
            "meta": self.meta,
        }


# ════════════════════════════════════════════════════════════════════════════════════════════
# THE SOLVER
# ════════════════════════════════════════════════════════════════════════════════════════════
def solve_ratio(db: DefectDB, T: float, pO2: float = 0.2, EcT_fraction: float = 0.40,
                rows: list | None = None, mu_pinned: dict | None = None,
                fd: bool = False, sth_flag: float = 1.0,
                Nd_bg: float = 0.0, Na_bg: float = 0.0,
                tol_dex: float = 1e-6, itmax: int = 60,
                enforce_cap: bool = True, strict: bool = True,
                stage: str = "anneal", anchor_mode: str = "absolute",
                quench_T: float | None = 300.0) -> RatioResult:
    """Nested solve with an arbitrary mix of ABSOLUTE and RATIO constraint rows.

    rows      : list of AbsRow / RatioRow. `len(rows)` must equal the number of FREE elements.
    mu_pinned : {element: mu_eV} — pinned elements are NOT unknowns and contribute no row.
                This is how a RESERVOIR anchor enters (mu from a compound tie).

    Numerics are engine's, unchanged: monotone E_F bisection inside, damped Newton with a
    Gauss-Seidel monotone-bisection fallback outside, engine's site-blocking cap, engine's
    validity gate. See the module docstring.
    """
    rows = list(rows or [])
    mu_pinned = {k: float(v) for k, v in (mu_pinned or {}).items()}
    if not rows:
        raise ValueError("solve_ratio requires at least one constraint row")

    free = []
    for row in rows:
        for e in row.elements():
            if e not in mu_pinned and e not in free:
                free.append(e)
    if len(rows) != len(free):
        raise ValueError(
            f"ill-posed constraint system: {len(rows)} rows vs {len(free)} free elements "
            f"{free}. Every free element needs exactly one independent row; pin an element "
            f"with mu_pinned (a reservoir anchor) or add an absolute row.")

    ctx = _Ctx(db, T, pO2, EcT_fraction, fd, sth_flag, Nd_bg, Na_bg, mu_pinned)
    all_elems = list(dict.fromkeys(list(free) + list(mu_pinned)))
    cols = {e: db.col(e) for e in all_elems}

    # ── pinned-only reference state: gives the pinned elements' totals, which anchor the
    #    implied-target propagation when the anchor is a RESERVOIR rather than an absolute row.
    pinned_tot = {}
    if mu_pinned:
        _, _, N0, tot0, _ = _state(ctx, {e: MU_OFF for e in free}, cols)
        pinned_tot = {e: float(tot0[e]) for e in mu_pinned}

    implied = _implied_targets(rows, free, pinned_tot)

    # ── continuation seed: each free element's converged SINGLE-dopant mu at its implied target
    seed, seed_lim, seed_failed = {}, {}, []
    for e in free:
        if e in implied and implied[e] > 0:
            m, lim, ok = _single_dopant_mu(db, e, implied[e], T, pO2, EcT_fraction, fd,
                                           sth_flag, Nd_bg, Na_bg, mu_pinned)
            seed[e], seed_lim[e] = m, lim
            if not ok:
                seed_failed.append(e)
        else:
            seed[e], seed_lim[e] = -6.0, False
            seed_failed.append(e)
    mu = np.array([seed[e] for e in free], dtype=float)

    assign = _row_assignment(rows, free)

    def evaluate(mu_arr):
        mu_map = {e: float(m) for e, m in zip(free, mu_arr)}
        EF, mu_vec, N_cs, tot, ok = _state(ctx, mu_map, cols)
        return _g_rows(rows, tot), EF, mu_vec, N_cs, tot, ok

    hist, method = [], "newton"
    g, EF, mu_vec, N_cs, tot, bracketed = evaluate(mu)
    nrm = float(np.max(np.abs(g)))
    for it in range(itmax):
        hist.append({"it": it, "mu": [float(v) for v in mu], "EF": float(EF),
                     "max_abs_dex": nrm, "step": method})
        if nrm < tol_dex:
            break
        h = 1e-3
        J = np.zeros((len(rows), len(free)))
        for j in range(len(free)):
            mp = mu.copy(); mp[j] += h
            gp, *_ = evaluate(mp)
            J[:, j] = (gp - g) / h
        try:
            step = np.linalg.solve(J, -g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, -g, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            step = -np.sign(g[:len(free)]) * 0.5
        step = np.clip(step, -2.0, 2.0)                       # trust region [eV]
        lam, accepted = 1.0, None
        for _ in range(30):                                   # backtracking line search
            trial = np.clip(mu + lam * step, MU_OFF, MU_HI)
            gt, EFt, mvt, Nt, tt, okt = evaluate(trial)
            if float(np.max(np.abs(gt))) < nrm:
                accepted = (trial, gt, EFt, mvt, Nt, tt, okt)
                break
            lam *= 0.5
        if accepted is not None:
            mu, g, EF, mu_vec, N_cs, tot, bracketed = accepted
            nrm = float(np.max(np.abs(g)))
            method = "newton"
            continue
        if assign is None:
            method = "stalled (no Gauss-Seidel: rows/elements have no perfect matching)"
            break
        # ---- Gauss-Seidel fallback: one monotone-bisection sweep, one row per element ----
        mu_map = {e: float(m) for e, m in zip(free, mu)}
        for i, row in enumerate(rows):
            e = assign[i]
            _, _, _, tnow, _ = _state(ctx, mu_map, cols)
            if isinstance(row, AbsRow):
                tgt_e = float(row.target_cm3)
            elif e == row.num:
                tgt_e = float(row.r) * float(tnow[row.den])
            else:
                tgt_e = float(tnow[row.num]) / float(row.r)
            if not np.isfinite(tgt_e) or tgt_e <= 0:
                continue
            m_e, _lim = _bisect_mu(ctx, e, tgt_e, cols, mu_map, enforce_cap=enforce_cap)
            mu_map[e] = float(m_e)
        new_mu = np.array([mu_map[e] for e in free])
        g2, EF2, mv2, N2, t2, ok2 = evaluate(new_mu)
        nrm2 = float(np.max(np.abs(g2)))
        if nrm2 >= nrm and np.allclose(new_mu, mu, atol=1e-12):
            method = "stalled"
            break
        mu, g, EF, mu_vec, N_cs, tot = new_mu, g2, EF2, mv2, N2, t2
        bracketed, nrm, method = ok2, nrm2, "gauss-seidel"

    # ── site-blocking cap: the SAME Gauss-Seidel sweep engine.solve_codoped runs, against the
    #    IMPLIED targets (which is what a cap needs — a level, not a ratio).
    sol_lim = {e: False for e in free}
    if enforce_cap and any(ctx.occ_of(N_cs, cols[e]) > OCC_CAP for e in free):
        mu_map = {e: float(m) for e, m in zip(free, mu)}
        cap_tgt = _implied_targets(rows, free, {e: float(tot[e]) for e in mu_pinned})
        for _sweep in range(60):
            prev = dict(mu_map)
            for e in free:
                if e not in cap_tgt or not np.isfinite(cap_tgt[e]) or cap_tgt[e] <= 0:
                    mu_map[e] = float(min(mu_map[e], _mu_cap(ctx, e, cols, mu_map)))
                    continue
                m_e, _lim = _bisect_mu(ctx, e, cap_tgt[e], cols, mu_map, enforce_cap=True)
                mu_map[e] = float(m_e)
            if max(abs(mu_map[e] - prev[e]) for e in free) < 1e-10:
                break
        mu = np.array([mu_map[e] for e in free])
        g, EF, mu_vec, N_cs, tot, bracketed = evaluate(mu)
    # FINAL-STATE site-blocking flag, never inherited from an earlier round (engine's rule).
    # "Undershoots" is only well defined for a row in which the element appears with a POSITIVE
    # sign: an AbsRow, or a RatioRow's NUMERATOR. A capped DENOMINATOR makes the ratio residual
    # go the other way, so no such row is used to raise the flag — the state fails the gate
    # rather than being excused by an ambiguous sign.
    for e in free:
        at_cap = ctx.occ_of(N_cs, cols[e]) >= OCC_CAP * (1.0 - OCC_AT_CAP_RTOL)
        pos = [i for i, r in enumerate(rows)
               if (isinstance(r, AbsRow) and r.element == e) or
                  (isinstance(r, RatioRow) and r.num == e)]
        sol_lim[e] = bool(at_cap and pos and float(g[pos[0]]) < -TOL_DEX)

    resid, N_cs, n, p = ctx.Q(EF, mu_vec)
    _, _, sth1, sth2 = ctx.carriers(EF)
    N_defect = np.zeros(db.nD)
    np.add.at(N_defect, db.cs_ID - 1, N_cs)

    # ── validity: engine's own gate for the charge residual and the global occupancy bound
    #    (fixed_conc={} so it adds no target rows), PLUS the per-ROW residual check, which is
    #    the thing a ratio constraint actually has to satisfy.
    val = _validity(db, N_cs, n, p, resid, {}, {}, Nd_bg, Na_bg)
    row_fail = []
    for i, row in enumerate(rows):
        if abs(float(g[i])) <= TOL_DEX or not np.isfinite(g[i]):
            if not np.isfinite(g[i]):
                row_fail.append(f"row {i} ({row.label()}): residual is not finite")
            continue
        # An excuse is admitted only for a row whose FREE elements all sit at the dilute cap AND
        # in which each of them appears with a POSITIVE sign (AbsRow element, or RatioRow
        # numerator). A capped RatioRow DENOMINATOR is never an excuse — see the flag above.
        els = [e for e in row.elements() if e in free]
        signed_ok = all((isinstance(row, AbsRow) and row.element == e)
                        or (isinstance(row, RatioRow) and row.num == e) for e in els)
        excused = bool(els and signed_ok and all(sol_lim.get(e, False) for e in els)
                       and float(g[i]) < 0.0)
        if not excused:
            row_fail.append(f"row {i} ({row.label()}): residual {float(g[i]):+.4f} dex "
                            f"(|.| > {TOL_DEX:g})")
    val["validity_failures"] = list(val["validity_failures"]) + row_fail
    if not bracketed:
        val["validity_failures"] = list(val["validity_failures"]) + [
            f"E_F not bracketed in [{engine.EF_LO}, {engine.EF_HI}] eV (clamped)"]
    val["converged"] = not val["validity_failures"]

    mu_solved = {e: float(m) for e, m in zip(free, mu)}
    meta = {"pO2": pO2, "EcT_fraction": EcT_fraction, "fd": fd, "T_K": float(T),
            "solver": "ratio_codoping.solve_ratio(bisect-EF + damped-Newton-mu, engine prims)",
            "rows": [r.label() for r in rows],
            "implied_targets_cm3": {e: float(v) for e, v in implied.items()},
            "seed_mu_single_dopant": {e: float(v) for e, v in seed.items()},
            "seed_failed_cold_start": list(seed_failed),
            "gauss_seidel_available": assign is not None,
            "site_blocking_limited": bool(any(sol_lim.values())),
            "site_blocking_limited_per_element": {e: bool(v) for e, v in sol_lim.items()},
            "site_blocking_occ_cap": OCC_CAP,
            "EF_bracketed": bool(bracketed),
            "outer_iters": len(hist), "outer_hist": hist,
            "n_Q_evals": int(ctx.n_Q), "last_step_kind": method,
            "Nd_bg": float(Nd_bg), "Na_bg": float(Na_bg),
            "strict": bool(strict)}
    meta.update({k: v for k, v in val.items() if k != "validity_failures"})
    meta["validity_failures"] = val["validity_failures"]
    if strict:
        _raise_if_invalid(val, f"ratio_codoping.solve_ratio(T={T}, pO2={pO2}, "
                               f"rows={[r.label() for r in rows]})",
                          {"EF": float(EF), "mu_solved": mu_solved})

    eq = EqResult(T=float(T), EF=float(EF), n=float(n), p=float(p), sth1=float(sth1),
                  sth2=float(sth2), N_cs=N_cs, N_defect=N_defect, mu_vec=mu_vec,
                  Ev=ctx.Ev, Ec=ctx.Ec, Eg=ctx.Eg, charge_residual=float(resid), meta=meta)

    n_q = float("nan")
    if quench_T is not None:
        try:
            from .codoping import quench_hardened
            q = quench_hardened(db, eq, T_quench=float(quench_T), EcT_fraction=EcT_fraction,
                                fd=fd, Nd_bg=Nd_bg, Na_bg=Na_bg)
            n_q = float(q.n)
            meta["quench_bracket_failed"] = bool(q.meta.get("quench_bracket_failed", False))
            meta["quench_EF_clamped"] = bool(q.meta.get("quench_EF_clamped", False))
        except Exception as ex:                                   # pragma: no cover
            meta["quench_error"] = f"{type(ex).__name__}: {ex}"

    ach = {e: float(tot[e]) for e in all_elems}
    ratios = {}
    for row in rows:
        if isinstance(row, RatioRow):
            ratios[f"{row.num}/{row.den}"] = float(ach[row.num] / max(ach[row.den], 1e-300))

    return RatioResult(
        eq=eq, rows=rows, free_elements=list(free), pinned_mu_eV=dict(mu_pinned),
        mu_solved_eV=mu_solved, achieved_cm3=ach, achieved_ratios=ratios,
        residual_dex={r.label(): float(g[i]) for i, r in enumerate(rows)},
        converged=bool(val["converged"]), stage=stage,
        n_cm3=float(n), n_quench_cm3=n_q, p_cm3=float(p), EF_eV=float(EF),
        max_site_occupancy=float(val["max_site_occupancy_all_cs"]),
        site_blocking_limited_per_element={e: bool(v) for e, v in sol_lim.items()},
        anchor_mode=anchor_mode,
        reservoir_reference_systematic_eV=(SIGMA_ENERGETICS_MEASURED_EV
                                           if anchor_mode == "reservoir" else None),
        UNGATED=True, ungated_reason=census_gate_status()["ungated_reason"], meta=meta)


# ════════════════════════════════════════════════════════════════════════════════════════════
# convenience constructors
# ════════════════════════════════════════════════════════════════════════════════════════════
def ratio_rows_from_mixture(mixture: dict, anchor: str | None = None):
    """{'Si': 3.8e20, 'Mg': 1.9e20} -> [AbsRow(Si, 3.8e20), RatioRow(Mg, Si, 0.5)].

    The anchor element carries the absolute level; every other element is tied to it by a
    RATIO. This is the representation a fixed-ratio PROCESS has, as opposed to the two
    independent absolute targets a finished-film SIMS measurement has.
    """
    els = list(mixture)
    if not els:
        raise ValueError("empty mixture")
    a = anchor or els[0]
    if a not in mixture:
        raise ValueError(f"anchor {a!r} not in mixture {els}")
    rows = [AbsRow(a, float(mixture[a]))]
    for e in els:
        if e == a:
            continue
        rows.append(RatioRow(e, a, float(mixture[e]) / float(mixture[a])))
    return rows


def solve_ratio_from_mixture(db, T, mixture: dict, anchor: str | None = None, **kw):
    """Absolute-anchored fixed-ratio solve for a whole mixture. See `ratio_rows_from_mixture`."""
    return solve_ratio(db, T, rows=ratio_rows_from_mixture(mixture, anchor), **kw)


def solve_ratio_reservoir(db, T, pO2, anchor_element: str, ratios: dict,
                          compound: str | None = None, **kw):
    """RESERVOIR-anchored fixed-ratio solve.

    `anchor_element`'s mu is PINNED by a compound tie (compound_reservoir.mu_limit /
    mu_from_compound) — it is not an unknown — and every element in `ratios`
    ({element: r}, meaning [element]/[anchor] = r) gets a RatioRow. The ABSOLUTE level of the
    whole system is therefore set by the compound, which is the point of the mode.

    THE COST, carried on the result as `reservoir_reference_systematic_eV`: this mode inherits
    the FULL 1.300 eV Round-12e energetics systematic undamped. The absolute-anchored mode does
    not (the solved mu absorbs it). Do not read an absolute concentration out of this mode
    without that band.
    """
    from . import compound_reservoir as _cr
    if compound is None:
        lim = _cr.mu_limit(anchor_element, T, pO2)
        if not lim["supported"]:
            raise ValueError(f"compound reservoir not supported for {anchor_element}: "
                             f"{lim['reason']}")
        compound, mu_a = lim["limiting_compound"], lim["mu_limit_eV"]
    else:
        mu_a = _cr.mu_from_compound(compound, T, pO2)
    rows = [RatioRow(e, anchor_element, float(r)) for e, r in ratios.items()]
    res = solve_ratio(db, T, pO2=pO2, rows=rows, mu_pinned={anchor_element: float(mu_a)},
                      anchor_mode="reservoir", **kw)
    res.meta["reservoir_compound"] = compound
    res.meta["reservoir_mu_anchor_eV"] = float(mu_a)
    res.meta["reservoir_systematic_note"] = (
        f"reservoir mode inherits the full Round-12e {SIGMA_ENERGETICS_MEASURED_EV} eV "
        f"energetics systematic with NO self-consistent damping; the absolute-anchored mode "
        f"does not (measured 2.7e-15 dex/eV vs up to 3.3 dex/eV, Round-20 Task 1)")
    return res


# ════════════════════════════════════════════════════════════════════════════════════════════
# the physics that is MISSING regardless of how well this solves
# ════════════════════════════════════════════════════════════════════════════════════════════
def db_pair_census(db: DefectDB) -> dict:
    """Which two-dopant charge states does the DB actually carry, and which does a lab want?"""
    dop = [i for i, e in enumerate(db.elements) if e not in ("Ga", "O")]
    pairs: dict = {}
    for i in range(db.nCS):
        els = tuple(sorted(db.elements[c] for c in dop if abs(db.dm[i, c]) > 0))
        if len(els) >= 2:
            pairs[els] = pairs.get(els, 0) + 1
    have = {"+".join(k): int(v) for k, v in sorted(pairs.items())}
    wanted = {}
    for a, b in LAB_PAIRS_OF_INTEREST:
        key = "+".join(sorted((a, b)))
        wanted[key] = int(pairs.get(tuple(sorted((a, b))), 0))
    return {
        "two_dopant_charge_states_total": int(sum(pairs.values())),
        "pairs_present": have,
        "pairs_without_H": {k: v for k, v in have.items() if "H" not in k.split("+")},
        "lab_pairs_of_interest_charge_state_count": wanted,
        "lab_pairs_missing": [k for k, v in wanted.items() if v == 0],
    }


def pair_omission_warning() -> str:
    return (
        "MISSING PHYSICS — donor-acceptor pair binding.\n"
        "The KROGER DB carries 76 two-dopant charge states over 9 pairs, 7 of which contain H "
        "(only Fe-Ti and Fe-Sn do not). It carries ZERO charge states for Si-Mg, Sn-Mg, Zn-Mg "
        "or Si-Sn — the pairs a lab co-doping beta-Ga2O3 actually asks about. The solver "
        "therefore treats those species as MEAN-FIELD non-interacting, coupled only through "
        "E_F and the site budget.\n"
        "MEASURED COST (Round-20 re-verification of results/codoping/synthetic_pair.json):\n"
        "  at FIXED TOTAL CONTENT   : <= 0.181 dex on n at Eb = 1.0 eV  -> negligible\n"
        "  at FIXED CHEMICAL POTENT.: 109.08x (2.038 dex) on the INCORPORATED TOTAL at "
        "Eb = 1.0 eV -> NOT negligible\n"
        "RESERVOIR MODE IS FIXED CHEMICAL POTENTIAL. So the mode that exists in order to "
        "PREDICT the incorporated concentration is exactly the mode in which the missing pair "
        "binding costs up to 2 dex on that concentration. In a fixed-ratio reservoir solve the "
        "error also lands on the ACHIEVED RATIO whenever the two species are at unequal levels, "
        "because a 1:1 pair adds the same absolute density to both totals.")


# ════════════════════════════════════════════════════════════════════════════════════════════
# gate status / UNGATED marker
# ════════════════════════════════════════════════════════════════════════════════════════════
def census_gate_status() -> dict:
    """The PRE-REGISTERED verdict for this capability.
    See results/mathwork/PREREG_round20_ratio.md §4 and results/mathwork/round20_ratio.json."""
    return {
        "gate": "GATE-RATIO-1 (fixed-ratio co-incorporation confrontation)",
        "prereg": "results/mathwork/PREREG_round20_ratio.md",
        "prediction_blocks_found": 0,
        "power_floor_blocks": 5,
        "verdict": "UNTESTABLE-ON-THIS-CORPUS => SILENT => CAPABILITY IS UNGATED",
        "ungated_reason": (
            "Census of BOTH corpora (transport 549 rows, device 488 rows) found 32 co-doping "
            "candidate rows (18 transport, 14 device) and ZERO survivors of the pre-registered "
            "scope filter. Rejection counts, S1..S6: 12 / 22 / 5 / 6 / 31 / 2. The BINDING "
            "filter is S5 — 31 of 32 candidates carry no per-element concentration pair, so no "
            "RATIO can be formed at all. Only three rows have every named element inside the "
            "KROGER DB: transport rows 86/87 (Si+Ta, PLD, carrier 1.12e20 measured — rejected "
            "by S4, no DOI, and S5, no concentrations) and device row 163 (Mg+Zn, "
            "DOI:10.3390/nano15070499, 'ALD (Zn/Mg mixed cycle)' — rejected by S3, no measured "
            "quantity, and S5). 0 blocks against a pre-registered floor of 5."),
        "measurement_that_would_gate_it": (
            "One paper reporting, on the SAME beta-Ga2O3 film: (a) TWO dopant elements both "
            "present in the KROGER DB (Si, Sn, Mg, Zn, Ti, Fe, Ta, Zr, Hf, Ge, Cr, Ca, H), "
            "(b) BOTH their concentrations - SIMS, not nominal target composition - so a RATIO "
            "exists, (c) the growth/anneal temperature AND the oxygen partial pressure, and "
            "(d) a measured carrier density on that same film. Five such papers, or five "
            "independently-conditioned films within one paper treated as separate blocks, "
            "clears the power floor. The highest-information version is a RATIO SERIES at "
            "fixed total content and fixed (T, pO2) - e.g. Si:Mg at 9:1, 3:1, 1:1, 1:3, 1:9 - "
            "because the compensation ratio is what the constraint predicts, and a series "
            "cancels the per-lab offset that the between-paper ICC (58-85%) otherwise imposes."),
    }


def ungated_marker() -> str:
    s = census_gate_status()
    return ("*** UNGATED CAPABILITY ***\n"
            f"{s['verdict']}\n{s['ungated_reason']}\n"
            f"WOULD BE GATED BY: {s['measurement_that_would_gate_it']}\n"
            "PLUS two limits that no measurement of the ratio itself would remove:\n"
            f"  (1) RESERVOIR-anchored ratios inherit the full "
            f"{SIGMA_ENERGETICS_MEASURED_EV} eV Round-12e energetics systematic, undamped.\n"
            "  (2) the DB has no donor-acceptor pair binding for Si-Mg / Sn-Mg / Zn-Mg / Si-Sn; "
            "cost <= 0.181 dex at fixed total content, up to 109x at fixed chemical potential "
            "(= reservoir mode).")
