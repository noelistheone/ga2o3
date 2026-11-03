"""General charge-neutrality defect-equilibrium engine for β-Ga2O3 (KROGER-class, clean-room).

Reproduces KROGER's dark equilibrium + full-quench physics on the published energetics
(src/sandbox/kroger_db.py) with our own thermochemistry (src/sandbox/thermo.py):

  dG(cs, E_F) = dHo + q·E_F − 3 k_B T s_vib(T) Σdm − dm·μ_vec
  N_cs        = prefactor · exp(−dG/k_B T)
  Q(E_F)      = Σ q·N_cs + p + sth1 + sth2 − n + Nd − Na = 0   (solve for E_F, VBM=0)

μ_vec: host Ga/O from thermo.mu_Ga_O(T,pO2); dopants OFF at −30 eV unless given a fixed μ or a
fixed total concentration (μ solved jointly). Carriers Boltzmann (FD optional). Quench: freeze
per-defect totals from T_anneal, re-solve E_F at T_quench redistributing charge states.

This is a numpy reference implementation prioritising correctness for Gate 1; a differentiable
torch twin (IFT gradients) can wrap the same equations later.

── Solver structure (post-2026-07 correctness fix) ──────────────────────────────────────────
  len(fixed_conc) == 0 : one E_F root find.
  len(fixed_conc) == 1 : monotone bisection on μ_X ([X] is strictly increasing in μ_X), with an
                         upper bound at the SITE-BLOCKING limit (see below) located BY BISECTION.
  len(fixed_conc) >= 2 : `solve_codoped` — inner monotone bisection on E_F over a fixed wide
                         window + outer damped Newton on log10-concentration residuals with a
                         Gauss-Seidel (per-element bisection) fallback.  This replaces the old
                         joint `scipy.optimize.fsolve` branch, which could return `ier=5` with
                         x == x0 (no move at all) or `ier=1` ("success") with ||F||~1e14 and
                         still be returned SILENTLY as a physical answer.
  Every branch ends in `_validity(...)`: normalised charge residual, achieved-vs-target dex for
  every element that is not site-blocking-limited, and max site occupancy <= 1.  The verdict is
  meta["converged"]; with strict=True (default) a failure RAISES DefectEquilibriumError instead
  of returning a silently wrong EqResult.

── "site blocking", not "solubility" ────────────────────────────────────────────────────────
  The μ_X ceiling enforced here is NOT a thermodynamic solubility / secondary-phase boundary
  (that would need competing-phase formation energies, which this DB does not carry).  It is the
  point where a dopant-bearing charge state reaches OCC_CAP = 10% of its own site prefactor,
  i.e. where the DILUTE (non-interacting, ideal-site-occupancy) approximation behind
  N = N_site·exp(−dG/kT) stops being defensible.  It is reported as `site_blocking_limited`
  (+ a per-element dict); `solubility_limited` is kept as a DEPRECATED ALIAS for API
  compatibility with full_forward / codoping / the probe scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from . import thermo
from .kroger_db import DefectDB, N_SITE

MU_OFF = -30.0  # eV, KROGER's "element absent" default
MU_HI = 5.0     # eV, upper box bound on any dopant μ
OCC_CAP = 0.1   # dilute-approximation / site-blocking ceiling (fraction of the site prefactor)
EF_LO, EF_HI = -40.0, 45.0   # eV, widest E_F window ever scanned (VBM = 0)

# validity tolerances.  Both are set from NUMERICS, not from any gate outcome:
#   TOL_RESID: brentq runs at xtol=1e-10 eV and dQ/dE_F ~ Q/kT with kT >= 0.026 eV, so a
#              converged root leaves |Q|/scale ~ 4e-9; 1e-6 is ~250x looser than that.
#   TOL_DEX  : the μ bisections converge to ~1e-11 eV (=> ~1e-10 dex); 1e-3 dex = 0.23% in
#              concentration, i.e. far below any physically meaningful level.
TOL_RESID = 1e-6
TOL_DEX = 1e-3

# How close to OCC_CAP an element must sit before "site-blocking-limited" is accepted as an
# excuse for missing its target.  This is a NUMERICS guard-band, not a physical parameter, and
# it is chosen from the two error sources it has to straddle:
#   floor  - μ is located to ~1e-10 eV and occ ~ exp(μ/kT) with kT >= 0.026 eV, so the μ search
#            alone leaves ~4e-9 relative slack in occ; the multi-element Gauss-Seidel fixed
#            point leaves up to ~1e-3 relative (measured: Si+Mg 20/20 at% lands at
#            occ/OCC_CAP = 0.99892).
#   ceiling- the defect it must catch is an element sitting far below its cap while claiming to
#            be capped (measured bug: occ/OCC_CAP = 0.718).
# 1e-2 is ~10x above the numerical floor and ~28x below the defect. No gate depends on it.
OCC_AT_CAP_RTOL = 1e-2


class DefectEquilibriumError(RuntimeError):
    """Raised when a defect-equilibrium solve did not produce a physically valid state.

    Carries the full validity report in `.report` (normalised charge residual, per-element
    achieved-vs-target dex, max site occupancy, and the list of failed conditions)."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = dict(report or {})


@dataclass
class EqResult:
    T: float
    EF: float                 # Fermi level above VBM(T) [eV]
    n: float
    p: float
    sth1: float
    sth2: float
    N_cs: np.ndarray          # per-charge-state concentration [cm^-3]
    N_defect: np.ndarray      # per-defect total [cm^-3]
    mu_vec: np.ndarray
    Ev: float
    Ec: float
    Eg: float
    charge_residual: float
    Nd: float = 0.0
    Na: float = 0.0
    meta: dict = field(default_factory=dict)


def _mu_vector(db: DefectDB, T: float, pO2: float, mu_override: dict) -> np.ndarray:
    mu = np.full(len(db.elements), MU_OFF, dtype=np.float64)
    mu_Ga, mu_O = thermo.mu_Ga_O(T, pO2)
    mu[0] = float(mu_Ga)
    mu[1] = float(mu_O)
    for k, v in (mu_override or {}).items():
        col = db.col(k) if isinstance(k, str) else k
        mu[col] = float(v)
    return mu


def _carriers(EF, kT, Ec, Ev, Nc, Nv, sth_flag=1.0, fd=False):
    if fd:
        n = _fd(( EF - Ec) / kT, Nc)
        p = _fd((-(EF - Ev)) / kT, Nv)
        sth1 = sth_flag * _fd(-((EF - Ev - thermo.E_RELAX_STH1) / kT), N_SITE)
        sth2 = sth_flag * _fd(-((EF - Ev - thermo.E_RELAX_STH2) / kT), N_SITE)
    else:
        n = Nc * np.exp((EF - Ec) / kT)
        p = Nv * np.exp(-(EF - Ev) / kT)
        sth1 = sth_flag * N_SITE * np.exp(-(EF - Ev - thermo.E_RELAX_STH1) / kT)
        sth2 = sth_flag * N_SITE * np.exp(-(EF - Ev - thermo.E_RELAX_STH2) / kT)
    return n, p, sth1, sth2


def _fd(eta, Nc3d):
    """n = 2/sqrt(pi) Nc F_{1/2}(eta); Bednarczyk analytic approx to the FD-1/2 integral."""
    eta = np.asarray(eta, dtype=np.float64)
    # Bednarczyk & Bednarczyk (1978) approximation to F_{1/2}
    nu = eta**4 + 50.0 + 33.6 * eta * (1.0 - 0.68 * np.exp(-0.17 * (eta + 1.0) ** 2))
    xi = 3.0 * np.sqrt(np.pi) / (4.0 * nu**0.375)
    F = 1.0 / (np.exp(-eta) + xi)
    return Nc3d * F  # already the 2/sqrt(pi)-normalised carrier density form


def _A_terms(db, T, mu_vec, sth_unused=None):
    """EF-independent part A(cs) = dHo − 3 kBT s_vib Σdm − dm·μ; dG = A + q·EF."""
    kT = thermo.KB_EV * T
    s = float(thermo.dSvib_norm(T))
    dG_vib = 3.0 * kT * s * db.sum_dm
    A = db.dHo - dG_vib - db.dm @ mu_vec
    return A, kT


# ── numerical primitives ─────────────────────────────────────────────────────
def _site_blocking_cap(occ_fn, lo: float = MU_OFF, hi: float = MU_HI,
                       occ_cap: float = OCC_CAP, iters: int = 100,
                       tol: float = 1e-12) -> float:
    """Largest μ in [lo, hi] whose max dopant site occupancy is <= occ_cap.

    occ(μ) is monotone increasing in μ, so this is a plain bisection and is EXACT to `tol`.
    Returns `hi` unchanged when the cap is not binding (so an uncapped solve follows exactly the
    same code path as before this fix), and `lo` when even μ=lo already exceeds the cap.
    The old code walked `hi -= 0.5` until occ <= cap and never refined, so it stopped up to one
    0.5 eV grid step BELOW the true cap and then falsely reported the target as unreachable."""
    if occ_fn(hi) <= occ_cap:
        return hi
    if occ_fn(lo) > occ_cap:
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if occ_fn(mid) <= occ_cap:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo                      # lo always satisfies occ <= occ_cap


def _bracket_scan(f, lo: float, hi: float, wide_lo: float = EF_LO, wide_hi: float = EF_HI,
                  n_scan: int = 171):
    """Bracket the (unique) root of the monotone-decreasing f without ever raising.

    Tries [lo, hi] first (so the normal case is bit-identical to the pre-fix brentq call), then
    scans the widest physical window on a fixed grid.  Returns (a, b, fa, fb) with f(a)*f(b) < 0,
    or (x, x, 0.0, 0.0) if a grid point is an exact root, or None if no sign change exists."""
    def opposite(a, b):
        # sign comparison, NOT a*b < 0: Q can reach ~1e300 and the product overflows to inf
        return (np.isfinite(a) and np.isfinite(b)
                and ((a > 0 and b < 0) or (a < 0 and b > 0)))

    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo, lo, 0.0, 0.0
    if fhi == 0.0:
        return hi, hi, 0.0, 0.0
    if opposite(flo, fhi):
        return lo, hi, flo, fhi
    grid = np.linspace(wide_lo, wide_hi, n_scan)
    vals = np.array([f(float(x)) for x in grid], dtype=float)
    for i in range(n_scan):
        if vals[i] == 0.0:
            return float(grid[i]), float(grid[i]), 0.0, 0.0
    for i in range(n_scan - 1):
        if opposite(vals[i], vals[i + 1]):
            return float(grid[i]), float(grid[i + 1]), float(vals[i]), float(vals[i + 1])
    return None


# ── post-solve validity ──────────────────────────────────────────────────────
def _validity(db: DefectDB, N_cs, n, p, resid, fixed_conc: dict, limited: dict,
              Nd_bg: float = 0.0, Na_bg: float = 0.0,
              tol_resid: float = TOL_RESID, tol_dex: float = TOL_DEX,
              occ_cap: float = OCC_CAP) -> dict:
    """Physical-validity verdict for a solved state (applied to EVERY branch).

      1. |Q| normalised by the largest charge density in play  <  tol_resid
      2. |log10(achieved/target)| < tol_dex for every fixed element, UNLESS that element is
         genuinely site-blocking-limited IN THE RETURNED STATE
      3. max site occupancy over ALL charge states <= 1 (dilute-model hard bound)

    Condition 2's excuse is verified against the STATE, not against a caller-supplied boolean.
    A missed target is excused only when all three hold:
        the caller flagged the element limited, AND
        its own max site occupancy is AT the dilute cap (>= occ_cap*(1-1e-9)), AND
        it UNDERSHOOTS (dex < 0 — a site-blocking ceiling can only remove dopant, never add it).
    BUG FIXED 2026-07-28: previously the boolean alone was enough, so a flag raised in an
    earlier cap round of solve_codoped (where it was true) excused an arbitrarily large miss in
    the final state (where the element had drifted back BELOW the cap and more dopant was
    demonstrably reachable).  Measured case: Si+Sn 50/50 at 1350 K returned converged=True with
    Si 1.05 dex short at occupancy 0.0718 < OCC_CAP = 0.1.
    """
    scale = max(abs(float(n)), abs(float(p)), abs(float(Nd_bg)), abs(float(Na_bg)),
                float(np.abs(db.charge * N_cs).sum()), 1.0)
    rn = abs(float(resid)) / scale
    achieved, dex, occ = {}, {}, {}
    for e, tgt in (fixed_conc or {}).items():
        col = db.col(e)
        tot = float((db.dm[:, col] * N_cs).sum())
        has = db.dm[:, col] > 0
        achieved[e] = tot
        dex[e] = float(np.log10(max(tot, 1e-30)) - np.log10(max(float(tgt), 1e-30)))
        occ[e] = float((N_cs[has] / db.prefactor[has]).max()) if has.any() else 0.0
    occ_all = float((N_cs / db.prefactor).max())
    fails = []
    if not np.isfinite(rn) or rn > tol_resid:
        fails.append(f"charge neutrality: |Q|/scale = {rn:.3e} > {tol_resid:g}")
    excused = {}
    for e, v in dex.items():
        at_cap = bool(occ.get(e, 0.0) >= occ_cap * (1.0 - OCC_AT_CAP_RTOL))
        ex = bool(limited.get(e, False)) and at_cap and np.isfinite(v) and v < 0.0
        excused[e] = ex
        if not ex and (not np.isfinite(v) or abs(v) > tol_dex):
            why = (f"flagged limited but occ = {occ.get(e, 0.0):.4g} < cap {occ_cap:g}"
                   if bool(limited.get(e, False)) and not at_cap else
                   (f"flagged limited but OVERSHOOTS the target" if bool(limited.get(e, False))
                    else f"{e} is not site-blocking-limited"))
            fails.append(f"[{e}] achieved/target = {v:+.4f} dex (|.| > {tol_dex:g}) and {why}")
    if not np.isfinite(occ_all) or occ_all > 1.0:
        fails.append(f"max site occupancy = {occ_all:.3e} > 1 (dilute model violated)")
    return {"converged": not fails,
            "charge_residual_normalised": rn,
            "achieved_cm3": achieved,
            "target_hit_dex": dex,
            "site_occupancy": occ,
            "target_miss_excused_by_site_blocking": excused,
            "max_site_occupancy_all_cs": occ_all,
            "validity_failures": fails,
            "tolerances": {"charge_residual_normalised": tol_resid, "target_dex": tol_dex,
                           "site_blocking_occ_cap": occ_cap}}


def _raise_if_invalid(val: dict, where: str, extra: dict | None = None):
    if val["converged"]:
        return
    msg = (f"{where}: solve did NOT converge to a physically valid state -> "
           + "; ".join(val["validity_failures"]))
    rep = dict(val)
    rep.update(extra or {})
    raise DefectEquilibriumError(msg, rep)


def solve_single(db: DefectDB, T: float, pO2: float = 0.2, EcT_fraction: float = 0.40,
                 mu_override: dict | None = None, fixed_conc: dict | None = None,
                 fd: bool = False, sth_flag: float = 1.0,
                 Nd_bg: float = 0.0, Na_bg: float = 0.0,
                 strict: bool = True) -> EqResult:
    """Solve dark charge-neutrality at one temperature. fixed_conc: {elem: total_cm3} solves μ.
    Nd_bg/Na_bg: fixed shallow background donor/acceptor density [cm^-3] (unintentional Si/H/Fe
    contamination — always-on in real films; UID Ga2O3 is n-type from ~1e17 Si/H).

    strict (default True): raise DefectEquilibriumError if the post-solve validity check fails.
    Pass strict=False to get the invalid EqResult back anyway — meta['converged'] is False and
    meta['validity_failures'] lists what went wrong. Never treat strict=False output as physics.
    """
    Eg, Ev, Ec = (float(x) for x in thermo.band_edges(T, EcT_fraction))
    Nc, Nv = float(thermo.Nc(T)), float(thermo.Nv(T))
    mu_override = dict(mu_override or {})
    fixed_conc = dict(fixed_conc or {})

    if len(fixed_conc) >= 2:
        # ONE implementation of the multi-element solve (see solve_codoped below).
        return solve_codoped(db, T, pO2=pO2, EcT_fraction=EcT_fraction,
                             mu_override=mu_override, fixed_conc=fixed_conc, fd=fd,
                             sth_flag=sth_flag, Nd_bg=Nd_bg, Na_bg=Na_bg, strict=strict)

    def defect_charge_and_conc(EF, mu_vec):
        A, kT = _A_terms(db, T, mu_vec)
        # clip exponent to avoid overflow; huge-positive dG -> ~0 concentration
        expo = np.clip(-(A + db.charge * EF) / kT, -700, 300)
        N_cs = db.prefactor * np.exp(expo)
        return N_cs

    def Q(EF, mu_vec):
        N_cs = defect_charge_and_conc(EF, mu_vec)
        n, p, sth1, sth2 = _carriers(EF, thermo.KB_EV * T, Ec, Ev, Nc, Nv, sth_flag, fd)
        return float((db.charge * N_cs).sum() + p + sth1 + sth2 - n + Nd_bg - Na_bg), N_cs

    def solve_EF(mu_vec):
        f = lambda ef: Q(ef, mu_vec)[0]
        br = _bracket_scan(f, Ev - 3.0, Ec + 3.0)
        if br is None:
            raise DefectEquilibriumError(
                f"solve_single(T={T}, pO2={pO2}): no E_F root in [{EF_LO}, {EF_HI}] eV — "
                f"Q({EF_LO})={f(EF_LO):.3e}, Q({EF_HI})={f(EF_HI):.3e}",
                {"T": T, "pO2": pO2, "mu_vec": [float(v) for v in mu_vec]})
        a, b, _, _ = br
        if a == b:
            return a
        return brentq(f, a, b, xtol=1e-10, rtol=1e-12, maxiter=200)

    site_blocking: dict = {}
    mu_box: dict = {}
    if not fixed_conc:
        mu_vec = _mu_vector(db, T, pO2, mu_override)
        EF = solve_EF(mu_vec)
    else:
        # robust 1-D bisection on the dopant μ: total [X] is strictly increasing in μ_X.
        # Upper bound = the SITE-BLOCKING / dilute-approximation limit (NOT a thermodynamic
        # phase boundary): the μ_X at which any X-containing charge state reaches OCC_CAP of
        # its site prefactor.  Located BY BISECTION (_site_blocking_cap), not by the old
        # `hi -= 0.5` descent, which truncated the reachable μ range by up to 0.5 eV and so
        # falsely reported reachable targets as limited (Sn: −0.535 dex at occ=9.1e-4, i.e.
        # 110x below the 0.1 cap it claimed to be enforcing).
        (elem, target), = fixed_conc.items()
        col = db.col(elem)
        has_x = db.dm[:, col] > 0

        def state_at(mu_x):
            mu_vec = _mu_vector(db, T, pO2, {**mu_override, elem: mu_x})
            EF = solve_EF(mu_vec)
            N_cs = defect_charge_and_conc(EF, mu_vec)
            tot = float((db.dm[:, col] * N_cs).sum())
            occ = float((N_cs[has_x] / db.prefactor[has_x]).max()) if has_x.any() else 0.0
            return EF, mu_vec, N_cs, tot, occ

        lo = MU_OFF
        hi = _site_blocking_cap(lambda m: state_at(m)[4], lo=MU_OFF, hi=MU_HI)
        # Is the DILUTE CAP what stopped the search, or the arbitrary μ box bound MU_HI?
        # Only the former is a physical statement ("this much cannot dissolve"); the latter is a
        # numerical box artefact and must NOT be dressed up as site blocking — _validity checks
        # the achieved occupancy against OCC_CAP and fails the solve if the excuse is not real.
        cap_is_binding = bool(hi < MU_HI - 1e-12)
        limited = False
        mu_box_limited = False
        if hi <= lo:
            # even the "element absent" μ already exceeds the site-blocking cap
            limited, mu_x = True, hi
        else:
            _, _, _, tot_hi, _ = state_at(hi)
            if tot_hi < target:
                limited = True
                mu_box_limited = not cap_is_binding
                mu_x = hi
            else:
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    _, _, _, tot_mid, _ = state_at(mid)
                    if tot_mid < target:
                        lo = mid
                    else:
                        hi = mid
                mu_x = 0.5 * (lo + hi)
        site_blocking[elem] = bool(limited)
        mu_box[elem] = bool(mu_box_limited)
        EF, mu_vec, _, _, _ = state_at(mu_x)
        mu_override[elem] = float(mu_x)

    resid, N_cs = Q(EF, mu_vec)
    n, p, sth1, sth2 = _carriers(EF, thermo.KB_EV * T, Ec, Ev, Nc, Nv, sth_flag, fd)
    val = _validity(db, N_cs, n, p, resid, fixed_conc, site_blocking, Nd_bg, Na_bg)
    # per-defect totals
    N_defect = np.zeros(db.nD)
    np.add.at(N_defect, db.cs_ID - 1, N_cs)
    meta = {"pO2": pO2, "EcT_fraction": EcT_fraction, "fd": fd,
            "fixed_conc": fixed_conc, "mu_override": mu_override,
            "solver": f"engine.solve_single(len={len(fixed_conc)})",
            "site_blocking_limited": bool(any(site_blocking.values())),
            "site_blocking_limited_per_element": dict(site_blocking),
            # target unreachable at the μ BOX bound (MU_HI) rather than at the dilute cap —
            # a numerical bound, not physics; _validity fails such a state.
            "mu_box_limited_per_element": dict(mu_box),
            # DEPRECATED aliases (this is a dilute/site-blocking limit, not a solubility limit)
            "solubility_limited": bool(any(site_blocking.values())),
            "solubility_limited_per_element": dict(site_blocking),
            "site_blocking_occ_cap": OCC_CAP,
            "strict": bool(strict)}
    meta.update(val)
    if strict:
        _raise_if_invalid(val, f"engine.solve_single(T={T}, pO2={pO2}, fixed_conc={fixed_conc})",
                          {"EF": float(EF), "mu_override": dict(mu_override)})
    return EqResult(T=T, EF=EF, n=float(n), p=float(p), sth1=float(sth1), sth2=float(sth2),
                    N_cs=N_cs, N_defect=N_defect, mu_vec=mu_vec, Ev=Ev, Ec=Ec, Eg=Eg,
                    charge_residual=float(resid), meta=meta)


# ════════════════════════════════════════════════════════════════════════════
# multi-element (co-doped) fixed-concentration solve — THE single implementation
# (previously duplicated in codoping.py; codoping.py now imports it from here.  The import
#  direction engine -> codoping would have been circular because codoping also pulls in the
#  forward stack (full_forward/transport/optical/device); keeping the numerics in engine.py
#  leaves engine dependency-free and lets the high-level module import downward.)
# ════════════════════════════════════════════════════════════════════════════
class _Ctx:
    """Thin re-exposure of Q(E_F, μ) at one (T, pO2, EcT_fraction, fd) point."""

    def __init__(self, db, T, pO2, EcT_fraction, fd, sth_flag, Nd_bg, Na_bg, mu_base):
        self.db, self.T, self.pO2 = db, float(T), float(pO2)
        self.fd, self.sth_flag = bool(fd), float(sth_flag)
        self.Nd_bg, self.Na_bg = float(Nd_bg), float(Na_bg)
        self.mu_base = dict(mu_base or {})
        self.Eg, self.Ev, self.Ec = (float(x) for x in thermo.band_edges(T, EcT_fraction))
        self.Nc, self.Nv = float(thermo.Nc(T)), float(thermo.Nv(T))
        self.kT = thermo.KB_EV * float(T)
        self.n_Q = 0

    def mu_vector(self, mu_map):
        return _mu_vector(self.db, self.T, self.pO2, {**self.mu_base, **mu_map})

    def N_cs(self, EF, mu_vec):
        A, kT = _A_terms(self.db, self.T, mu_vec)
        expo = np.clip(-(A + self.db.charge * EF) / kT, -700, 300)
        return self.db.prefactor * np.exp(expo)

    def carriers(self, EF):
        return _carriers(EF, self.kT, self.Ec, self.Ev, self.Nc, self.Nv, self.sth_flag, self.fd)

    def Q(self, EF, mu_vec):
        self.n_Q += 1
        N_cs = self.N_cs(EF, mu_vec)
        n, p, s1, s2 = self.carriers(EF)
        q = float((self.db.charge * N_cs).sum() + p + s1 + s2 - n + self.Nd_bg - self.Na_bg)
        return q, N_cs, float(n), float(p)

    def solve_EF(self, mu_vec):
        """Monotone bisection. NEVER raises: clamps to the window end and flags instead
        (the caller records EF_bracketed=False, which fails the validity gate)."""
        lo, hi = EF_LO, EF_HI
        flo = self.Q(lo, mu_vec)[0]
        fhi = self.Q(hi, mu_vec)[0]
        if not (flo > 0):          # covers <=0 and NaN
            return lo, False
        if not (fhi < 0):
            return hi, False
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.Q(mid, mu_vec)[0] > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        return 0.5 * (lo + hi), True

    # -- per-element diagnostics -------------------------------------------
    def total_of(self, N_cs, col):
        return float((self.db.dm[:, col] * N_cs).sum())

    def occ_of(self, N_cs, col):
        has = self.db.dm[:, col] > 0
        if not has.any():
            return 0.0
        return float((N_cs[has] / self.db.prefactor[has]).max())


def _state(ctx, mu_map, cols):
    mu_vec = ctx.mu_vector(mu_map)
    EF, bracketed = ctx.solve_EF(mu_vec)
    N_cs = ctx.N_cs(EF, mu_vec)
    tot = {e: ctx.total_of(N_cs, c) for e, c in cols.items()}
    return EF, mu_vec, N_cs, tot, bracketed


def _g_vec(ctx, mu_arr, elems, cols, targets):
    mu_map = {e: float(m) for e, m in zip(elems, mu_arr)}
    EF, mu_vec, N_cs, tot, ok = _state(ctx, mu_map, cols)
    g = np.array([np.log10(max(tot[e], 1e-30)) - np.log10(max(targets[e], 1e-30))
                  for e in elems])
    return g, EF, mu_vec, N_cs, tot, ok


def _mu_cap(ctx, elem, cols, mu_map, occ_cap=OCC_CAP):
    """Largest μ_elem with max site occupancy of elem-containing states <= occ_cap.
    Same bisection primitive the len==1 branch uses (_site_blocking_cap)."""
    col = cols[elem]

    def occ(mu_x):
        _, _, N_cs, _, _ = _state(ctx, {**mu_map, elem: float(mu_x)}, cols)
        return ctx.occ_of(N_cs, col)

    return _site_blocking_cap(occ, lo=MU_OFF, hi=MU_HI, occ_cap=occ_cap, iters=80, tol=1e-10)


def _bisect_mu(ctx, elem, target, cols, mu_map, enforce_cap=True):
    """Monotone bisection on μ_elem for total [elem] == target, other μ held fixed.
    Returns (μ, site_blocking_limited)."""
    col = cols[elem]
    hi = _mu_cap(ctx, elem, cols, mu_map) if enforce_cap else MU_HI

    def tot(mu_x):
        _, _, N_cs, _, _ = _state(ctx, {**mu_map, elem: float(mu_x)}, cols)
        return ctx.total_of(N_cs, col)

    if tot(hi) < target:
        return hi, True
    lo = MU_OFF
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if tot(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-11:
            break
    return 0.5 * (lo + hi), False


def _single_dopant_mu(db, elem, target, T, pO2, EcT_fraction, fd, sth_flag, Nd_bg, Na_bg,
                      mu_base):
    """Continuation seed: μ_elem from the len==1 bisection branch, elem alone.

    Returns (μ_seed, site_blocking_limited, seed_ok).  seed_ok=False marks the -6.0 COLD START
    fallback so a caller can never read the sentinel as a solved μ (a legitimately solved μ of
    exactly -6.0 eV was previously indistinguishable from the failure value).  Only
    DefectEquilibriumError is caught — any other exception is a bug and must propagate.
    """
    try:
        r = solve_single(db, T, pO2=pO2, EcT_fraction=EcT_fraction,
                         mu_override=dict(mu_base or {}), fixed_conc={elem: target},
                         fd=fd, sth_flag=sth_flag, Nd_bg=Nd_bg, Na_bg=Na_bg, strict=False)
        return (float(r.meta["mu_override"][elem]),
                bool(r.meta.get("site_blocking_limited", False)), True)
    except DefectEquilibriumError:
        return -6.0, False, False


def solve_codoped(db: DefectDB, T: float, pO2: float = 0.2, EcT_fraction: float = 0.40,
                  mu_override: dict | None = None, fixed_conc: dict | None = None,
                  fd: bool = False, sth_flag: float = 1.0,
                  Nd_bg: float = 0.0, Na_bg: float = 0.0,
                  tol_dex: float = 1e-6, itmax: int = 60,
                  enforce_cap: bool = True, strict: bool = True) -> EqResult:
    """Charge-neutrality solve with an arbitrary number of fixed total dopant concentrations.

      0 or 1 entries in fixed_conc -> delegates to solve_single (bit-identical).
      >= 2 entries                 -> nested solve:
          inner  E_F by MONOTONE BISECTION on the fixed wide window [EF_LO, EF_HI]; Q(E_F) is
                 strictly decreasing so bisection cannot fail, and an unbracketed window clamps
                 + flags instead of raising.
          outer  damped Newton on g_i(μ) = log10([X_i](μ)) − log10(target_i) with a
                 forward-difference Jacobian, ±2 eV trust region and backtracking line search;
                 a per-element monotone-bisection Gauss-Seidel sweep is the fallback whenever the
                 Newton step fails to reduce max|g|.
          start  CONTINUATION from each element's converged single-dopant μ.
          gate   explicit post-solve validity check (_validity).  fsolve's `ier` is NOT a
                 residual test: the legacy branch returned ier=5 with x==x0 for Si+Mg, and ier=1
                 ("success") with ||F||=1.46e14 for Fe+Ti from x0=[1.0,−6,−6].
    """
    fixed_conc = dict(fixed_conc or {})
    mu_override = dict(mu_override or {})
    if len(fixed_conc) <= 1:
        return solve_single(db, T, pO2=pO2, EcT_fraction=EcT_fraction,
                            mu_override=mu_override, fixed_conc=fixed_conc,
                            fd=fd, sth_flag=sth_flag, Nd_bg=Nd_bg, Na_bg=Na_bg, strict=strict)

    ctx = _Ctx(db, T, pO2, EcT_fraction, fd, sth_flag, Nd_bg, Na_bg, mu_override)
    elems = list(fixed_conc.keys())
    cols = {e: db.col(e) for e in elems}
    targets = {e: float(v) for e, v in fixed_conc.items()}

    # ---- continuation start: each element solved ALONE by the len==1 branch ----
    seed, seed_lim, seed_failed = {}, {}, []
    for e in elems:
        m, lim, ok = _single_dopant_mu(db, e, targets[e], T, pO2, EcT_fraction, fd, sth_flag,
                                       Nd_bg, Na_bg, mu_override)
        seed[e], seed_lim[e] = m, lim
        if not ok:
            seed_failed.append(e)
    mu = np.array([seed[e] for e in elems], dtype=float)

    hist, method = [], "newton"
    g, EF, mu_vec, N_cs, tot, bracketed = _g_vec(ctx, mu, elems, cols, targets)
    nrm = float(np.max(np.abs(g)))
    for it in range(itmax):
        hist.append({"it": it, "mu": [float(v) for v in mu], "EF": float(EF),
                     "max_abs_dex": nrm, "step": method})
        if nrm < tol_dex:
            break
        # forward-difference Jacobian dg_i/dmu_j
        h = 1e-3
        J = np.zeros((len(elems), len(elems)))
        for j in range(len(elems)):
            mp = mu.copy(); mp[j] += h
            gp, *_ = _g_vec(ctx, mp, elems, cols, targets)
            J[:, j] = (gp - g) / h
        try:
            step = np.linalg.solve(J, -g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, -g, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            step = -np.sign(g) * 0.5
        step = np.clip(step, -2.0, 2.0)                       # trust region [eV]
        lam, accepted = 1.0, None
        for _ in range(30):                                   # backtracking line search
            trial = np.clip(mu + lam * step, MU_OFF, MU_HI)
            gt, EFt, mvt, Nt, tt, okt = _g_vec(ctx, trial, elems, cols, targets)
            if float(np.max(np.abs(gt))) < nrm:
                accepted = (trial, gt, EFt, mvt, Nt, tt, okt)
                break
            lam *= 0.5
        if accepted is not None:
            mu, g, EF, mu_vec, N_cs, tot, bracketed = accepted
            nrm = float(np.max(np.abs(g)))
            method = "newton"
        else:
            # ---- Gauss-Seidel fallback: one monotone bisection sweep over mu_i ----
            mu_map = {e: float(m) for e, m in zip(elems, mu)}
            for e in elems:
                m_e, _lim = _bisect_mu(ctx, e, targets[e], cols, mu_map,
                                       enforce_cap=enforce_cap)
                mu_map[e] = float(m_e)
            new_mu = np.array([mu_map[e] for e in elems])
            g2, EF2, mv2, N2, t2, ok2 = _g_vec(ctx, new_mu, elems, cols, targets)
            nrm2 = float(np.max(np.abs(g2)))
            if nrm2 >= nrm and np.allclose(new_mu, mu, atol=1e-12):
                method = "stalled"
                break
            mu, g, EF, mu_vec, N_cs, tot = new_mu, g2, EF2, mv2, N2, t2
            bracketed, nrm, method = ok2, nrm2, "gauss-seidel"

    # ---- site-blocking cap: Gauss-Seidel sweep over ALL elements, nothing is ever retired ----
    # `_bisect_mu(..., enforce_cap=True)` first locates the element's own dilute cap against the
    # CURRENT mu of the others and then bisects below it, so it can never place an element above
    # its cap and never needs the element to be frozen.  Re-solving every element in every sweep
    # is what makes the final state self-consistent.
    #
    # BUG FIXED 2026-07-28: the old loop capped an element, marked it `limited`, and then
    # excluded it from every later sweep (`... and not sol_lim[e]` / `free = [e for e in elems if
    # not sol_lim[e]]`).  Its mu stayed frozen at a cap computed against the OTHER elements'
    # then-current mu; those mu's kept moving, so the returned state could have the "limited"
    # element sitting BELOW its cap with more dopant demonstrably reachable, while the stale flag
    # excused the resulting >1 dex target miss in _validity.  Measured: Si+Sn 50/50 at 1350 K
    # returned converged=True with Si at occ 0.0718 (cap 0.1) and -1.049 dex, while the same
    # module's own _mu_cap put a further 0.16 dex of Si within reach at the same cap.
    sol_lim = {e: False for e in elems}
    if enforce_cap and any(ctx.occ_of(N_cs, cols[e]) > OCC_CAP for e in elems):
        mu_map = {e: float(m) for e, m in zip(elems, mu)}
        for _sweep in range(60):
            prev = dict(mu_map)
            for e in elems:
                m_e, _lim = _bisect_mu(ctx, e, targets[e], cols, mu_map, enforce_cap=True)
                mu_map[e] = float(m_e)
            if max(abs(mu_map[e] - prev[e]) for e in elems) < 1e-10:
                break
        mu = np.array([mu_map[e] for e in elems])
        g, EF, mu_vec, N_cs, tot, bracketed = _g_vec(ctx, mu, elems, cols, targets)
    # FINAL-STATE flag (never inherited from an earlier round): an element is site-blocking-
    # limited iff, in the state actually returned, it sits AT its dilute cap AND undershoots.
    for e in elems:
        d = float(np.log10(max(tot[e], 1e-30)) - np.log10(max(targets[e], 1e-30)))
        sol_lim[e] = bool(ctx.occ_of(N_cs, cols[e]) >= OCC_CAP * (1.0 - OCC_AT_CAP_RTOL)
                          and d < -TOL_DEX)

    resid, N_cs, n, p = ctx.Q(EF, mu_vec)
    _, _, sth1, sth2 = ctx.carriers(EF)
    N_defect = np.zeros(db.nD)
    np.add.at(N_defect, db.cs_ID - 1, N_cs)
    for e, m in zip(elems, mu):
        mu_override[e] = float(m)

    val = _validity(db, N_cs, n, p, resid, fixed_conc, sol_lim, Nd_bg, Na_bg)
    if not bracketed:
        val["validity_failures"] = list(val["validity_failures"]) + [
            f"E_F not bracketed in [{EF_LO}, {EF_HI}] eV (clamped)"]
        val["converged"] = False
    meta = {"pO2": pO2, "EcT_fraction": EcT_fraction, "fd": fd,
            "fixed_conc": fixed_conc, "mu_override": mu_override,
            "solver": "engine.solve_codoped(bisect-EF + damped-Newton-mu)",
            "site_blocking_limited": bool(any(sol_lim.values())),
            "site_blocking_limited_per_element": {e: bool(v) for e, v in sol_lim.items()},
            # DEPRECATED aliases kept for API compatibility
            "solubility_limited": bool(any(sol_lim.values())),
            "solubility_limited_per_element": {e: bool(v) for e, v in sol_lim.items()},
            "site_blocking_occ_cap": OCC_CAP,
            "seed_mu_single_dopant": {e: float(v) for e, v in seed.items()},
            # elements whose continuation seed FAILED and fell back to the -6.0 eV cold start
            "seed_failed_cold_start": list(seed_failed),
            "seed_site_blocking_limited": {e: bool(v) for e, v in seed_lim.items()},
            "seed_solubility_limited": {e: bool(v) for e, v in seed_lim.items()},
            "mu_solved": {e: float(m) for e, m in zip(elems, mu)},
            "EF_bracketed": bool(bracketed),
            "outer_iters": len(hist),
            "outer_hist": hist,
            "n_Q_evals": int(ctx.n_Q),
            "last_step_kind": method,
            "strict": bool(strict)}
    meta.update(val)
    if strict:
        _raise_if_invalid(val, f"engine.solve_codoped(T={T}, pO2={pO2}, "
                               f"fixed_conc={fixed_conc})",
                          {"EF": float(EF), "mu_solved": meta["mu_solved"]})
    return EqResult(
        T=float(T), EF=float(EF), n=float(n), p=float(p),
        sth1=float(sth1), sth2=float(sth2), N_cs=N_cs, N_defect=N_defect, mu_vec=mu_vec,
        Ev=ctx.Ev, Ec=ctx.Ec, Eg=ctx.Eg, charge_residual=float(resid), meta=meta)


def quench(db: DefectDB, anneal: EqResult, T_quench: float = 300.0,
           EcT_fraction: float = 0.40, fd: bool = False, sth_flag: float = 1.0,
           frozen_defect_totals: np.ndarray | None = None,
           Nd_bg: float = 0.0, Na_bg: float = 0.0) -> EqResult:
    """Full quench: freeze per-defect totals from the anneal, re-solve E_F at T_quench,
    redistributing charge states within each defect by a Gibbs (Boltzmann) factor.
    Nd_bg/Na_bg: frozen background shallow donor/acceptor density [cm^-3].

    The E_F bracket is found by scan-and-clamp over [EF_LO, EF_HI] instead of six fixed ±2 eV
    widenings, so this can no longer die with a bare scipy "f(a) and f(b) must have different
    signs" ValueError; a genuinely rootless neutrality condition raises DefectEquilibriumError
    with the offending frozen totals."""
    T = T_quench
    Eg, Ev, Ec = (float(x) for x in thermo.band_edges(T, EcT_fraction))
    Nc, Nv = float(thermo.Nc(T)), float(thermo.Nv(T))
    kT = thermo.KB_EV * T
    totals = anneal.N_defect if frozen_defect_totals is None else frozen_defect_totals
    s = float(thermo.dSvib_norm(T))
    dG_vib = 3.0 * kT * s * db.sum_dm
    # EF-dependent relative free energy within a defect (μ term cancels within a defect: same dm)
    dG_rel0 = db.dHo - dG_vib                       # + q*EF added below

    # precompute per-defect charge-state index groups
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
        n, p, sth1, sth2 = _carriers(EF, kT, Ec, Ev, Nc, Nv, sth_flag, fd)
        return float((db.charge * N_cs).sum() + p + sth1 + sth2 - n + Nd_bg - Na_bg)

    br = _bracket_scan(Q, Ev - 3.0, Ec + 3.0)
    if br is None:
        tot = np.asarray(totals, dtype=float)
        bad = np.where(~np.isfinite(tot) | (tot < 0))[0]
        big = np.argsort(-np.nan_to_num(tot, nan=-np.inf))[:5]
        raise DefectEquilibriumError(
            f"quench(T_quench={T}): no E_F root in [{EF_LO}, {EF_HI}] eV — "
            f"Q({EF_LO})={Q(EF_LO):.3e}, Q({EF_HI})={Q(EF_HI):.3e}. "
            f"Frozen totals: {len(bad)} non-finite/negative "
            f"(defect indices {bad[:10].tolist()}); largest totals "
            f"{[(int(i), float(tot[i])) for i in big]}; Nd_bg={Nd_bg:g}, Na_bg={Na_bg:g}.",
            {"T_quench": T, "n_bad_totals": int(bad.size),
             "bad_total_indices": bad[:50].tolist(),
             "largest_totals": [(int(i), float(tot[i])) for i in big],
             "Q_lo": Q(EF_LO), "Q_hi": Q(EF_HI)})
    a, b, _, _ = br
    EF = a if a == b else brentq(Q, a, b, xtol=1e-10, rtol=1e-12, maxiter=200)
    N_cs = N_cs_of(EF)
    n, p, sth1, sth2 = _carriers(EF, kT, Ec, Ev, Nc, Nv, sth_flag, fd)
    N_defect = np.zeros(db.nD)
    np.add.at(N_defect, db.cs_ID - 1, N_cs)
    return EqResult(T=T, EF=EF, n=float(n), p=float(p), sth1=float(sth1), sth2=float(sth2),
                    N_cs=N_cs, N_defect=N_defect, mu_vec=anneal.mu_vec, Ev=Ev, Ec=Ec, Eg=Eg,
                    charge_residual=Q(EF), meta={"quench_from_T": anneal.T, "fd": fd})


# ── helpers for interpreting a result ────────────────────────────────────────
def defect_totals_by_element(db: DefectDB, r: EqResult, elem: str) -> float:
    col = db.col(elem)
    return float((db.dm[:, col] * r.N_cs).sum())


def net_acceptor_donor(db: DefectDB, r: EqResult):
    """Ionized donor (positive-charge defects) and acceptor (negative) densities [cm^-3]."""
    q = db.charge
    Nd = float((np.clip(q, 0, None) * r.N_cs).sum())
    Na = float((np.clip(-q, 0, None) * r.N_cs).sum())
    return Nd, Na
