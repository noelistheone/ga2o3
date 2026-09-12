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
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, fsolve

from . import thermo
from .kroger_db import DefectDB, N_SITE

MU_OFF = -30.0  # eV, KROGER's "element absent" default


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


def solve_single(db: DefectDB, T: float, pO2: float = 0.2, EcT_fraction: float = 0.40,
                 mu_override: dict | None = None, fixed_conc: dict | None = None,
                 fd: bool = False, sth_flag: float = 1.0,
                 Nd_bg: float = 0.0, Na_bg: float = 0.0) -> EqResult:
    """Solve dark charge-neutrality at one temperature. fixed_conc: {elem: total_cm3} solves μ.
    Nd_bg/Na_bg: fixed shallow background donor/acceptor density [cm^-3] (unintentional Si/H/Fe
    contamination — always-on in real films; UID Ga2O3 is n-type from ~1e17 Si/H)."""
    Eg, Ev, Ec = (float(x) for x in thermo.band_edges(T, EcT_fraction))
    Nc, Nv = float(thermo.Nc(T)), float(thermo.Nv(T))
    mu_override = dict(mu_override or {})
    fixed_conc = dict(fixed_conc or {})

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
        lo, hi = Ev - 3.0, Ec + 3.0
        f = lambda ef: Q(ef, mu_vec)[0]
        flo, fhi = f(lo), f(hi)
        tries = 0
        while flo * fhi > 0 and tries < 6:      # widen bracket if needed
            lo -= 2.0; hi += 2.0; flo, fhi = f(lo), f(hi); tries += 1
        return brentq(f, lo, hi, xtol=1e-10, rtol=1e-12, maxiter=200)

    solubility_limited = False
    if not fixed_conc:
        mu_vec = _mu_vector(db, T, pO2, mu_override)
        EF = solve_EF(mu_vec)
    elif len(fixed_conc) == 1:
        # robust 1-D bisection on the dopant μ: total [X] is strictly increasing in μ_X.
        # Physical upper bound: μ_X where any X-containing charge state reaches 10% of its
        # site prefactor (dilute-approximation validity / site blocking); if the target
        # concentration is unreachable below that cap, return the capped (solubility/site-
        # limited) solution and flag it — mirrors KROGER's kinetic-trapping/solubility logic.
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
        hi = 5.0
        # pull hi down to the site-blocking cap: occ(hi) <= 0.1
        for _ in range(80):
            _, _, _, _, occ_hi = state_at(hi)
            if occ_hi <= 0.1:
                break
            hi -= 0.5
        _, _, _, tot_hi, _ = state_at(hi)
        if tot_hi < target:
            solubility_limited = True
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
        EF, mu_vec, _, _, _ = state_at(mu_x)
        mu_override[elem] = float(mu_x)
    else:
        # multi-element: joint fsolve (legacy path; single-dopant should use the branch above)
        elems = list(fixed_conc.keys())
        cols = [db.col(e) for e in elems]

        def residuals(x):
            EF = x[0]
            mu_vec = _mu_vector(db, T, pO2, mu_override)
            for e, mv in zip(elems, x[1:]):
                mu_vec[db.col(e)] = mv
            N_cs = defect_charge_and_conc(EF, mu_vec)
            res = [Q(EF, mu_vec)[0] / max(Nc, 1e18)]     # scale charge residual
            for e, col in zip(elems, cols):
                tot = float((db.dm[:, col] * N_cs).sum())
                target = fixed_conc[e]
                res.append(np.log10(max(tot, 1.0)) - np.log10(max(target, 1.0)))
            return res

        x0 = [0.5 * (Ev + Ec)] + [-6.0 for _ in elems]
        sol = fsolve(residuals, x0, full_output=True, xtol=1e-10)
        x = sol[0]
        EF = float(x[0])
        for e, mv in zip(elems, x[1:]):
            mu_override[e] = float(mv)
        mu_vec = _mu_vector(db, T, pO2, mu_override)

    resid, N_cs = Q(EF, mu_vec)
    n, p, sth1, sth2 = _carriers(EF, thermo.KB_EV * T, Ec, Ev, Nc, Nv, sth_flag, fd)
    # per-defect totals
    N_defect = np.zeros(db.nD)
    np.add.at(N_defect, db.cs_ID - 1, N_cs)
    return EqResult(T=T, EF=EF, n=float(n), p=float(p), sth1=float(sth1), sth2=float(sth2),
                    N_cs=N_cs, N_defect=N_defect, mu_vec=mu_vec, Ev=Ev, Ec=Ec, Eg=Eg,
                    charge_residual=float(resid),
                    meta={"pO2": pO2, "EcT_fraction": EcT_fraction, "fd": fd,
                          "fixed_conc": fixed_conc, "mu_override": mu_override,
                          "solubility_limited": solubility_limited})


def quench(db: DefectDB, anneal: EqResult, T_quench: float = 300.0,
           EcT_fraction: float = 0.40, fd: bool = False, sth_flag: float = 1.0,
           frozen_defect_totals: np.ndarray | None = None,
           Nd_bg: float = 0.0, Na_bg: float = 0.0) -> EqResult:
    """Full quench: freeze per-defect totals from the anneal, re-solve E_F at T_quench,
    redistributing charge states within each defect by a Gibbs (Boltzmann) factor.
    Nd_bg/Na_bg: frozen background shallow donor/acceptor density [cm^-3]."""
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

    lo, hi = Ev - 3.0, Ec + 3.0
    flo, fhi = Q(lo), Q(hi)
    tries = 0
    while flo * fhi > 0 and tries < 6:
        lo -= 2.0; hi += 2.0; flo, fhi = Q(lo), Q(hi); tries += 1
    EF = brentq(Q, lo, hi, xtol=1e-10, rtol=1e-12, maxiter=200)
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
