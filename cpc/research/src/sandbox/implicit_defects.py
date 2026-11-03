"""Implicitly differentiable dilute defect equilibrium and full-quench layer.

The equations, thermochemical references and FD approximation match engine.py.
This layer solves interior fixed-incorporation roots; it does not differentiate
through site ceilings, root fallback, phase transitions, or missing protocols.
Parameters: log10(Tf/K), log10(Nd/cm^-3), energy-basis weights/eV,
log10(incorporated element concentrations/cm^-3), log10(pO2/atm).

No numeric third-party database is embedded here. Energy bases are explicit
selectors on a separately loaded DefectDB. A learned effective energy correction
is not automatically a microscopic DFT energy.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, root
from . import thermo, transport
from .kroger_db import DefectDB

LN10 = np.log(10.0)


class ImplicitDomainError(RuntimeError):
    """An unsupported, invalid or nonregular interior state."""


@dataclass
class ImplicitPrediction:
    values: np.ndarray
    jacobian: np.ndarray
    names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    state: np.ndarray
    diagnostics: dict


def energy_basis(db: DefectDB, kind="native_Ga_vacancy"):
    """Return one shared, parent-consistent formation-energy basis."""
    native = (np.abs(db.dm[:, 2:]).sum(axis=1) == 0)
    if kind == "native_Ga_vacancy":
        mask = native & (db.dm[:, 0] == -1) & (db.dm[:, 1] == 0)
    elif kind == "native_acceptor_parent":
        parents = np.unique(db.cs_ID[native & (db.charge < 0)])
        mask = np.isin(db.cs_ID, parents)
    elif kind == "negative_charge_states":
        # Diagnostic competitor; changes relative charge energies as well.
        mask = db.charge < 0
    elif kind == "none":
        return np.zeros((db.nCS, 0))
    else:
        raise ValueError(kind)
    if not mask.any():
        raise ValueError("Empty energy selector")
    return mask.astype(float)[:, None]


def _fd_log(eta):
    """Stable log of engine._fd's normalized FD approximation and derivative."""
    eta = np.asarray(eta, dtype=float)
    a = np.exp(-0.17 * (eta + 1.) ** 2)
    nu = eta**4 + 50. + 33.6 * eta * (1. - .68 * a)
    dnu = 4 * eta**3 + 33.6 * (1. - .68 * a) + 33.6 * eta * .68 * .34 * (eta + 1.) * a
    xi = 3. * np.sqrt(np.pi) / (4. * nu**.375)
    logden = np.logaddexp(-eta, np.log(xi))
    wn = np.exp(-eta - logden)
    wx = np.exp(np.log(xi) - logden)
    dlog = wn + wx * .375 * dnu / nu
    return -logden, dlog


def _thermo_derivatives(T, logp):
    """Exact derivatives of the existing piecewise thermochemistry in eV/K."""
    c = thermo.J_PER_MOL_TO_EV
    gg = (684.8332 - 112.3935 * (np.log(T) + 1.)
          - 2. * .00796268819 * T - 1080114. / T**2) / c
    if T <= 900.:
        go = (-51.1831467 - 22.25862 * (np.log(T) + 1.)
              - 2. * .01023867 * T + 3. * 1.339947e-6 * T**2 + 76749.55 / T**2) / c
    else:
        go = (24.7432966 - 33.55726 * (np.log(T) + 1.)
              - 2. * .0012348985 * T + 3. * 1.66943333e-8 * T**2 + 539886. / T**2) / c
    dmuO = .5 * (go + thermo.KB_EV * LN10 * logp)
    dmuGa = .5 * (gg - 3. * dmuO)
    n = 1. / np.expm1(thermo.VIBENT_T0 / T)
    ds = thermo.VIBENT_T0**2 / T**3 * n * (1. + n)
    return dmuGa, dmuO, ds


def _carriers_with_derivatives(EF, T):
    """n, p, two STH populations; partial EF and partial T at fixed EF."""
    _, Ev, Ec = map(float, thermo.band_edges(T, .4))
    beta = 1. / (thermo.KB_EV * T)
    dgap = thermo.VARSHNI_A * T * (T + 2 * thermo.VARSHNI_B) / (T + thermo.VARSHNI_B)**2
    dEc, dEv = -.4 * dgap, .6 * dgap
    etas = np.array([(EF - Ec) * beta, (Ev - EF) * beta,
                     (Ev + thermo.E_RELAX_STH1 - EF) * beta,
                     (Ev + thermo.E_RELAX_STH2 - EF) * beta])
    logs, g = _fd_log(etas)
    pref = np.array([float(thermo.Nc(T)), float(thermo.Nv(T)),
                     thermo.N_SITE, thermo.N_SITE])
    values = pref * np.exp(logs)
    de = values * g * beta * np.array([1., -1., -1., -1.])
    dteta = np.array([-dEc, dEv, dEv, dEv]) * beta - etas / T
    dt = values * (g * dteta + np.array([1.5 / T, 1.5 / T, 0., 0.]))
    return values, de, dt


def _charge_log(N, dN, carriers, dcarriers, q, Nd, dNd):
    """Log positive minus log negative charge, with consistently scaled Jacobian."""
    pos, neg = np.maximum(q, 0.), np.maximum(-q, 0.)
    P = pos @ N + carriers[1:].sum() + Nd
    M = neg @ N + carriers[0]
    if not (np.isfinite(P) and np.isfinite(M) and P > 0 and M > 0):
        raise ImplicitDomainError("Nonfinite charge budgets")
    dP = pos @ dN + dcarriers[1:].sum(axis=0) + dNd
    dM = neg @ dN + dcarriers[0]
    return np.log(P) - np.log(M), dP / P - dM / M, P, M


class ImplicitDefectLayer:
    """CPU reference layer with analytic partials and first-order IFT gradients.

    Multiple incorporated elements are supported by the same joint constraint
    block. Background donors are an effective explicit model input. All chemical
    potentials of absent elements retain engine.MU_OFF = -30 eV.
    """
    def __init__(self, db: DefectDB, elements=("Si",), basis=None, hall="heuristic"):
        self.db = db
        self.elements = tuple(elements)
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("Duplicate incorporated element")
        self.cols = [db.col(x) for x in self.elements]
        if any(c < 2 for c in self.cols):
            raise ValueError("Only dopant concentrations are constraints")
        self.atoms = db.dm[:, self.cols].astype(float)
        if np.any(self.atoms < 0):
            raise ValueError("Negative dopant atom counts need a different constraint formulation")
        self.basis = np.zeros((db.nCS, 0)) if basis is None else np.asarray(basis, float)
        if self.basis.ndim != 2 or self.basis.shape[0] != db.nCS:
            raise ValueError("Energy basis shape must be [charge states, basis functions]")
        if hall not in ("heuristic", "rH1"):
            raise ValueError(hall)
        self.hall = hall
        self.parent = db.cs_ID - 1
        self.q = db.charge.astype(float)
        self.k = len(self.elements)
        self.b = self.basis.shape[1]
        self.nhot = 1 + self.k
        self.nu = self.nhot + 1
        self.np = 3 + self.b + self.k
        self.parameter_names = (("log10_Tf_K", "log10_Nd_cm3")
            + tuple(f"energy_weight_{i}_eV" for i in range(self.b))
            + tuple(f"log10_{x}_cm3" for x in self.elements) + ("log10_pO2_atm",))
        self.output_names = ("log10_nH_cm3", "log10_muH_cm2Vs", "log10_sigma_S_cm")
        self.last_hot = None
        self.root_calls = 0
        self.root_evaluations = 0

    def parameters(self, T=1325., Nd=9.1e15, weights=None, doses=(1e18,), pO2=1e-5):
        w = np.zeros(self.b) if weights is None else np.asarray(weights, float)
        return np.r_[np.log10(T), np.log10(Nd), w, np.log10(doses), np.log10(pO2)]

    def _unpack(self, pars):
        p = np.asarray(pars, float)
        if p.shape != (self.np,) or not np.all(np.isfinite(p)):
            raise ValueError(f"Expected {self.np} finite parameters")
        T, Nd = 10.**p[0], 10.**p[1]
        doses = 10.**p[2+self.b:2+self.b+self.k]
        if not (300. < T <= 2200. and Nd > 0 and np.all(doses > 0)):
            raise ImplicitDomainError("Outside the declared positive interior parameter domain")
        return p, T, Nd, doses

    def _hot(self, hot, pars):
        self.root_evaluations += 1
        p, T, Nd, doses = self._unpack(pars)
        db, q = self.db, self.q
        beta = 1. / (thermo.KB_EV * T)
        mu = np.full(len(db.elements), -30.)
        mu[:2] = thermo.mu_Ga_O(T, 10.**p[-1])
        mu[self.cols] = hot[1:]
        s = float(thermo.dSvib_norm(T))
        A = db.dHo + self.basis @ p[2:2+self.b] - 3. / beta * s * db.sum_dm - db.dm @ mu
        x = -(A + q * hot[0]) * beta
        active = (x > -700.) & (x < 300.)
        N = db.prefactor * np.exp(np.clip(x, -700., 300.))
        dU = N[:, None] * np.column_stack([-q * beta, self.atoms * beta]) * active[:, None]
        dP = np.zeros((db.nCS, self.np))
        dga, do, ds = _thermo_derivatives(T, p[-1])
        AT = -3. * thermo.KB_EV * (s + T * ds) * db.sum_dm - db.dm[:, 0] * dga - db.dm[:, 1] * do
        dP[:, 0] = N * (-AT * beta - x / T) * T * LN10
        dP[:, 2:2+self.b] = -N[:, None] * self.basis * beta
        dmuO = .5 / beta * LN10
        dP[:, -1] = N * beta * (db.dm[:, 1] - 1.5 * db.dm[:, 0]) * dmuO
        dP *= active[:, None]
        car, ce, ct = _carriers_with_derivatives(hot[0], T)
        cU = np.zeros((4, self.nhot)); cU[:, 0] = ce
        cP = np.zeros((4, self.np)); cP[:, 0] = ct * T * LN10
        ndU = np.zeros(self.nhot); ndP = np.zeros(self.np); ndP[1] = Nd * LN10
        F0, J0, _, _ = _charge_log(N, np.c_[dU, dP], car, np.c_[cU, cP], q, Nd, np.r_[ndU, ndP])
        C = self.atoms.T @ N
        if self.k and np.any(C <= 0):
            raise ImplicitDomainError("Nonpositive incorporated population")
        F = np.r_[F0, np.log(C) - np.log(doses)]
        jac = np.vstack([J0, self.atoms.T @ np.c_[dU, dP] / C[:, None]])
        for j in range(self.k):
            jac[1+j, self.nhot+2+self.b+j] -= LN10
        return F, jac[:, :self.nhot], jac[:, self.nhot:], N, dU, dP, car

    def _solve_hot(self, pars):
        p, T, _, _ = self._unpack(pars)
        Ec = float(thermo.band_edges(T, .4)[2])
        seed = np.r_[Ec - .15, np.full(self.k, -5.)]
        # Initial atom-budget correction at a fixed EF is only a starting guess.
        for _ in range(6):
            F, J, *_ = self._hot(seed, p)
            for j in range(self.k):
                if abs(J[1+j, 1+j]) > 1e-12:
                    seed[1+j] -= np.clip(F[1+j] / J[1+j, 1+j], -2., 2.)
        seeds = ([self.last_hot.copy()] if self.last_hot is not None else []) + [seed]
        best = None
        for initial in seeds:
            sol = root(lambda u: self._hot(u, p)[0], initial,
                       jac=lambda u: self._hot(u, p)[1], method="hybr",
                       options={"xtol": 2e-11, "maxfev": 160})
            residual = float(np.max(np.abs(self._hot(sol.x, p)[0])))
            if best is None or residual < best[0]:
                best = residual, sol.x
            if residual < 2e-9:
                break
        if best[0] > 2e-8:
            raise ImplicitDomainError(f"Hot root residual {best[0]:g}")
        if np.any(best[1][1:] <= -30.) or np.any(best[1][1:] >= 5.):
            raise ImplicitDomainError("Solved chemical potential is outside the interior box")
        self.last_hot = best[1].copy()
        return best[1]

    def _joint(self, u, pars):
        p, T, Nd, doses = self._unpack(pars)
        Fh, Jh, Ph, Nh, dNhU, dNhP, _ = self._hot(u[:self.nhot], p)
        totals = np.bincount(self.parent, weights=Nh, minlength=self.db.nD)
        dtotalU = np.array([np.bincount(self.parent, weights=dNhU[:, j], minlength=self.db.nD)
                           for j in range(self.nhot)]).T
        dtotalP = np.array([np.bincount(self.parent, weights=dNhP[:, j], minlength=self.db.nD)
                           for j in range(self.np)]).T
        beta = 1. / (thermo.KB_EV * 300.)
        base = (self.db.dHo + self.basis @ p[2:2+self.b]
                - 3. / beta * float(thermo.dSvib_norm(300.)) * self.db.sum_dm)
        x = -(base + self.q * u[-1]) * beta
        mx = np.full(self.db.nD, -np.inf); np.maximum.at(mx, self.parent, x)
        wx = np.exp(np.maximum(x - mx[self.parent], -700.))
        den = np.bincount(self.parent, weights=wx, minlength=self.db.nD)
        pi = wx / den[self.parent]
        Nc = totals[self.parent] * pi
        meanq = np.bincount(self.parent, weights=pi*self.q, minlength=self.db.nD)
        dNU = np.zeros((self.db.nCS, self.nu))
        dNU[:, :self.nhot] = pi[:, None] * dtotalU[self.parent]
        dNU[:, -1] = -beta * Nc * (self.q - meanq[self.parent])
        dNP = pi[:, None] * dtotalP[self.parent]
        for j in range(self.b):
            meanb = np.bincount(self.parent, weights=pi*self.basis[:, j], minlength=self.db.nD)
            dNP[:, 2+j] -= beta * Nc * (self.basis[:, j] - meanb[self.parent])
        car, ce, _ = _carriers_with_derivatives(u[-1], 300.)
        cU = np.zeros((4, self.nu)); cU[:, -1] = ce
        cP = np.zeros((4, self.np))
        ndP = np.zeros(self.np); ndP[1] = Nd*LN10
        Fc, Jc, P, M = _charge_log(Nc, np.c_[dNU, dNP], car, np.c_[cU, cP],
                                   self.q, Nd, np.r_[np.zeros(self.nu), ndP])
        Fu = np.zeros((self.nu, self.nu)); Fu[:self.nhot, :self.nhot] = Jh; Fu[-1] = Jc[:self.nu]
        Fp = np.vstack([Ph, Jc[self.nu:]])
        return np.r_[Fh, Fc], Fu, Fp, Nc, np.c_[dNU, dNP], car, ce, Nh, P, M

    def predict(self, pars, *, require_regular=True):
        self.root_calls += 1
        p, T, Nd, doses = self._unpack(pars)
        hot = self._solve_hot(p)
        Nh = self._hot(hot, p)[3]
        totals = np.bincount(self.parent, weights=Nh, minlength=self.db.nD)
        beta = 1. / (thermo.KB_EV * 300.)
        base = (self.db.dHo + self.basis @ p[2:2+self.b]
                - 3. / beta * float(thermo.dSvib_norm(300.)) * self.db.sum_dm)
        def cold(x):
            rel = -(base + self.q*x)*beta
            mx = np.full(self.db.nD, -np.inf); np.maximum.at(mx,self.parent,rel)
            w = np.exp(np.maximum(rel-mx[self.parent],-700.))
            den = np.bincount(self.parent,weights=w,minlength=self.db.nD)
            nc = totals[self.parent]*w/den[self.parent]
            car = _carriers_with_derivatives(x,300.)[0]
            P = np.maximum(self.q,0.)@nc + car[1:].sum() + Nd
            M = np.maximum(-self.q,0.)@nc + car[0]
            return np.log(P)-np.log(M)
        ef = brentq(cold, -5., 12., xtol=2e-12, rtol=2e-14)
        u = np.r_[hot, ef]
        F, Fu, Fp, N, dN, car, ce, Nh, P, M = self._joint(u, p)
        condition = float(np.linalg.cond(Fu))
        n = float(car[0])
        logn = np.log10(n)
        rh = float(transport.hall_factor(300., n)) if self.hall == "heuristic" else 1.
        drh = (-.5/2.7 if self.hall == "heuristic" and 17. < logn < 19.7 else 0.)
        all_deriv = np.zeros((3, self.nu+self.np))
        all_deriv[0, self.nu-1] = ce[0] / (n*LN10) * (1. - drh/(rh*LN10))
        NI = float(np.abs(self.q) @ N + Nd)
        NII = float(self.q**2 @ N + Nd)
        dNI = np.abs(self.q) @ dN
        dNI[self.nu+1] += Nd*LN10
        mu = float(transport.mu_caughey_thomas(NI, 300.))
        zz = (NI / transport.CT_N_REF)**transport.CT_ALPHA
        dmu_dNI = -(transport.CT_MU_MAX-transport.CT_MU_MIN)*transport.CT_ALPHA*zz/(NI*(1.+zz)**2)
        all_deriv[1] = dmu_dNI*dNI/(mu*LN10)
        all_deriv[2] = all_deriv[0] + all_deriv[1]
        vals = np.array([logn-np.log10(rh), np.log10(mu),
                         np.log10(thermo.Q)+logn-np.log10(rh)+np.log10(mu)])
        root_derivative = -np.linalg.solve(Fu, Fp)
        jac = all_deriv[:, self.nu:] + all_deriv[:, :self.nu] @ root_derivative
        occ = Nh/self.db.prefactor
        dopant_occ = {e: float(occ[self.atoms[:, j]>0].max()) for j,e in enumerate(self.elements)}
        failures = []
        if np.max(np.abs(F)) > 2e-8: failures.append("root_residual")
        if not np.isfinite(condition) or condition > 1e12: failures.append("singular_or_ill_conditioned_root")
        if any(v >= .1*(1.-1e-6) for v in dopant_occ.values()): failures.append("site_ceiling_not_interior")
        if occ.max() > 1.: failures.append("unphysical_site_occupancy")
        if abs(T-900.) < 1e-6: failures.append("thermochemistry_piece_boundary")
        if self.hall == "heuristic" and min(abs(logn-17.),abs(logn-19.7))<1e-8:
            failures.append("hall_interpolation_kink")
        if not np.all(np.isfinite(jac)): failures.append("nonfinite_gradient")
        donor_budget = float(np.maximum(self.q, 0.) @ N + Nd)
        acceptor_budget = float(np.maximum(-self.q, 0.) @ N)
        diagnostics = dict(root_residual_log=float(np.max(np.abs(F))), jacobian_condition=condition,
            gradient_valid=not failures, invalid_reasons=failures,
            max_site_occupancy=float(occ.max()), dopant_site_occupancy=dopant_occ,
            free_n_cm3=n, free_p_cm3=float(car[1]), hall_factor=rh,
            single_electron_Hall_valid=bool(n>car[1]), above_carrier_detection_floor=bool(n/rh>=1e13),
            ionized_absolute_charge_cm3=NI, charge_squared_scatter_budget_cm3=NII,
            compensation_margin=float((donor_budget-acceptor_budget)/max(donor_budget,acceptor_budget,1.)),
            charge_relative_residual=float(abs(P-M)/max(P,M)),
            achieved_concentrations_cm3={e:float(self.atoms[:,j]@Nh) for j,e in enumerate(self.elements)},
            Tf_K=T, Nd_cm3=Nd, pO2_atm=float(10.**p[-1]), hall_convention=self.hall,
            energy_basis_count=self.b, energy_interpretation="effective shared correction unless independently energy-labeled")
        if require_regular and failures:
            raise ImplicitDomainError("; ".join(failures))
        return ImplicitPrediction(vals, jac, self.output_names, self.parameter_names, u, diagnostics)

    def tensor(self, pars):
        """Torch bridge: CPU state solve, analytic first-order backward on input device."""
        import torch
        from torch.autograd.function import once_differentiable
        if pars.dtype != torch.float64:
            raise ValueError("Use float64 for the scientific implicit layer")
        layer = self
        class _ImplicitFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, p):
                result = layer.predict(p.detach().cpu().double().numpy())
                jac = torch.as_tensor(result.jacobian, dtype=p.dtype, device=p.device)
                ctx.save_for_backward(jac)
                return torch.as_tensor(result.values, dtype=p.dtype, device=p.device)

            @staticmethod
            @once_differentiable
            def backward(ctx, output_gradient):
                jac, = ctx.saved_tensors
                return output_gradient @ jac
        return _ImplicitFunction.apply(pars)
