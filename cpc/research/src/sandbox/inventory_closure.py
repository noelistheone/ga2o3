"""Differentiable effective parent-stock closures for incorporated Si or Sn.

The map is an auditable open-Ga stock construction, not a reaction mechanism,
rate equation, or independently measured defect inventory. Energies are supplied
unchanged. The cold carrier state is solved again; output densities are never
linearly blended. See docs/ML_SANDBOX_INVENTORY_PHASE7_20261017.md.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from . import thermo, transport
from .implicit_defects import (
    ImplicitDefectLayer, ImplicitPrediction, ImplicitDomainError, LN10,
    _carriers_with_derivatives, _charge_log,
)
from .property_metric_learning import element_database


class InventoryClosureLayer(ImplicitDefectLayer):
    """Equilibrate hot, transform parent stocks, re-equilibrate cold charges.

    The last parameter is eta in [0,1], following the original layer parameters.
    donor allocation is fixed equally between the two audited pure Ga-site
    substitution parents. Association is a composition-conserving control.
    """
    def __init__(self, full_db, element, basis=None, hall="heuristic", map_kind="fill"):
        if element not in ("Si", "Sn") or map_kind not in ("fill", "association"):
            raise ValueError("Only audited Si/Sn fill or association maps are supported")
        db, keep = element_database(full_db, element)
        reduced_basis = None if basis is None else np.asarray(basis)[keep]
        super().__init__(db, (element,), reduced_basis, hall)
        self.base_np = self.np
        self.inventory_parameter_names = self.parameter_names + ("eta_inventory",)
        self.map_kind = map_kind
        self.first = np.array([np.flatnonzero(self.parent == j)[0] for j in range(db.nD)])
        self.original_parents = full_db.cs_ID[keep[self.first]].astype(int)
        self.parent_dm = db.dm[self.first].astype(float)
        for j in range(db.nD):
            if not np.all(db.dm[self.parent == j] == self.parent_dm[j]):
                raise ValueError("Charge states of one parent have different atom composition")
        col = db.col(element)
        self.kX = self.parent_dm[:, col]
        self.vGa = -self.parent_dm[:, 0] - self.kX
        # Exact removal of other impurities occurred in element_database.
        self.affected = (self.parent_dm[:, 1] == 0) & (self.vGa >= 1)
        donor_ids = (169, 170) if element == "Si" else (151, 152)
        donors = [int(np.flatnonzero(self.original_parents == x)[0]) for x in donor_ids]
        wanted = np.zeros(len(db.elements)); wanted[0] = -1; wanted[col] = 1
        for j in donors:
            states = self.parent == j
            if not (np.array_equal(self.parent_dm[j], wanted)
                    and states.sum() == 1 and self.q[states][0] == 1):
                raise ValueError("Audited substitution-donor topology changed")
        native = int(np.flatnonzero(self.original_parents == 15)[0])
        self.S = np.eye(db.nD)
        for j in np.flatnonzero(self.affected):
            if map_kind == "association" and self.kX[j] == 0:
                continue
            self.S[:, j] = 0
            for d in donors:
                self.S[d, j] += self.kX[j] / len(donors)
            if map_kind == "association":
                self.S[native, j] += self.vGa[j]
        self.D = self.S - np.eye(db.nD)
        target = np.zeros_like(self.parent_dm)
        if map_kind == "fill":
            target[self.affected, 0] = self.vGa[self.affected]
        # dm.T @ D columns describe composition added to each parent.
        if not np.allclose(self.parent_dm.T @ self.D, target.T, atol=1e-14, rtol=0):
            raise ValueError("Map violates its declared atom/reservoir accounting")
        if np.any(self.S < 0):
            raise ValueError("Endpoint map is not nonnegative")
        self.map_audit = dict(
            element=element, kind=map_kind, donor_original_parent_ids=list(donor_ids),
            donor_allocation=[.5, .5],
            affected_original_parent_ids=self.original_parents[self.affected].tolist(),
            changed_original_parent_ids=self.original_parents[np.any(self.D != 0, axis=0)].tolist(),
            host_boundary="open Ga reservoir" if map_kind == "fill" else "composition conserving",
            interpretation="effective stock closure; no elementary path or rate is asserted",
        )

    def inventory_parameters(self, *, eta=0., **kwargs):
        return np.r_[super().parameters(**kwargs), eta]

    def predict(self, pars, *, require_regular=True):
        self.root_calls += 1
        v = np.asarray(pars, float)
        if v.shape != (self.base_np + 1,) or not np.all(np.isfinite(v)):
            raise ValueError("Expected original layer parameters followed by finite eta")
        p, eta = v[:-1], float(v[-1])
        if not 0 <= eta <= 1:
            raise ImplicitDomainError("Inventory fraction must lie in [0,1]")
        p, T, Nd, doses = self._unpack(p)
        hot = self._solve_hot(p)
        Fh, Jh, Ph, Nh, dNhU, dNhP, _ = self._hot(hot, p)
        C0 = np.bincount(self.parent, weights=Nh, minlength=self.db.nD)
        C0U = np.array([np.bincount(self.parent, weights=a, minlength=self.db.nD)
                       for a in dNhU.T]).T
        C0P = np.array([np.bincount(self.parent, weights=a, minlength=self.db.nD)
                       for a in dNhP.T]).T
        delta = self.D @ C0
        C = C0 + eta * delta
        CU = C0U + eta * (self.D @ C0U)
        CP = np.c_[C0P + eta * (self.D @ C0P), delta]
        beta = 1 / (thermo.KB_EV * 300.)
        base = (self.db.dHo + self.basis @ p[2:2+self.b]
                - 3 / beta * float(thermo.dSvib_norm(300.)) * self.db.sum_dm)

        def populations(ef):
            x = -(base + self.q * ef) * beta
            mx = np.full(self.db.nD, -np.inf); np.maximum.at(mx, self.parent, x)
            w = np.exp(np.maximum(x - mx[self.parent], -700.))
            pi = w / np.bincount(self.parent, weights=w, minlength=self.db.nD)[self.parent]
            N = C[self.parent] * pi
            mq = np.bincount(self.parent, weights=pi*self.q, minlength=self.db.nD)
            return N, pi, mq

        def cold(ef):
            N, _, _ = populations(ef)
            car = _carriers_with_derivatives(ef, 300.)[0]
            return _charge_log(N, np.zeros((self.db.nCS, 1)), car,
                               np.zeros((4, 1)), self.q, Nd, np.zeros(1))[0]

        ef = brentq(cold, -5., 12., xtol=2e-12, rtol=2e-14)
        N, pi, mq = populations(ef)
        dNU = np.zeros((self.db.nCS, self.nu))
        dNU[:, :self.nhot] = pi[:, None] * CU[self.parent]
        dNU[:, -1] = -beta * N * (self.q - mq[self.parent])
        dNP = pi[:, None] * CP[self.parent]
        for j in range(self.b):
            mb = np.bincount(self.parent, weights=pi*self.basis[:, j], minlength=self.db.nD)
            dNP[:, 2+j] -= beta * N * (self.basis[:, j] - mb[self.parent])
        car, ce, _ = _carriers_with_derivatives(ef, 300.)
        cU = np.zeros((4, self.nu)); cU[:, -1] = ce
        cP = np.zeros((4, self.base_np+1))
        ndP = np.zeros(self.base_np+1); ndP[1] = Nd*LN10
        Fc, Jc, P, M = _charge_log(N, np.c_[dNU, dNP], car, np.c_[cU, cP],
                                   self.q, Nd, np.r_[np.zeros(self.nu), ndP])
        Fu = np.zeros((self.nu, self.nu)); Fu[:self.nhot, :self.nhot] = Jh
        Fu[-1] = Jc[:self.nu]
        Fp = np.zeros((self.nu, self.base_np+1)); Fp[:self.nhot, :self.base_np] = Ph
        Fp[-1] = Jc[self.nu:]
        root_derivative = -np.linalg.solve(Fu, Fp)
        condition = float(np.linalg.cond(Fu))
        n = float(car[0]); logn = np.log10(n)
        rh = float(transport.hall_factor(300., n)) if self.hall == "heuristic" else 1.
        drh = -.5/2.7 if self.hall == "heuristic" and 17 < logn < 19.7 else 0.
        out = np.zeros((3, self.nu+self.base_np+1))
        out[0, self.nu-1] = ce[0]/(n*LN10)*(1-drh/(rh*LN10))
        NI = float(np.abs(self.q)@N + Nd)
        dNI = np.abs(self.q)@np.c_[dNU, dNP]; dNI[self.nu+1] += Nd*LN10
        mu = float(transport.mu_caughey_thomas(NI, 300.))
        zz = (NI/transport.CT_N_REF)**transport.CT_ALPHA
        dmudNI = -(transport.CT_MU_MAX-transport.CT_MU_MIN)*transport.CT_ALPHA*zz/(NI*(1+zz)**2)
        out[1] = dmudNI*dNI/(mu*LN10); out[2] = out[0]+out[1]
        jac = out[:, self.nu:] + out[:, :self.nu]@root_derivative
        vals = np.array([logn-np.log10(rh), np.log10(mu),
                         np.log10(thermo.Q)+logn-np.log10(rh)+np.log10(mu)])
        hot_occ = Nh/self.db.prefactor
        # Cold charge probabilities may redistribute but each parent stock must
        # also satisfy the declared dilute endpoint occupancy check.
        minpref = np.array([self.db.prefactor[self.parent == j].min() for j in range(self.db.nD)])
        parent_occ = C/minpref
        dopant_occ = float(max(hot_occ[self.atoms[:, 0] > 0].max(),
                              parent_occ[self.kX > 0].max()))
        reasons = []
        residual = float(max(np.abs(Fh).max(), abs(Fc)))
        if residual > 2e-8: reasons.append("root_residual")
        if not np.isfinite(condition) or condition > 1e12: reasons.append("singular_or_ill_conditioned_root")
        if dopant_occ >= .1*(1-1e-6): reasons.append("site_ceiling_not_interior")
        if max(hot_occ.max(), parent_occ.max()) > 1: reasons.append("unphysical_site_occupancy")
        if C.min() < -1e-8*max(C.max(), 1): reasons.append("negative_parent_stock")
        if abs(T-900) < 1e-6: reasons.append("thermochemistry_piece_boundary")
        if self.hall == "heuristic" and min(abs(logn-17), abs(logn-19.7)) < 1e-8:
            reasons.append("hall_interpolation_kink")
        if not np.all(np.isfinite(jac)): reasons.append("nonfinite_gradient")
        variance = np.bincount(self.parent, weights=pi*(self.q-mq[self.parent])**2, minlength=self.db.nD)
        QE = float(ce[1:].sum()-ce[0]-beta*(C@variance))
        reservoir = eta*float(self.vGa[self.affected]@C0[self.affected]) if self.map_kind == "fill" else 0.
        atom_change = self.parent_dm.T@(C-C0)
        stock_jac = np.c_[C0P + C0U@(-np.linalg.solve(Jh, Ph)), np.zeros(self.db.nD)]
        stock_jac = stock_jac + eta*(self.D@stock_jac)
        stock_jac[:, -1] = delta
        diagnostics = dict(
            root_residual_log=residual, charge_relative_residual=float(abs(P-M)/max(P,M)),
            jacobian_condition=condition, gradient_valid=not reasons, invalid_reasons=reasons,
            free_n_cm3=n, free_p_cm3=float(car[1]), hall_factor=rh,
            single_electron_Hall_valid=bool(n > car[1]), above_carrier_detection_floor=bool(n/rh >= 1e13),
            max_site_occupancy=float(max(hot_occ.max(), parent_occ.max())),
            dopant_site_occupancy={self.elements[0]:dopant_occ},
            achieved_concentrations_cm3={self.elements[0]:float(self.kX@C)},
            hot_achieved_concentrations_cm3={self.elements[0]:float(self.kX@C0)},
            atom_change_cm3=atom_change.tolist(), open_Ga_consumed_cm3=reservoir,
            ionized_absolute_charge_cm3=NI, charge_squared_scatter_budget_cm3=float(self.q**2@N+Nd),
            cold_Q_EF_cm3_per_eV=QE, eta=eta, map_kind=self.map_kind,
            vacancy_equivalent_hot_cm3=float(self.vGa[self.affected]@C0[self.affected]),
            Tf_K=T, Nd_cm3=Nd, pO2_atm=float(10**p[-1]), hall_convention=self.hall,
            energy_basis_count=self.b, inventory_interpretation="effective open-stock closure, not measured kinetics",
        )
        self.last_inventory = dict(hot=C0, mixed=C, jacobian=stock_jac, charge=N, pi=pi,
                                   original_parent_ids=self.original_parents, hot_state=hot)
        if require_regular and reasons:
            raise ImplicitDomainError("; ".join(reasons))
        return ImplicitPrediction(vals, jac, self.output_names, self.inventory_parameter_names,
                                  np.r_[hot, ef], diagnostics)

    def tensor(self, pars):
        # The inherited bridge intentionally calls self.predict dynamically.
        return super().tensor(pars)
