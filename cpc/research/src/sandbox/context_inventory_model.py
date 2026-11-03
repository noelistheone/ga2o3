"""A shared conditional inventory relation inside the frozen Hall Sandbox.

This is an effective endpoint model. Bounded numerical coordinates, a KL tilt
and an optimizer certificate do not establish microscopic kinetics or accuracy.
"""
from dataclasses import replace
import numpy as np
from .relative_inventory_model import RelativeInventoryModel


def field_features(row, basis):
    if basis not in ('constant', 'dose', 'process'):
        raise ValueError('Explicit constant/dose/process field basis required')
    logc = float(np.log10(row.incorporated_cm3))
    v = [1.]
    if basis != 'constant':
        v.append((logc - 18.) / 2.)
    if basis == 'process':
        if not np.isfinite(row.growth_T_C):
            raise ValueError('Process field requires traceable reported growth temperature')
        v.extend([(float(row.growth_T_C) - 700.) / 300., float(row.element == 'Sn')])
    return np.asarray(v)


def bounded_field(row, coefficients, basis):
    x = field_features(row, basis)
    theta = np.asarray(coefficients, float)
    if theta.shape != x.shape or not np.isfinite(theta).all():
        raise ValueError('Finite coefficients matching the declared field basis required')
    z = float(x @ theta / 6.)
    th = float(np.tanh(z))
    factor = 1. - th * th
    dx = np.zeros((3 if row.H_known else 2) + 1)
    if basis != 'constant':
        dx[0] = factor * theta[1] / 2.
    if basis == 'process':
        dx[-1] = factor * theta[2] / 300.
    return 6. * th, factor * x, dx


class ContextInventoryModel:
    """Lift h_theta(C,u) through exact existing state and transport derivatives."""
    def __init__(self, *, basis, field, profile, db=None):
        if basis not in ('none', 'constant', 'dose', 'process') or field not in ('inventory', 'output'):
            raise ValueError('Declared physical field and basis required')
        self.basis = basis
        self.field = field
        self.profile = profile
        self.db = db
        self.models = {}
        n = dict(none=0, constant=1, dose=2, process=4)[basis]
        self.parameter_names = ('logit_Tf', 'log10_background') + tuple(f'field_coefficient_{i}' for i in range(n))

    def engine_for(self, row):
        pressure = float(row.scenario_pO2_atm)
        if not np.isfinite(pressure) or pressure <= 0:
            raise ValueError('An explicit per-condition positive oxygen scenario is required')
        if pressure not in self.models:
            self.models[pressure] = RelativeInventoryModel(dict(field=self.field), profile=self.profile,
                pO2_atm=pressure, db=self.db)
        return self.models[pressure]

    def point(self, row, parameters, *, source_kind='none', source_shift=0., require_observation=True):
        w = np.asarray(parameters, float)
        if w.shape != (len(self.parameter_names),) or not np.isfinite(w).all():
            raise ValueError('Finite parameters matching the saved context model required')
        if self.basis == 'none':
            h, dh, hx = 0., np.zeros(0), np.zeros((3 if row.H_known else 2) + 1)
        else:
            h, dh, hx = bounded_field(row, w[2:], self.basis)
        raw = self.engine_for(row).point(row, np.r_[w[:2], h], source_kind=source_kind,
            source_shift=source_shift, require_observation=require_observation)
        chain = np.zeros((3, len(w)))
        chain[:2, :2] = np.eye(2)
        chain[2, 2:] = dh
        # The existing input partial holds h fixed. The outer-product term is
        # essential when dose changes the learned field itself.
        jac = raw.jacobian @ chain
        tj = raw.transport_jacobian @ chain
        ij = np.c_[raw.input_jacobian, np.zeros(len(raw.values))] + raw.jacobian[:, 2, None] * hx[None, :]
        tij = np.c_[raw.transport_input_jacobian, np.zeros(len(raw.transport_values))] + raw.transport_jacobian[:, 2, None] * hx[None, :]
        diagnostics = dict(raw.diagnostics, context_field_basis=self.basis, context_field_value=h,
            context_field_coefficients=w[2:].tolist(), input_field_partial=hx.tolist(),
            numerical_field_range='open interval(-6,6); representation choice, not a physical bound',
            per_condition_oxygen_input_kind=getattr(row, 'oxygen_input_kind', 'explicit supplied scenario'),
            context_interpretation='shared conditional frozen-inventory endpoint;not measured kinetics or a causal process law')
        return replace(raw, jacobian=jac, transport_jacobian=tj, input_jacobian=ij,
            transport_input_jacobian=tij, input_names=raw.input_names + ('growth_T_C',), diagnostics=diagnostics)

    def counters(self):
        values = [m.counters() for m in self.models.values()]
        return {k:sum(v[k] for v in values) for k in ('hot_solve_calls', 'hot_cache_hits', 'cold_solve_calls')}
