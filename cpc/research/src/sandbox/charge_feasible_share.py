"""Input-dependent free-charge threshold for the existing absolute parent pools.

The threshold is conditional on the supplied material, reference inventory and
background.  It guarantees a free-electron floor, not a Hall floor or accuracy.
Frozen earlier allocation and charge modules are deliberately unchanged.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from . import thermo
from .fixed_inventory_charge import InventoryState
from .implicit_defects import ImplicitDomainError, LN10


@dataclass
class FreeChargeThreshold:
    fraction: float
    jacobian: np.ndarray
    reference: object
    complex_fraction: np.ndarray
    isolated_fraction: np.ndarray
    complex_jacobian: np.ndarray
    isolated_jacobian: np.ndarray
    atoms: float
    atoms_jacobian: np.ndarray
    diagnostics: dict


def _cold_floor(layer, pars, *, free_floor_cm3):
    """Return charge-state means and their full raw derivatives at n=n_floor."""
    p, _, _, _ = layer._unpack(pars)
    T = 10 ** p[layer.iobs]
    H, HP, HT, profile = layer._reference(T, p)
    target = float(free_floor_cm3)
    ef = brentq(lambda E: np.log10(profile.carriers(E, T)[0][0]) - np.log10(target),
                -5., 12., xtol=2e-14, rtol=1e-15)
    carriers, ce, ct, cm = profile.carriers(ef, T)
    if ce[0] <= 0:
        raise ImplicitDomainError('Nonincreasing electron density at the threshold')
    raw_car = np.zeros((4, layer.np))
    raw_car[:, layer.iobs] = ct * T * LN10
    raw_car[:, layer.imat] = cm
    ef_jac = -raw_car[0] / ce[0]
    car_jac = raw_car + ce[:, None] * ef_jac
    entropy, entropy_T = layer._entropy(T)
    beta = 1 / (thermo.KB_EV * T)
    A = H - 3 * thermo.KB_EV * T * entropy * layer.db.sum_dm
    AT = HT - 3 * thermo.KB_EV * (entropy + T * entropy_T) * layer.db.sum_dm
    z = -(A + layer.q * ef) * beta
    logw = z + layer.logg
    maxima = np.full(layer.db.nD, -np.inf)
    np.maximum.at(maxima, layer.parent, logw)
    weights = np.exp(logw - maxima[layer.parent])
    pi = weights / np.bincount(layer.parent, weights=weights, minlength=layer.db.nD)[layer.parent]
    logw_jac = -beta * HP - beta * layer.q[:, None] * ef_jac
    logw_jac[:, layer.iobs] += (-beta * AT - z / T) * T * LN10
    means = np.array([np.bincount(layer.parent, weights=pi * v, minlength=layer.db.nD)
                      for v in logw_jac.T]).T
    pi_jac = pi[:, None] * (logw_jac - means[layer.parent])
    ell, ell_jac = layer.fraction(p)
    for plus, neutral in layer.pairs:
        difference = logw[neutral] - logw[plus]
        f, complement = float(expit(difference)), float(expit(-difference))
        dprob = f * ell_jac + ell * f * complement * (logw_jac[neutral] - logw_jac[plus])
        pi[neutral] = ell * f
        pi[plus] = (1 - ell) + ell * complement
        pi_jac[neutral] = dprob
        pi_jac[plus] = -dprob
    qmean = np.bincount(layer.parent, weights=pi * layer.q, minlength=layer.db.nD)
    qmean_jac = np.array([np.bincount(layer.parent, weights=layer.q * v, minlength=layer.db.nD)
                          for v in pi_jac.T]).T
    return qmean, qmean_jac, carriers, car_jac, ef


def free_charge_threshold(solver, pars, chain, *, free_floor_cm3=1e13):
    chain = np.asarray(chain, float)
    layer = solver.layer
    p, _, Nd, _ = layer._unpack(pars)
    if chain.ndim != 2 or chain.shape[0] != layer.np or not np.isfinite(chain).all():
        raise ValueError('Explicit finite raw-parameter derivative chain required')
    if not np.isfinite(free_floor_cm3) or free_floor_cm3 <= 0:
        raise ValueError('Explicit positive free-electron threshold required')
    ref = solver.reference(p)
    c0, dc0 = ref.totals, ref.jacobian @ chain
    cx, isolated, k = solver.complex, solver.receivers, solver.dopant_count
    U, V = float(k[cx] @ c0[cx]), float(c0[isolated].sum())
    W = U + V
    if not np.isfinite([U, V, W]).all() or min(U, V) <= 0:
        raise ImplicitDomainError('Positive reference atom pools required')
    dU, dV = k[cx] @ dc0[cx], dc0[isolated].sum(axis=0)
    dW = dU + dV
    rc, ri = c0[cx] / U, c0[isolated] / V
    drc = (dc0[cx] - rc[:, None] * dU) / U
    dri = (dc0[isolated] - ri[:, None] * dV) / V
    # Hot T and background do not enter the conditional cold weights or the
    # fixed-n carrier root.  All other raw coordinates remain in this key.
    key = (float(free_floor_cm3), tuple(np.delete(p, [0, 1])))
    cache = getattr(solver, '_free_floor_cache', None)
    if cache is not None and cache[0] == key:
        block = cache[1]
    else:
        block = _cold_floor(layer, p, free_floor_cm3=free_floor_cm3)
        solver._free_floor_cache = (key, block)
    mean, raw_mean_jac, carriers, raw_car_jac, ef = block
    dm = raw_mean_jac @ chain
    dcar = raw_car_jac @ chain
    qc, qi = float(rc @ mean[cx]), float(ri @ mean[isolated])
    dqc = rc @ dm[cx] + mean[cx] @ drc
    dqi = ri @ dm[isolated] + mean[isolated] @ dri
    other = ~(cx | isolated)
    QR = float(Nd + c0[other] @ mean[other] + carriers[1:].sum() - carriers[0])
    dQR = Nd * LN10 * chain[1] + mean[other] @ dc0[other] + c0[other] @ dm[other]
    dQR += dcar[1:].sum(axis=0) - dcar[0]
    N, D = QR + W * qc, W * (qi - qc)
    dN, dD = dQR + dW * qc + W * dqc, dW * (qi - qc) + W * (dqi - dqc)
    if not np.isfinite(D) or D <= 0:
        raise ImplicitDomainError('Free-charge threshold requires a positive pool-charge contrast')
    critical = -N / D
    derivative = -(dN + critical * dD) / D
    if not np.isfinite(critical) or not np.isfinite(derivative).all():
        raise ImplicitDomainError('Nonfinite charge-feasible threshold or derivative')
    return FreeChargeThreshold(float(critical), derivative, ref, rc, ri, drc, dri, W, dW,
        dict(free_floor_cm3=float(free_floor_cm3), free_floor_EF_eV=float(ef),
             critical_free_fraction=float(critical), q_per_complex_atom=qc, q_per_isolated_atom=qi,
             threshold_charge_N_cm3=float(N), threshold_charge_D_cm3=float(D),
             threshold_lower_branch='active' if critical > 0 else 'inactive',
             threshold_branch_derivative_regular=bool(critical != 0),
             free_floor_not_Hall_floor=True))


def allocate_feasible_share(solver, pars, chain, psi, psi_jacobian, *,
                            free_floor_cm3=1e13, endpoint=None):
    psi, chain, dpsi = float(psi), np.asarray(chain, float), np.asarray(psi_jacobian, float)
    if not np.isfinite(psi) or abs(psi) > 80:
        raise ValueError('Share logit outside the declared numerical interval [-80,80]')
    if dpsi.shape != (chain.shape[1],) or not np.isfinite(dpsi).all():
        raise ValueError('Finite matching logit derivative chain required')
    if endpoint not in (None, 0, 1):
        raise ValueError('Endpoint must be explicitly0,1 or absent')
    threshold = free_charge_threshold(solver, pars, chain, free_floor_cm3=free_floor_cm3)
    critical = threshold.fraction
    if critical >= 1:
        raise ImplicitDomainError('No interior free-charge-feasible fraction under this reference/background')
    if critical == 0:
        raise ImplicitDomainError('Nonsmooth threshold-branch intersection; smooth Jacobian unavailable')
    lower = max(critical, 0.)
    dlower = threshold.jacobian if critical > 0 else np.zeros(chain.shape[1])
    s, complement = float(expit(psi)), float(expit(-psi))
    if endpoint is not None:
        s, complement = float(endpoint), float(1 - endpoint)
    fi, fc = lower + (1 - lower) * s, (1 - lower) * complement
    derivative = complement * dlower
    if endpoint is None:
        derivative += (1 - lower) * s * complement * dpsi
    ref = threshold.reference
    c0, dc0 = ref.totals, ref.jacobian @ chain
    cx, isolated = solver.complex, solver.receivers
    rc, ri = threshold.complex_fraction, threshold.isolated_fraction
    drc, dri = threshold.complex_jacobian, threshold.isolated_jacobian
    W, dW = threshold.atoms, threshold.atoms_jacobian
    c, dc = c0.copy(), dc0.copy()
    c[cx] = W * fc * rc
    dc[cx] = rc[:, None] * (fc * dW - W * derivative) + W * fc * drc
    c[isolated] = W * fi * ri
    dc[isolated] = ri[:, None] * (fi * dW + W * derivative) + W * fi * dri
    layer = solver.layer
    _, _, _, doses = layer._unpack(pars)
    achieved = solver.parent_atoms.T @ c
    expected = doses[:, None] * LN10 * chain[2 + layer.b:2 + layer.b + layer.k]
    atom_error = float(np.max(abs(achieved / doses - 1)))
    derivative_error = float(np.max(abs(solver.parent_atoms.T @ dc - expected) / (1 + doses[:, None] * LN10)))
    occupancy = c / layer.db.prefactor[layer.first]
    foreign = np.any(solver.parent_atoms > 0, axis=1)
    if np.any(c < 0) or not np.isfinite(c).all() or not np.isfinite(dc).all():
        raise ImplicitDomainError('Nonpositive/nonfinite charge-feasible inventory or derivative')
    if atom_error > 3e-8 or derivative_error > 3e-7:
        raise ImplicitDomainError('Charge-feasible foreign atom/derivative budget failed')
    if occupancy.max() > 1 or occupancy[foreign].max() >= .1 * (1 - 1e-6):
        raise ImplicitDomainError('Charge-feasible parent dilution domain failed')
    host = layer.db.dm[layer.first, :2].T @ (c - c0)
    if abs(host[1]) > max(W, 1) * 1e-12:
        raise ImplicitDomainError('Unexpected O exchange in the declared allocation')
    result = InventoryState(c, dc, dict(**ref.diagnostics, **threshold.diagnostics,
        excess_share_logit=psi, excess_share_coordinate=s, feasible_lower_fraction=float(lower),
        isolated_atom_fraction_before=float(ref.totals[isolated].sum() / W),
        isolated_atom_fraction_after=fi, complex_atom_fraction_after=fc,
        free_charge_threshold_passed_by_construction=True, endpoint=endpoint,
        foreign_atom_relative_error=atom_error, foreign_atom_derivative_error=derivative_error,
        max_parent_fraction=float(occupancy.max()), native_inventory_changed=False,
        host_Ga_O_inventory_change_cm3=host.tolist(), host_atom_exchange_scope='openGa endpoint;O/foreign conserved',
        allocation_interpretation='learned excess fraction above a conditional free-charge floor;not measured kinetics'))
    solver.last_inventory = result
    return result
