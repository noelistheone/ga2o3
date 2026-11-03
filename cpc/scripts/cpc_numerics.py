"""Forward-only kernels for fair numerical differentiation of the frozen solver.

The root equations, analytic *state* Jacobians, tolerances and warm-start policy
are shared with the original solver. Parameter-Jacobian construction inside the
hot residual is retained in both paths; the forward path never computes the IFT
backsolve or output parameter gradients. No physical fit is changed here.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import brentq

BASE = Path(__file__).resolve().parents[1]
RESEARCH = BASE / 'research'
sys.path.insert(0, str(RESEARCH / 'src'))
from sandbox import thermo, transport
from sandbox.implicit_defects import _carriers_with_derivatives, ImplicitDomainError
from sandbox.temperature_defects import TemperatureDefectLayer


def reset(layer):
    """Cold start for each independently timed Jacobian, not between FD points."""
    layer.last_hot = None
    if hasattr(layer, '_hot_cache'):
        layer._hot_cache = None
    layer.root_calls = layer.root_evaluations = 0
    if hasattr(layer, 'hot_solve_calls'):
        layer.hot_solve_calls = layer.hot_cache_hits = 0


def forward(layer, parameters, *, diagnostics=False):
    """Return the same log Hall density with no parameter IFT computation."""
    p, Tf, Nd, doses = layer._unpack(parameters)
    layer.root_calls += 1
    thermal = isinstance(layer, TemperatureDefectLayer)
    if thermal:
        hot, block = layer._hot_solution(p)
        H, _, _, profile = layer._reference(10**p[layer.iobs], p)
        T = 10**p[layer.iobs]
        entropy = layer._entropy(T)[0]
        logg = layer.logg
        carriers = lambda ef: profile.carriers(ef, T)[0]
    else:
        hot = layer._solve_hot(p)
        block = layer._hot(hot, p)
        T = 300.
        H = layer.db.dHo + layer.basis @ p[2:2+layer.b]
        entropy = float(thermo.dSvib_norm(T))
        logg = np.zeros(layer.db.nCS)
        carriers = lambda ef: _carriers_with_derivatives(ef, T)[0]
    Nh = block[3]
    totals = np.bincount(layer.parent, weights=Nh, minlength=layer.db.nD)
    beta = 1/(thermo.KB_EV*T)
    A = H - 3/beta*entropy*layer.db.sum_dm

    def population(ef):
        z = -(A + layer.q*ef)*beta + logg
        mx = np.full(layer.db.nD, -np.inf)
        np.maximum.at(mx, layer.parent, z)
        shift = z-mx[layer.parent]
        w = np.exp(shift if thermal else np.maximum(shift, -700.))
        pi = w/np.bincount(layer.parent, weights=w, minlength=layer.db.nD)[layer.parent]
        return totals[layer.parent]*pi

    def charge(ef):
        N, car = population(ef), carriers(ef)
        pos = np.maximum(layer.q, 0)@N + car[1:].sum()+Nd
        neg = np.maximum(-layer.q, 0)@N + car[0]
        return (pos-neg)/max(pos, neg, 1.) if thermal else np.log(pos)-np.log(neg)

    ef = brentq(charge, -5., 12., xtol=2e-14 if thermal else 2e-12,
                rtol=1e-15 if thermal else 2e-14)
    car = carriers(ef)
    rh = float(transport.hall_factor(T, car[0])) if layer.hall == 'heuristic' else 1.
    value = np.log10(car[0]/rh)
    residual = max(float(abs(block[0]).max()), abs(charge(ef)))
    if residual > 2e-8 or not np.isfinite(value):
        raise ImplicitDomainError('Forward-only root/finite-value gate failed')
    if diagnostics:
        return value, dict(hot=hot.tolist(), cold_ef_eV=float(ef),
                          residual=residual, free_n_cm3=float(car[0]),
                          root_evaluations=layer.root_evaluations)
    return value


def central_jacobian(layer, p, coordinates, step=1e-5):
    """Centered differences, absolute step in explicit log/eV coordinates."""
    result = []
    for j in coordinates:
        a, b = p.copy(), p.copy()
        a[j] += step
        b[j] -= step
        result.append((forward(layer, a)-forward(layer, b))/(2*step))
    return np.asarray(result)
