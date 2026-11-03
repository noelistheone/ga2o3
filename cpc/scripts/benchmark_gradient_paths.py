"""Audit preparation and charge-occupation contributions to the Hall Jacobian.

No fit, catalogue label, or prediction model is modified. The physical settings
are the eight unique conditions of the archived CPC numerical benchmark. The
fixed-inventory control is differentiated analytically and independently by
central differences with parent totals held at their baseline values.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from cpc_numerics import BASE, central_jacobian, reset
from sandbox import thermo, transport
from sandbox.kroger_db import load, NPZ
from sandbox.energy_learning import relative_charge_records
from sandbox.implicit_defects import ImplicitDefectLayer, _carriers_with_derivatives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(version=1, elements=['Si', 'Sn'], Tf_K=[1100., 1500.],
                    C_cm3=[3e17, 3e19], Nd_cm3=9.1e15, pO2_atm=1e-5,
                    Tmeasurement_K=300., fd_steps_eV=[1e-4, 1e-5],
                    primary_step_eV=1e-5, coordinate_order='all stored relative labels',
                    scope='mechanism of the numerical derivative; no materials refit',
                    frozen_control='hold ALL parent concentrations at baseline; recompute cold charge occupations and neutrality',
                    score='Euclidean norm of omitted inventory contribution / norm of full energetic Jacobian; all coordinates retained',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest = out/'protocol.json'
    if dest.exists() and json.loads(dest.read_text()) != protocol:
        raise RuntimeError('Protocol differs; select a new output directory')
    dest.write_text(json.dumps(protocol, indent=2)+'\n')
    db = load()
    records, _ = relative_charge_records(db, np.load(NPZ)['cs_num_each_site'])
    detail, checks, rows = [], [], []
    for element, Tf, C in product(protocol['elements'], protocol['Tf_K'], protocol['C_cm3']):
        labels = [r for r in records if r['element'] == element]
        B = np.zeros((db.nCS, len(labels)))
        for j, r in enumerate(labels):
            B[r['index'], j] = -r['delta_q']
        layer = ImplicitDefectLayer(db, (element,), B)
        p = layer.parameters(T=Tf, Nd=9.1e15, doses=(C,), pO2=1e-5)
        reset(layer)
        pred = layer.predict(p)
        F, Fu, Fp, Nc, dNc, car, ce, Nh, pos, neg = layer._joint(pred.state, p)
        totals = np.bincount(layer.parent, weights=Nh, minlength=db.nD)
        beta = 1/(thermo.KB_EV*300.)

        def occupation(pars, ef):
            energy = db.dHo + B@pars[2:2+len(labels)] - 3/beta*float(thermo.dSvib_norm(300.))*db.sum_dm
            x = -(energy+layer.q*ef)*beta
            mx = np.full(db.nD, -np.inf)
            np.maximum.at(mx, layer.parent, x)
            w = np.exp(np.maximum(x-mx[layer.parent], -700.))
            return w/np.bincount(layer.parent, weights=w, minlength=db.nD)[layer.parent]

        def frozen(pars):
            def charge(ef):
                N = totals[layer.parent]*occupation(pars, ef)
                ca = _carriers_with_derivatives(ef, 300.)[0]
                return np.log(np.maximum(layer.q, 0.)@N + ca[1:].sum()+9.1e15)-np.log(np.maximum(-layer.q, 0.)@N+ca[0])
            ef = brentq(charge, -5., 12., xtol=2e-12, rtol=2e-14)
            n = _carriers_with_derivatives(ef, 300.)[0][0]
            return np.log10(n/float(transport.hall_factor(300., n)))

        pi = occupation(p, pred.state[-1])
        meanq = np.bincount(layer.parent, weights=pi*layer.q, minlength=db.nD)
        Qe = float(-beta*np.sum(Nc*(layer.q-meanq[layer.parent])**2)+ce[1:].sum()-ce[0])
        Qxi = -beta*np.sum((Nc*(layer.q-meanq[layer.parent]))[:, None]*B, axis=0)
        rh = float(transport.hall_factor(300., car[0]))
        drh_dlogn = -.5/2.7 if 17. < np.log10(car[0]) < 19.7 else 0.
        ge = ce[0]/(car[0]*np.log(10))*(1-drh_dlogn/(rh*np.log(10)))
        cold = -ge*Qxi/Qe
        full = pred.jacobian[0, 2:2+len(labels)]
        # Independent hot-inventory derivative, including chemical-potential response.
        _, Jh, Ph, _, dNhU, dNhP, _ = layer._hot(pred.state[:layer.nhot], p)
        dNh = dNhP-dNhU@np.linalg.solve(Jh, Ph)
        dc = np.array([np.bincount(layer.parent, weights=dNh[:, j], minlength=db.nD)
                       for j in range(2, 2+len(labels))]).T
        stock = -ge*(meanq@dc)/Qe
        common = dict(element=element, Tf_K=Tf, C_cm3=C, parameters=len(labels))
        for step in protocol['fd_steps_eV']:
            reset(layer)
            fd = central_jacobian(layer, p, range(2, 2+len(labels)), step)
            for j, label in enumerate(labels):
                plus, minus = p.copy(), p.copy()
                plus[2+j] += step; minus[2+j] -= step
                fd_cold = (frozen(plus)-frozen(minus))/(2*step)
                checks.append(dict(**common, step=step, label=label['label_id'], full=full[j], full_FD=fd[j],
                                   fixed_inventory=cold[j], fixed_inventory_FD=fd_cold,
                                   full_scaled_error=abs(full[j]-fd[j])/max(1., abs(full[j]), abs(fd[j])),
                                   cold_scaled_error=abs(cold[j]-fd_cold)/max(1., abs(cold[j]), abs(fd_cold))))
        for j, label in enumerate(labels):
            detail.append(dict(**common, label=label['label_id'], charge=int(db.charge[label['index']]),
                               full=full[j], occupation=cold[j], inventory=stock[j],
                               decomposition_error=abs(full[j]-cold[j]-stock[j])))
        rows.append(dict(**common, full_norm=float(np.linalg.norm(full)),
                         omitted_inventory_relative_norm=float(np.linalg.norm(stock)/np.linalg.norm(full)),
                         full_vs_fixed_relative_norm=float(np.linalg.norm(full-cold)/np.linalg.norm(full)),
                         max_decomposition_error=float(np.max(np.abs(full-cold-stock))),
                         baseline_output_difference=abs(frozen(p)-pred.values[0]),
                         root_residual=pred.diagnostics['root_residual_log'],
                         condition=pred.diagnostics['jacobian_condition']))
    pd.DataFrame(rows).to_csv(out/'settings.csv', index=False)
    pd.DataFrame(detail).to_csv(out/'coordinates.csv', index=False)
    pd.DataFrame(checks).to_csv(out/'finite_differences.csv', index=False)
    primary = [r for r in checks if r['step']==protocol['primary_step_eV']]
    summary = dict(settings=len(rows), energetic_coordinates=len(detail),
                   derivative_comparisons=2*len(checks),
                   max_full_gradient_error=max(r['full_scaled_error'] for r in primary),
                   max_frozen_gradient_error=max(r['cold_scaled_error'] for r in primary),
                   max_additive_error=max(r['max_decomposition_error'] for r in rows),
                   max_baseline_value_error=max(r['baseline_output_difference'] for r in rows),
                   omitted_relative_min=min(r['omitted_inventory_relative_norm'] for r in rows),
                   omitted_relative_max=max(r['omitted_inventory_relative_norm'] for r in rows))
    summary['passed'] = bool(summary['max_full_gradient_error']<3e-4 and summary['max_frozen_gradient_error']<3e-4
                             and summary['max_additive_error']<1e-8 and summary['max_baseline_value_error']<1e-9)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    if not summary['passed']:
        raise RuntimeError('Gradient-path audit did not pass')


if __name__ == '__main__':
    main()
