"""Frozen-before-execution gates for charge-feasible predictive learning."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from sandbox.charge_feasible_model import ChargeFeasibleModel
from sandbox.charge_feasible_share import free_charge_threshold, allocate_feasible_share
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.context_saved_model import INPUT_NAMES
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

OUT = e.BASE / 'phase27_charge_feasible'
PARENT = e.BASE / 'phase26_pool_share'


def vector(point):
    return np.r_[point.values[0], point.transport_values]


def main():
    assert not (OUT / 'numerical_gates.json').exists(), 'Preserve an earlier result before versioned repair'
    manifest = e.BASE / 'artifact_manifest_phase26.json'
    assert e.sha(manifest) == '4489891819c035c45b3bae9c128f6a9e79b6f63cd79742c755d3a42eaacc6bcc'
    table = read(e.BASE / 'phase25_shared_process/data/conditions33.csv')
    selected = [table[table.element.eq('Si') & table.H_known.eq(True)].iloc[0],
        table[table.element.eq('Si') & table.H_known.eq(False) & table.family.ne('10.1063/1.5047214')].iloc[0],
        table[table.element.eq('Sn')].iloc[0],
        table[table.sample_id.eq('10.1063/1.5047214::target-1')].iloc[0]]
    cases = [SimpleNamespace(**{k: e.clean(r[k]) for k in INPUT_NAMES}) for r in selected]
    code = [Path(__file__), ROOT / 'src/sandbox/charge_feasible_share.py', ROOT / 'src/sandbox/charge_feasible_model.py',
        ROOT / 'src/sandbox/pool_share_model.py', ROOT / 'src/sandbox/fixed_inventory_charge.py',
        ROOT / 'src/sandbox/coherent_transport.py', ROOT / 'src/sandbox/coherent_fermi.py']
    inputs = [manifest, e.BASE / 'phase25_shared_process/data/conditions33.csv',
        PARENT / 'inner_domain_diagnosis/protocol_before_replay.json',
        PARENT / 'inner_domain_diagnosis/point_diagnoses.csv',
        ROOT / 'docs/ML_SANDBOX_CHARGE_FEASIBLE_NEXT_20261018.md']
    diagnostic_protocol = json.loads(inputs[2].read_text())
    for key in sorted({r['fit_key'] for r in diagnostic_protocol['cases']}):
        inputs.append(PARENT / 'fits' / (key + '.json'))
    protocol = dict(created_utc=e.now(), purpose='conditional threshold,atom/derivative and fixed-state gates;not accuracy',
        cases=[vars(c) for c in cases], readouts=['physical', 'endpoint_envelope'], bases=['constant', 'dose', 'process'],
        free_floor_cm3=1e13, Hall_floor_log10=13., Hall_floor_not_guaranteed_by_free_threshold=True,
        envelope_definition='log-Hall interpolation of physical endpoint interval intersected with the existing Hall detection floor;no midpoint state',
        parameter_step=1e-3, input_log_step=1e-3, growth_step_C=.2, source_step=1e-3,
        derivative_scaled_tolerance=3e-5, saturation_logits=[-80., -30., -6., 0., 6., 30., 80.],
        retained_failed_inner_cases=diagnostic_protocol['cases'], new_shared_training=0, new_measurements=0,
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in dict.fromkeys(inputs)})
    e.write(OUT / 'gate_protocol_before.json', protocol)
    checks, states, errors = [], [], []
    def check(name, error, tol):
        checks.append(dict(name=name, value=float(error), tolerance=tol, passed=bool(np.isfinite(error) and error <= tol)))
    profile = published_coherent_profile()
    for readout in ('physical', 'endpoint_envelope'):
        for basis in ('constant', 'dose', 'process'):
            model = ChargeFeasibleModel(basis=basis, profile=profile, readout=readout)
            w = np.r_[.5, 16., np.array([2.4, .4, -.2, .1])[:len(model.parameter_names) - 2]]
            for i, row in enumerate(cases):
                label = f'{readout}/{basis}/{i}'
                try:
                    point = model.point(row, w, source_kind='share', source_shift=.1, require_observation=False)
                    numerical = []
                    for k in range(len(w)):
                        step = np.zeros(len(w)); step[k] = .001
                        numerical.append((vector(model.point(row, w + step, source_kind='share', source_shift=.1, require_observation=False)) -
                            vector(model.point(row, w - step, source_kind='share', source_shift=.1, require_observation=False))) / .002)
                    analytic = np.vstack([point.jacobian[0], point.transport_jacobian])
                    check(label + '/parameters', np.max(abs(np.array(numerical).T - analytic) / (1 + abs(analytic))), 3e-5)
                    numerical = []
                    for name in point.input_names:
                        plus, minus = SimpleNamespace(**vars(row)), SimpleNamespace(**vars(row))
                        step = .2 if name == 'growth_T_C' else .001
                        if name == 'growth_T_C':
                            plus.growth_T_C += step; minus.growth_T_C -= step
                        else:
                            key = {'log10_dose_cm3': 'incorporated_cm3', 'log10_Tobs_K': 'Tobs_K', 'log10_H_cm3': 'H_cm3'}[name]
                            setattr(plus, key, getattr(plus, key) * 10 ** step)
                            setattr(minus, key, getattr(minus, key) * 10 ** (-step))
                        numerical.append((vector(model.point(plus, w, source_kind='share', source_shift=.1, require_observation=False)) -
                            vector(model.point(minus, w, source_kind='share', source_shift=.1, require_observation=False))) / (2 * step))
                    analytic = np.vstack([point.input_jacobian[0], point.transport_input_jacobian])
                    check(label + '/inputs', np.max(abs(np.array(numerical).T - analytic) / (1 + abs(analytic))), 3e-5)
                    for kind in ('share', 'offset'):
                        mid = model.point(row, w, source_kind=kind, source_shift=.1, require_observation=False)
                        plus = model.point(row, w, source_kind=kind, source_shift=.101, require_observation=False)
                        minus = model.point(row, w, source_kind=kind, source_shift=.099, require_observation=False)
                        analytic = np.r_[mid.source_jacobian[0], mid.transport_source_jacobian]
                        check(label + '/' + kind + '/source', np.max(abs((vector(plus) - vector(minus)) / .002 - analytic) / (1 + abs(analytic))), 3e-5)
                    point = model.point(row, w, source_kind='share', source_shift=.1, require_observation=False)
                    dg = point.diagnostics
                    check(label + '/foreign', dg['foreign_atom_relative_error'], 3e-8)
                    check(label + '/foreign_derivative', dg['foreign_atom_derivative_error'], 3e-7)
                    check(label + '/charge_residual', dg['charge_relative_residual'], 2e-8)
                    check(label + '/derived_sigma', abs(point.transport_values[1] - np.log10(transport_e()) - point.values[0] - point.transport_values[0]), 2e-10)
                    if readout == 'physical':
                        check(label + '/free_floor', max(0., 1 - dg['free_n_cm3'] / 1e13), 1e-6)
                    else:
                        check(label + '/envelope', max(0., dg['regression_lower_log10_nH'] - point.values[0], point.values[0] - dg['endpoint_high_log10_nH']), 1e-12)
                    states.append(dict(case=label, log_nH=float(point.values[0]), log_mu=float(point.transport_values[0]), valid=dg['observation_domain_valid'], critical=dg['critical_free_fraction']))
                except Exception as exc:
                    errors.append(dict(case=label, error=type(exc).__name__ + ': ' + str(exc)))
                    check(label + '/exception', 1., 0.)
    model = ChargeFeasibleModel(basis='process', profile=profile)
    for i, row in enumerate(cases):
        for psi in protocol['saturation_logits']:
            try:
                a = model.point(row, [.5, 16., psi, 0., 0., 0.], require_observation=False)
                dg = a.diagnostics
                check(f'saturation/{i}/{psi}/free_floor', max(0., 1 - dg['free_n_cm3'] / 1e13), 1e-6)
                check(f'saturation/{i}/{psi}/finite', 0. if np.isfinite(vector(a)).all() else 1., 0.)
                check(f'saturation/{i}/{psi}/foreign', dg['foreign_atom_relative_error'], 3e-8)
            except Exception as exc:
                errors.append(dict(case=f'saturation/{i}/{psi}', error=type(exc).__name__ + ': ' + str(exc)))
                check(f'saturation/{i}/{psi}/exception', 1., 0.)
    for i, row in enumerate(cases):
        layer, solver = model.engine_for(row).layer_for(row)
        pars = layer.parameters(T=1330., Nd=1e16, weights=[0.],
            doses=(row.incorporated_cm3, row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm, Tobs=row.Tobs_K, binding_eV=.0366, screen_s=0.)
        chain = np.eye(layer.np)
        a = free_charge_threshold(solver, pars, chain)
        columns = [0, 1, layer.idose, layer.iobs, *range(layer.imat.start, layer.imat.stop)]
        if row.H_known: columns.append(2 + layer.b + 1)
        for k in columns:
            delta = np.zeros(layer.np); delta[k] = .001
            plus = free_charge_threshold(solver, pars + delta, chain)
            minus = free_charge_threshold(solver, pars - delta, chain)
            numerical = (plus.fraction - minus.fraction) / .002
            check(f'threshold/{i}/raw{k}', abs(numerical - a.jacobian[k]) / (1 + abs(a.jacobian[k])), 3e-5)
    rows = {r.sample_id: r for r in table.itertuples()}
    recovered, models = [], {}
    for case in diagnostic_protocol['cases']:
        key = case['fit_key']
        if key not in models:
            fit = json.loads((PARENT / 'fits' / (key + '.json')).read_text())
            models[key] = (fit, ChargeFeasibleModel(basis='process', profile=profile))
        fit, model = models[key]
        try:
            a = model.point(rows[case['sample_id']], fit['parameters'], require_observation=False)
            dg = a.diagnostics
            record = dict(case, numeric=True, valid=dg['observation_domain_valid'], log10_nH=float(a.values[0]),
                free_n_cm3=dg['free_n_cm3'], critical_fraction=dg['critical_free_fraction'], fraction=dg['isolated_atom_fraction_after'])
            check('old_failed_state/' + key + '/' + case['sample_id'], max(0., 1 - dg['free_n_cm3'] / 1e13), 1e-6)
        except Exception as exc:
            record = dict(case, numeric=False, valid=False, error=type(exc).__name__ + ': ' + str(exc))
            errors.append(dict(case=case, error=record['error']))
            check('old_failed_state/' + key + '/' + case['sample_id'] + '/exception', 1., 0.)
        recovered.append(record)
    e.csvwrite(OUT / 'gate_states.csv', states)
    e.csvwrite(OUT / 'old_failed_state_probe.csv', recovered)
    result = dict(passed=all(c['passed'] for c in checks), created_utc=e.now(), checks=len(checks), rows=checks, errors=errors,
        state_cases=len(states), old_failed_states=len(recovered), recovered_numeric=sum(r['numeric'] for r in recovered),
        recovered_valid=sum(r['valid'] for r in recovered), recovered_states_not_new_accuracy=True,
        maximum_scaled_derivative_error=max(c['value'] for c in checks if c['name'].endswith(('/parameters', '/inputs', '/source')) or c['name'].startswith('threshold/')),
        no_new_training_or_measurements=True, free_floor_does_not_certify_Hall_or_accuracy=True,
        code_hashes=protocol['code_hashes'], protocol_sha256=e.sha(OUT / 'gate_protocol_before.json'), goal_complete=False)
    e.write(OUT / 'numerical_gates.json', result)
    print('Phase27 predictive feasibility gates', result['passed'], len(checks), 'max derivative', result['maximum_scaled_derivative_error'],
        'old failed numeric/valid', result['recovered_numeric'], result['recovered_valid'], flush=True)
    if not result['passed']:
        raise SystemExit(1)


def transport_e():
    from sandbox.transport import E
    return E


if __name__ == '__main__':
    main()
