"""Two-width convergence probe for the actual raw-temperature gate failures."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from sandbox.charge_feasible_model import ChargeFeasibleModel
from sandbox.charge_feasible_share import free_charge_threshold
from sandbox.coherent_fermi import published_coherent_profile
import phase2_energy as e

OUT = e.BASE / 'phase27_charge_feasible'


def main():
    assert not (OUT / 'gradient_diagnosis.json').exists(), 'Preserve the first diagnosis'
    gate = json.loads((OUT / 'numerical_gates.json').read_text())
    assert not gate['passed']
    failed = [r['name'] for r in gate['rows'] if not r['passed']]
    assert failed == ['threshold/0/raw0', 'threshold/2/raw0']
    protocol = json.loads((OUT / 'gate_protocol_before.json').read_text())
    code = [Path(__file__), ROOT / 'src/sandbox/charge_feasible_share.py', ROOT / 'src/sandbox/charge_feasible_model.py']
    steps = [.01, .003, .002, .001, .0005, .0003, .0001, .00003, .00001]
    e.write(OUT / 'gradient_diagnosis_protocol_before.json', dict(created_utc=e.now(),
        question='raw logTf central-difference truncation versus incorrect analytic chain',
        failed_names=failed, cases='all four original gate inputs,not only failing two', steps=steps,
        richardson_width_pairs=[[.001, .0005], [.002, .001]],
        original_tolerance=3e-5, no_new_physics_or_training=True,
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in code},
        input_hashes={str((OUT / p).relative_to(ROOT)): e.sha(OUT / p) for p in ['numerical_gates.json', 'gate_protocol_before.json']}))
    model = ChargeFeasibleModel(basis='process', profile=published_coherent_profile())
    rows, richardson = [], []
    for i, values in enumerate(protocol['cases']):
        row = SimpleNamespace(**values)
        layer, solver = model.engine_for(row).layer_for(row)
        pars = layer.parameters(T=1330., Nd=1e16, weights=[0.],
            doses=(row.incorporated_cm3, row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm, Tobs=row.Tobs_K, binding_eV=.0366, screen_s=0.)
        chain = np.eye(layer.np)
        a = free_charge_threshold(solver, pars, chain)
        derivatives = {}
        for h in steps:
            step = np.zeros(layer.np); step[0] = h
            plus = free_charge_threshold(solver, pars + step, chain)
            minus = free_charge_threshold(solver, pars - step, chain)
            derivatives[h] = (plus.fraction - minus.fraction) / (2 * h)
            rows.append(dict(case=i, step=h, analytic=float(a.jacobian[0]), numerical=derivatives[h],
                scaled_error=abs(derivatives[h] - a.jacobian[0]) / (1 + abs(a.jacobian[0]))))
        values = []
        for coarse, fine in [(.001, .0005), (.002, .001)]:
            d = (4 * derivatives[fine] - derivatives[coarse]) / 3
            error = abs(d - a.jacobian[0]) / (1 + abs(a.jacobian[0]))
            values.append(d)
            richardson.append(dict(case=i, coarse_step=coarse, fine_step=fine, analytic=float(a.jacobian[0]),
                numerical=d, scaled_error=error, passed=bool(error <= 3e-5)))
        richardson.append(dict(case=i, coarse_step=0., fine_step=0., analytic=float(a.jacobian[0]),
            numerical=float(abs(values[0] - values[1])),
            scaled_error=abs(values[0] - values[1]) / (1 + abs(a.jacobian[0])),
            passed=bool(abs(values[0] - values[1]) / (1 + abs(a.jacobian[0])) <= 3e-5)))
    e.csvwrite(OUT / 'threshold_temperature_step_convergence.csv', rows)
    e.csvwrite(OUT / 'threshold_temperature_richardson.csv', richardson)
    result = dict(passed=all(r['passed'] for r in richardson), created_utc=e.now(),
        code_unchanged=True, tolerance_unchanged=3e-5, derivative_formula_unchanged=True,
        original_failures_preserved=True, two_independent_width_pairs=True,
        maximum_richardson_scaled_error=max(r['scaled_error'] for r in richardson),
        case_rows=len(rows), richardson_checks=len(richardson),
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in code},
        protocol_sha256=e.sha(OUT / 'gradient_diagnosis_protocol_before.json'), goal_complete=False)
    e.write(OUT / 'gradient_diagnosis.json', result)
    print('Phase27 actual raw-temperature difference diagnosis', result['passed'], result['maximum_richardson_scaled_error'], flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
