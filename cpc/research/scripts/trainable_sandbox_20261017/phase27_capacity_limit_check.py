"""Independent charge-difference identity in24 fixed-parameter physical states."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel
from sandbox.charge_feasible_share import _cold_floor
from sandbox.coherent_fermi import published_coherent_profile
import phase2_energy as e

OUT = e.BASE / 'phase27_charge_feasible/capacity_limit'


def main():
    assert not (OUT / 'audit.json').exists()
    parent = OUT.parent
    original = json.loads((parent / 'gate_protocol_before.json').read_text())
    code = [Path(__file__), ROOT / 'src/sandbox/charge_feasible_share.py', ROOT / 'src/sandbox/charge_feasible_model_v2.py',
        ROOT / 'src/sandbox/charge_feasible_model.py', ROOT / 'docs/ML_SANDBOX_CHARGE_CAPACITY_LIMIT_20261018.md']
    e.write(OUT / 'protocol_before.json', dict(created_utc=e.now(),
        purpose='charge capacity and saturation interpretation;not additional learned model or accuracy',
        inputs=original['cases'], bases=['constant', 'dose', 'process'], constant_coefficients=[-.7, 1.4],
        other_coefficients=[.4, -.2, .1], reference_parameters=[.5, 16.], free_floor_cm3=1e13,
        expected_states=24, relative_tolerance=3e-8, no_training_or_labels=True,
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in code},
        main_protocol_sha256=e.sha(parent / 'pre_evaluation_protocol.json'),
        original_gate_protocol_sha256=e.sha(parent / 'gate_protocol_before.json'), goal_complete=False))
    rows, checks = [], []
    def check(name, error):
        checks.append(dict(name=name, error=float(error), tolerance=3e-8, passed=bool(error <= 3e-8)))
    for basis in ('constant', 'dose', 'process'):
        model = ChargeFeasibleModel(basis=basis, profile=published_coherent_profile())
        for intercept in (-.7, 1.4):
            n = len(model.parameter_names)
            w = np.r_[.5, 16., np.array([intercept, .4, -.2, .1])[:n - 2]]
            for i, inputs in enumerate(original['cases']):
                row = SimpleNamespace(**inputs)
                point = model.point(row, w, require_observation=False); dg = point.diagnostics
                layer, solver = model.engine_for(row).layer_for(row)
                inventory, cold = solver.last_inventory, solver.last_prediction
                pars = layer.parameters(T=dg['Tf_K'], Nd=1e16, weights=[0.],
                    doses=(row.incorporated_cm3, row.H_cm3) if row.H_known else (row.incorporated_cm3,),
                    pO2=row.scenario_pO2_atm, Tobs=row.Tobs_K, binding_eV=.0366, screen_s=0.)
                mean_star, _, car_star, _, _ = _cold_floor(layer, pars, free_floor_cm3=1e13)
                mean_root = np.bincount(layer.parent, weights=cold.probabilities * layer.q, minlength=layer.db.nD)
                car_root = model.profile.carriers(cold.ef_eV, row.Tobs_K)[0]
                f = dg['isolated_atom_fraction_after']
                Qstar = dg['threshold_charge_N_cm3'] + dg['threshold_charge_D_cm3'] * f
                direct = 1e16 + inventory.totals @ mean_star + car_star[1:].sum() - car_star[0]
                excess = float(car_root[0] - 1e13)
                loss = float(inventory.totals @ (mean_star - mean_root) + car_star[1:].sum() - car_root[1:].sum())
                scale = max(abs(Qstar), abs(direct), abs(car_root[0]), abs(loss), 1e13)
                label = f'{basis}/{intercept}/{i}'
                check(label + '/affine_charge', abs(Qstar - direct) / scale)
                check(label + '/difference_identity', abs(Qstar - excess - loss) / scale)
                check(label + '/positive_charge_loss', max(0., -loss) / scale)
                check(label + '/capacity_upper_bound', max(0., excess - Qstar) / scale)
                check(label + '/positive_excess', max(0., -excess) / scale)
                rows.append(dict(case=label, basis=basis, constant=intercept, input_case=i,
                    implemented_valid=dg['observation_domain_valid'], threshold_fraction=dg['critical_free_fraction'],
                    isolated_fraction=f, Qstar_cm3=float(Qstar), free_excess_cm3=excess,
                    charge_repartition_loss_cm3=loss, excess_to_capacity=float(excess / Qstar),
                    relative_charge_repartition=float(loss / Qstar), this_is_not_measured_activation=True))
    assert len(rows) == 24
    e.csvwrite(OUT / 'fixed_states.csv', rows)
    value = dict(passed=all(c['passed'] for c in checks), checks=len(checks), rows=checks, states=len(rows),
        maximum_scaled_identity_error=max(c['error'] for c in checks if c['name'].endswith('/difference_identity')),
        minimum_excess_to_capacity=min(r['excess_to_capacity'] for r in rows),
        maximum_excess_to_capacity=max(r['excess_to_capacity'] for r in rows),
        near_saturation_is_conditional_not_all_materials=True, new_fits=0, new_measurements=0,
        no_actual_heldout_accuracy_claim=True, main_method_or_protocol_changed=False,
        protocol_sha256=e.sha(OUT / 'protocol_before.json'),
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in code}, goal_complete=False)
    e.write(OUT / 'audit.json', value)
    print('Actual24-state charge capacity check', value['passed'], len(checks), 'identity', value['maximum_scaled_identity_error'],
        'excess/capacity', value['minimum_excess_to_capacity'], value['maximum_excess_to_capacity'], flush=True)
    if not value['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
