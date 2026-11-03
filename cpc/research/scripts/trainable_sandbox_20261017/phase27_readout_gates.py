"""Versioned gate for the envelope offset readout and saved evaluation contract."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel
from sandbox.charge_feasible_model import ChargeFeasibleModel as Draft
from sandbox.charge_feasible_saved_model import SavedFeasibleModel
from sandbox.coherent_fermi import published_coherent_profile
from sandbox import transport
import phase2_energy as e

OUT = e.BASE / 'phase27_charge_feasible'


def vec(p):
    return np.r_[p.values[0], p.transport_values]


def main():
    assert not (OUT / 'readout_gates.json').exists()
    previous = json.loads((OUT / 'numerical_gates_v2.json').read_text()); assert previous['passed']
    proto = json.loads((OUT / 'gate_protocol_before.json').read_text())
    code = [Path(__file__), ROOT / 'src/sandbox/charge_feasible_model_v2.py', ROOT / 'src/sandbox/charge_feasible_saved_model.py',
        ROOT / 'src/sandbox/charge_feasible_model.py', ROOT / 'src/sandbox/charge_feasible_share.py']
    e.write(OUT / 'readout_protocol_before.json', dict(created_utc=e.now(),
        purpose='CT updated output-offset and saved adapter;shared zero-source/physical map unchanged',
        cases=proto['cases'], bases=proto['bases'], source_shift=.1, difference_step=.001,
        scaled_derivative_tolerance=3e-5, replay_tolerance=2e-10, no_training=True,
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in code},
        previous_gate_sha256=e.sha(OUT / 'numerical_gates_v2.json')))
    checks = []
    def check(name, error, tol):
        checks.append(dict(name=name, error=float(error), tolerance=tol, passed=bool(error <= tol)))
    for basis in proto['bases']:
        for readout, prefix in [('physical', 'Feasible'), ('endpoint_envelope', 'Envelope')]:
            new = ChargeFeasibleModel(basis=basis, readout=readout, profile=published_coherent_profile())
            old = Draft(basis=basis, readout=readout, profile=published_coherent_profile())
            w = np.r_[.5, 16., np.array([2.4, .4, -.2, .1])[:len(new.parameter_names) - 2]]
            saved = SavedFeasibleModel(dict(method=prefix + basis.title(), parameters=w.tolist(), training_valid=True, selected_success=True))
            for i, inputs in enumerate(proto['cases']):
                row = SimpleNamespace(**inputs); label = f'{prefix}/{basis}/{i}'
                for kind in ('none', 'share'):
                    z = 0. if kind == 'none' else .1
                    a = new.point(row, w, source_kind=kind, source_shift=z, require_observation=False)
                    b = old.point(row, w, source_kind=kind, source_shift=z, require_observation=False)
                    check(label + '/' + kind + '/unchanged_value', max(abs(vec(a) - vec(b))), 0.)
                    check(label + '/' + kind + '/unchanged_derivative', max(abs(a.jacobian[0] - b.jacobian[0])), 0.)
                a = new.point(row, w, source_kind='offset', source_shift=.1, require_observation=False)
                p = new.point(row, w, source_kind='offset', source_shift=.101, require_observation=False)
                m = new.point(row, w, source_kind='offset', source_shift=.099, require_observation=False)
                analytic = np.r_[a.source_jacobian[0], a.transport_source_jacobian]
                check(label + '/offset_source', max(abs((vec(p) - vec(m)) / .002 - analytic) / (1 + abs(analytic))), 3e-5)
                if readout == 'endpoint_envelope':
                    mu = np.log10(transport.mu_caughey_thomas(10 ** a.values[0], 300.))
                    check(label + '/offset_CT', abs(a.transport_values[0] - mu), 2e-12)
                    numerical = []
                    for k in range(len(w)):
                        step = np.zeros(len(w)); step[k] = .001
                        p = new.point(row, w + step, source_kind='offset', source_shift=.1, require_observation=False)
                        m = new.point(row, w - step, source_kind='offset', source_shift=.1, require_observation=False)
                        numerical.append((vec(p) - vec(m)) / .002)
                    analytic = np.vstack([a.jacobian[0], a.transport_jacobian])
                    check(label + '/offset_parameters', np.max(abs(np.array(numerical).T - analytic) / (1 + abs(analytic))), 3e-5)
                for kind in ('field', 'offset'):
                    p = saved.point(row, source_kind=kind, shift=.1, require_observation=False)
                    core = new.point(row, w, source_kind='share' if kind == 'field' else kind, source_shift=.1, require_observation=False)
                    check(label + '/' + kind + '/saved', max(abs(np.r_[p.log_nH, p.log_muH, p.log_sigma] - vec(core))), 2e-10)
                base = saved.point(row, require_observation=False)
                fast = saved.offset_point(row, base, .1, require_observation=False)
                slow = saved.point(row, source_kind='offset', shift=.1, require_observation=False)
                check(label + '/offset_fast', max(abs(np.r_[fast.log_nH, fast.log_muH, fast.log_sigma] - np.r_[slow.log_nH, slow.log_muH, slow.log_sigma])), 2e-10)
    result = dict(passed=all(c['passed'] for c in checks), checks=len(checks), rows=checks,
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in code},
        CT_after_envelope_offset=True, zero_source_training_unchanged=True, physical_map_unchanged=True,
        old_gate_failure_preserved=True, no_new_training_or_material_measurements=True, goal_complete=False)
    e.write(OUT / 'readout_gates.json', result)
    print('Phase27 readout and saved model gates', result['passed'], len(checks), flush=True)
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
