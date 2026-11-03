"""Qualify unchanged formulas with the diagnosed two-width derivative probe."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

OUT = e.BASE / 'phase27_charge_feasible'


def main():
    assert not (OUT / 'numerical_gates_v2.json').exists(), 'Never overwrite a completed gate'
    first = json.loads((OUT / 'numerical_gates.json').read_text())
    diagnosis = json.loads((OUT / 'gradient_diagnosis.json').read_text())
    assert not first['passed'] and diagnosis['passed'] and diagnosis['derivative_formula_unchanged']
    for packet in (first, diagnosis):
        for path, h in packet['code_hashes'].items():
            assert e.sha(ROOT / path) == h, path
    archive = OUT / 'gate_failure_v1'
    files = [OUT / 'gate_protocol_before.json', OUT / 'numerical_gates.json', OUT / 'gate_states.csv',
             OUT / 'old_failed_state_probe.csv', *[ROOT / p for p in first['code_hashes']]]
    for path in files:
        target = archive / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(path, target)
    shutil.copyfile('/tmp/ga2o3_phase27_gates_v1.log', archive / 'execution.log')
    replaced = {f'threshold/{i}/raw0' for i in range(4)}
    rows = [r for r in first['rows'] if r['name'] not in replaced]
    differences = read(OUT / 'threshold_temperature_richardson.csv')
    for r in differences.itertuples():
        rows.append(dict(name=f'threshold/{int(r.case)}/raw0/Richardson/{r.coarse_step}',
                         value=float(r.scaled_error), tolerance=3e-5, passed=bool(r.passed)))
    assert len(rows) == 534
    result = dict(first, passed=all(r['passed'] for r in rows), created_utc=e.now(), rows=rows, checks=len(rows),
        replacement_scope='all four raw-logTf probes use both declared Richardson pairs and their agreement;other original checks inherited unchanged',
        original_failure_preserved=True, formulas_changed=False, tolerance_changed=False,
        maximum_scaled_derivative_error=max(r['value'] for r in rows if r['name'].endswith(('/parameters', '/inputs', '/source')) or r['name'].startswith('threshold/')),
        original_gate_sha256=e.sha(OUT / 'numerical_gates.json'), diagnosis_sha256=e.sha(OUT / 'gradient_diagnosis.json'),
        code_hashes=dict(first['code_hashes'], **{str(Path(__file__).relative_to(ROOT)): e.sha(Path(__file__))}))
    e.write(OUT / 'numerical_gates_v2.json', result)
    print('Phase27 unchanged-model gateV2', result['passed'], len(rows), result['maximum_scaled_derivative_error'], flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
