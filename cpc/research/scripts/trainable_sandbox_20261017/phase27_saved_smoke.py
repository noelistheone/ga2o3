"""Replay exact completed partial-training snapshots without any new fit."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from sandbox.charge_feasible_saved_model import SavedFeasibleModel
from sandbox.context_saved_model import INPUT_NAMES
import phase27_training as t
import phase2_energy as e


def main():
    out = t.OUT / 'saved_model_smoke'
    assert not (out / 'audit.json').exists()
    context = t.parent.contexts('DOI')[0]
    specs = [(m, .1) for m in t.METHODS] + [('FeasibleProcess', None), ('EnvelopeProcess', None)]
    snapshots = []
    for method, reg in specs:
        _, path = t.path_for('DOI', context, method, reg)
        assert path.exists(), path
        r = json.loads(path.read_text())
        assert r['training_valid'] and r['selected_success'], (method, reg)
        snapshots.append(dict(method=method, reg=reg, fit_path=str(path.relative_to(ROOT)), fit_sha256=e.sha(path)))
    e.write(out / 'protocol_before.json', dict(created_utc=e.now(), snapshots=snapshots,
        purpose='saved values/flags and dimension replay on training inputs;not generalization',
        numerical_tolerance_dex=2e-6, snapshot_selection='first declared DOI4-unitcontext,gamma.1for6families+2explicitfallbacks',
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in [Path(__file__), ROOT / 'src/sandbox/charge_feasible_saved_model.py']}))
    checks, points = [], 0
    for spec in snapshots:
        fit = json.loads((ROOT / spec['fit_path']).read_text())
        model = SavedFeasibleModel(fit)
        table = t.parent.subset(fit['scheme'], fit['train_units'])
        for i, row in enumerate(table.itertuples()):
            inputs = SimpleNamespace(**{k: getattr(row, k) for k in INPUT_NAMES})
            p = model.point(inputs, require_observation=False)
            error = abs(p.log_nH - fit['selected_training_values'][i]); points += 1
            checks.append(dict(name=fit['key'] + '/' + row.sample_id + '/nH', error=error, tolerance=2e-6, passed=bool(error <= 2e-6)))
            checks.append(dict(name=fit['key'] + '/' + row.sample_id + '/flag', error=float(p.valid != fit['selected_training_valid'][i]), tolerance=0., passed=bool(p.valid == fit['selected_training_valid'][i])))
        checks.append(dict(name=fit['key'] + '/actual_dimension', error=float(len(model.operator.parameter_names) != len(fit['parameters'])), tolerance=0., passed=bool(len(model.operator.parameter_names) == len(fit['parameters']))))
    bad = dict(fit, training_valid=False)
    try: SavedFeasibleModel(bad)
    except ValueError: checks.append(dict(name='invalid_fixture_refused', error=0., tolerance=0., passed=True))
    else: checks.append(dict(name='invalid_fixture_refused', error=1., tolerance=0., passed=False))
    result = dict(passed=all(c['passed'] for c in checks), checks=len(checks), rows=checks, fit_snapshots=snapshots,
        replayed_points=points, actual_partial_training_models=True, new_fits=0,
        grade='software snapshot replay;not held-source accuracy or full deployment', goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in [Path(__file__), ROOT / 'src/sandbox/charge_feasible_saved_model.py']})
    e.write(out / 'audit.json', result)
    print('Phase27 actual saved snapshots', result['passed'], len(checks), points, flush=True)
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
