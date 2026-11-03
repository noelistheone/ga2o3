"""Bind all primary choices and prediction/reporting operations before outer scores."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from sandbox.kroger_db import NPZ
import phase27_prediction as p
import phase2_energy as e

OUT = p.t.OUT


def write_once(name, value):
    path = OUT / name
    assert not path.exists(), 'Never refresh a frozen protocol;retain a failed version first'
    e.write(path, value)


def main():
    assert not (OUT / 'prediction/execution_started.json').exists(), 'Declare before outer predictions'
    gates = ('numerical_gates_v2.json', 'readout_gates.json', 'learning_gates.json', 'source_gates.json',
        'saved_model_smoke/audit.json', 'source_smoke/audit.json', 'interface_smoke/audit.json')
    for name in gates:
        data = json.loads((OUT / name).read_text()); assert data['passed'], name
        for path, h in data['code_hashes'].items(): assert e.sha(ROOT / path) == h, path
    folder = Path(__file__).parent
    scripts = ['phase27_prediction.py', 'phase27_registry.py', 'phase27_source_runtime.py',
        'phase27_training_audit.py', 'phase27_prediction_audit.py', 'phase27_export.py',
        'phase27_api_runtime.py', 'phase27_cohort_replay.py', 'phase27_external.py',
        'phase27_external_audit.py', 'phase27_analysis.py', 'phase27_physical_scope.py',
        'phase27_delivery_final_check.py', 'phase27_information_audit.py', 'phase27_after_training.py',
        'phase27_verify_parents.py', 'predict_feasible_properties_v1.py']
    code = [Path(__file__), *[folder / name for name in scripts],
        ROOT / 'src/sandbox/charge_feasible_saved_model.py', ROOT / 'src/sandbox/charge_feasible_model_v2.py',
        ROOT / 'src/sandbox/charge_feasible_model.py', ROOT / 'src/sandbox/charge_feasible_share.py',
        ROOT / 'src/sandbox/feasible_property_predictor.py', ROOT / 'src/sandbox/feasible_property_predictor_v2.py', ROOT / 'src/sandbox/pool_property_predictor.py',
        ROOT / 'src/sandbox/context_source.py', ROOT / 'src/sandbox/functional_source.py',
        ROOT / 'docs/ML_SANDBOX_CHARGE_FEASIBLE_PROTOCOL_20261018.md']
    inputs = [OUT / 'training_protocol.json', *[OUT / name for name in gates],
        p.PARENT / 'prediction/combined_predictions.csv', p.PARENT / 'prediction/combined_summary.csv',
        p.DATA_PARENT / 'data/conditions33.csv', p.DATA_PARENT / 'data/episodes33.csv', NPZ,
        e.BASE / 'artifact_manifest_phase26.json']
    hashes = {str(path.relative_to(ROOT)): e.sha(path) for path in [*code, *inputs]}
    write_once('pre_evaluation_protocol.json', dict(created_utc=e.now(), primary='FeasibleProcess_internal',
        variants=p.VARIANTS, new_variants=12, old_variants=36, combined_variants=48,
        new_rows=2376, combined_rows=9504, new_shared_choices=78, new_plans=156,
        new_source_selections=468, new_summary_rows=144, combined_summary_rows=576,
        source_lambdas=p.LAM, selection='corresponding training-inner labels only;no outerquery selection',
        point_budget=32768, source_search='same33cubicnodes,atmost3searches,80iterations;sampledzero-adjacentbranch',
        qualification='invalid/unconverged models refused;nonfinite inner scores retained;no partial-coverage cohort MAE',
        cohort='unchanged33/6DOI/5groups,old29roles/common18 retained,common20fixed',
        prior='conditional electron-conducting domain;free floor does not ensure Hall floor or material conductivity',
        matched_control='sameparametercount endpoint-Hall interval control;energy information disclosed separately',
        new_mu_supervision=False, independent_sigma_labels=0, retrospective=True, goal_complete=False, hashes=hashes))
    post_files = [folder / name for name in scripts if name not in ['phase27_prediction.py', 'phase27_registry.py', 'phase27_source_runtime.py']]
    write_once('post_prediction_predeclared.json', dict(created_utc=e.now(),
        primary='FeasibleProcess_internal', API_requests=1728, API_point_rows=9504,
        presets=624, expected_models=390, external_requests=720, external_point_rows=5760,
        external_summary_rows=1440, no_source_hyperparameter_or_shared_refit_on_external=True,
        external_scope='same12AzizieHallconditions;A2roundedchemistry,B12processproxies overlap;not14samples or blind/lab proof',
        code_hashes={str(path.relative_to(ROOT)): e.sha(path) for path in post_files},
        input_hashes={str((OUT / 'pre_evaluation_protocol.json').relative_to(ROOT)): e.sha(OUT / 'pre_evaluation_protocol.json')},
        goal_complete=False))
    write_once('analysis_predeclared.json', dict(created_utc=e.now(), primary='FeasibleProcess_internal',
        all_methods=48, metric_rows=576, primary_comparisons=282,
        controls='every other47,not a selected weak subset;including same-parameter endpoint regression',
        metrics='DOI/groupmacro,worstDOI,perDOIbetter/tied/worse,leaveonegroup;all0/1/2 and fixedcommon20',
        scope='conditional retrospective Si/Sn;no significance/globalnovelty/independent lab claim',
        physical_scope='3newFeasiblefamilies x2fullcontexts x33knownconditions;no inferred midpoint forEnvelope',
        uncertainty='no calibrated material predictive intervals from solver tests',
        stop_rule='if no increment,end this allocation-parameter expansion;use independent paired information or a documented physical-reference risk',
        code_hashes={str((folder / name).relative_to(ROOT)): e.sha(folder / name) for name in
            ['phase27_analysis.py', 'phase27_physical_scope.py', 'phase27_information_audit.py']},
        input_hashes={str((OUT / name).relative_to(ROOT)): e.sha(OUT / name) for name in
            ['pre_evaluation_protocol.json', 'post_prediction_predeclared.json']}, goal_complete=False))
    print('Frozen phase27 primary48method design', len(hashes), 'dependencies;no outer scores', flush=True)


if __name__ == '__main__': main()
