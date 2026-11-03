"""Verify the actual phase27 predictive deliverable after full result review."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from phase27_report import AUDITS, PRIMARY, VARIANTS
import phase2_energy as e

OUT = e.BASE / 'phase27_charge_feasible'
PROTOCOLS = ('training_protocol.json', 'pre_evaluation_protocol.json',
    'post_prediction_predeclared.json', 'prediction/protocol.json',
    'delivery/cohort_replay/protocol.json', 'external_Azizie/protocol_before_transfer.json',
    'analysis_predeclared.json', 'input_scope_annotations/protocol_before.json',
    'capacity_limit/protocol_before.json', 'prediction_audit_resume_protocol_v2.json',
    'independent_input_audit/lowdose_transfer/protocol_before_transfer.json',
    'independent_input_audit/lowdose_transfer/audit_protocol_v2.json')
HASH_MAPS = ('code_hashes', 'input_hashes', 'hashes', 'inherited_parent_fit_sha256', 'files')


def verify_hash_maps(value, label):
    for kind in HASH_MAPS:
        values = value.get(kind, {})
        assert isinstance(values, dict), (label, kind)
        for name, digest in values.items():
            assert e.sha(ROOT / name) == digest, (label, kind, name)


def verify_claim(claim, cache):
    path = ROOT / claim['path']
    assert e.sha(path) == claim['sha256'], path
    if path not in cache: cache[path] = read(path)
    frame = cache[path]
    for key, value in claim['selector'].items(): frame = frame[frame[key].eq(value)]
    assert len(frame) == 1, (path, claim['selector'])
    actual = float(frame[claim['column']].iloc[0]); expected = claim['value']
    assert not np.isfinite(actual) if expected is None else actual == expected, (path, actual, expected)


def main():
    audits = {}
    for name in (*AUDITS, 'capacity_limit/audit.json', 'independent_input_audit/lowdose_transfer/audit.json'):
        value = json.loads((OUT / name).read_text()); assert value['passed'], name
        verify_hash_maps(value, name); audits[name] = value
    for name in PROTOCOLS:
        verify_hash_maps(json.loads((OUT / name).read_text()), name)
    for name in ('pipeline_complete.json', 'training_complete.json', 'delivery/export_complete.json',
        'external_Azizie/execution_complete.json'):
        assert (OUT / name).is_file(), name
    pipeline = json.loads((OUT / 'pipeline_complete.json').read_text())
    expected = ['phase27_training_audit.py', 'phase27_prediction.py', 'phase27_prediction_audit_v2.py',
        'phase27_export.py', 'phase27_cohort_replay.py', 'phase27_external.py', 'phase27_external_audit.py',
        'phase27_analysis.py', 'phase27_physical_scope.py', 'phase27_delivery_final_check.py', 'phase27_information_audit.py']
    assert [r['script'] for r in pipeline['successful_events']] == expected
    assert len(pipeline['events']) == 12 and len(pipeline['failed_events']) == 1
    assert pipeline['failed_events'][0]['script'] == 'phase27_prediction_audit.py'
    assert pipeline['failed_events'][0]['exit_code'] != 0
    assert pipeline['original_failed_audit_preserved'] and pipeline['original_training_and_predictions_reused_unchanged']
    assert pipeline['primary'] == PRIMARY
    design = json.loads((OUT / 'report_design_before_results.json').read_text())
    assert design['source_code_sha256'] == e.sha(ROOT / 'scripts/trainable_sandbox_20261017/phase27_report.py')
    assert design['method_order'] == list(VARIANTS) and design['primary'] == PRIMARY
    assert e.sha(ROOT / design['theory_path']) == design['theory_sha256']
    assert not json.loads((OUT / 'numerical_gates.json').read_text())['passed']
    assert (OUT / 'gate_failure_v1').is_dir() and (OUT / 'predeclare_failure_v1').is_dir()
    failed = json.loads((OUT / 'prediction_audit_failure_v1/audit.json').read_text())
    assert not failed['passed'] and sum(not r['passed'] for r in failed['rows']) == 84
    assert audits['prediction/audit.json']['audit_version'] == 2
    assert audits['prediction/audit.json']['no_predictions_models_labels_or_scores_changed']
    assert audits['carbon_scope/audit.json']['checks'] == 46
    assert audits['input_scope_annotations/audit.json']['passed']
    assert audits['capacity_limit/audit.json']['states'] == 24
    assert audits['independent_input_audit/lowdose_transfer/audit.json']['physical_conditions'] == 1

    report = OUT / 'ML与Sandbox_电荷可行学习与性质预测.md'; pdf = report.with_suffix('.pdf')
    claims = json.loads((OUT / 'report_claims.json').read_text())
    assert claims['report_sha256'] == e.sha(report)
    assert e.sha(ROOT / claims['theory_path']) == claims['theory_sha256']
    assert claims['all48_methods_reported'] and not claims['outer_method_reselected']
    assert claims['source_code_sha256'] == design['source_code_sha256']
    cache = {}
    for claim in claims['claims']: verify_claim(claim, cache)
    for name, digest in claims['json_audit_sources'].items(): assert e.sha(ROOT / name) == digest, name
    conclusion = json.loads((OUT / 'analysis/conclusion_receipt.json').read_text())
    assert conclusion['notes_sha256'] == e.sha(OUT / 'analysis/predictive_conclusions.md')
    assert conclusion['facts'] and conclusion['fixed_primary'] == PRIMARY and not conclusion['outer_method_reselected']
    assert conclusion['next_predictive_work'], 'Actual evidence must guide the next research choice'
    for claim in conclusion['facts']: verify_claim(claim, cache)
    for name, digest in conclusion.get('supplemental_json_sources', {}).items(): assert e.sha(ROOT / name) == digest, name
    builds = list(OUT.glob('_pdf_build_*/export.json')); assert len(builds) == 1
    build = json.loads(builds[0].read_text())
    assert build['source_sha256'] == e.sha(report) and build['pdf_sha256'] == e.sha(pdf)
    assert not build['warnings']
    assert not any(ord(c) < 32 and c != '\n' for c in report.read_text())
    info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True, check=True).stdout
    pages = int(info.split('Pages:')[1].splitlines()[0])
    visual = json.loads((OUT / 'visual_review.json').read_text())
    assert visual['passed'] and visual['pdf_sha256'] == e.sha(pdf)
    assert visual['pages_inspected'] == list(range(1, pages + 1))
    e.write(OUT / 'rendering_check.json', dict(passed=True, pages=pages,
        numeric_claims_verified=len(claims['claims']), conclusion_facts_verified=len(conclusion['facts']),
        report_sha256=e.sha(report), pdf_sha256=e.sha(pdf), no_render_warnings=True,
        all_pages_actually_inspected=True, code_sha256=e.sha(Path(__file__)), goal_complete=False))

    bundle = OUT / 'delivery/predictor_bundle'; manifest = json.loads((bundle / 'manifest.json').read_text())
    assert len(manifest['presets']) == 624 and len(manifest['files']) == 390
    for item in manifest['files']: assert e.sha(bundle / item['path']) == item['sha256'], item['path']
    goal = json.loads((OUT / 'goal_reconfirmed_during_training.json').read_text())
    assert goal['system_goal']['status'] == 'active'
    train = audits['training_audit.json']; prediction = audits['prediction/audit.json']
    replay = audits['delivery/cohort_replay/audit.json']
    external = json.loads((OUT / 'external_Azizie/execution_complete.json').read_text())
    direction = dict(created_utc=e.now(), system_goal_status='active', goal_complete=False,
        direction='trained ML+Sandbox dopant-property prediction;theory serves actual predictive validation',
        primary=PRIMARY, main_variants=48, conditions=33, DOIs=6, related_groups=5,
        original29_roles_common18_preserved=True, common20_fixed=True,
        training_fit_records=train['fit_records'], actual_nonlinear_fits=train['actual_nonlinear_fits'],
        optimizer_starts=train['actual_optimizer_starts'], training_invalid=train['training_invalid'],
        selected_not_stopped=train['selected_not_stopped'],
        new_property_rows=prediction['prediction_rows'], combined_property_rows=9504,
        API_requests=replay['requests'], API_point_rows=replay['point_rows'],
        API_max_log_error=replay['maximum_log_property_difference'], external_rows=external['rows'],
        external_numeric=external['numeric'], external_valid=external['valid'],
        actual_learning_object='shared excess-over-charge-compensation atomshare;not measured microscopic rate',
        all48_methods_and_adverse_results_retained=True,
        known_carbon_unmodeled_in_physical_inventory_disclosed=True,
        matched_energy_endpoint_control_retained=True, capacity_limit_is_fixed_state_not_accuracy=True,
        new_mu_supervision=False, independent_sigma_labels=0,
        physical_information_budget_separate=True, no_new_DFT_MD=True, no_new_material_measurements=True,
        GPU_used=False, GPU_authorized='GPU1only',
        prediction_conclusions=conclusion, next=conclusion['next_predictive_work'])
    e.write(OUT / 'goal_direction_check.json', direction)
    for name in ('progress.json', 'goal_requirements_audit.json'):
        path = e.BASE / name; value = json.loads(path.read_text())
        value['phase27'] = dict(direction, status='complete_bounded_prediction_checkpoint', frozen=False)
        value.update(current_phase=27, goal_complete=False, latest_utc=e.now(),
            current_execution='phase27 actual full48 prediction deliverable checked;overall goal active',
            current_turn='progress:actual sharedtraining,48method predictions/API replay,transfer,theory/results reviewed',
            active_jobs=[], next_concrete_work=[direction['next']])
        if name == 'progress.json':
            value.update(status='active', current_report=str(pdf.relative_to(ROOT)),
                continuation=str((e.BASE / 'CONTINUE_PHASE27_HERE.md').relative_to(ROOT)))
        else:
            value.update(goal_status='active', objective=goal['system_goal']['objective'],
                sufficient_to_mark_goal_complete=False)
        e.write(path, value)
    print('Verified actual phase27 deliverable', pages, 'pages', len(claims['claims']),
        'values;full goal audit remains separate', flush=True)


if __name__ == '__main__': main()
