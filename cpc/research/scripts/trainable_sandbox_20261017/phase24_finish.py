"""Close the chemical-learning checkpoint without closing the predictive goal."""
from pathlib import Path
import sys, json, subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE = e.BASE
OUT = BASE / 'phase24_chemical_process'


def main():
    report = OUT / 'ML与Sandbox_化学过程学习与完整性质预测.md'
    pdf = report.with_suffix('.pdf')
    source = report.read_text()
    assert not any(ord(c) < 32 and c != '\n' for c in source)
    assert all(s in source for s in ('配对选择的full5', '倒温度校准坐标', r'\nu_i', r'\frac'))
    claims = json.loads((OUT / 'report_claims.json').read_text())
    assert claims['report_sha256'] == e.sha(report) and len(claims['claims']) == 571
    cache = {}
    for c in claims['claims']:
        p = ROOT / c['path']
        assert e.sha(p) == c['sha256'], p
        if p not in cache:
            cache[p] = read(p)
        d = cache[p]
        for k, v in c['selector'].items():
            d = d[d[k].eq(v)]
        assert len(d) == 1 and float(d[c['column']].iloc[0]) == c['value'], c
    builds = list(OUT.glob('_pdf_build_*/export.json'))
    assert len(builds) == 1
    render = json.loads(builds[0].read_text())
    assert not render['warnings'] and render['source_sha256'] == e.sha(report) and render['pdf_sha256'] == e.sha(pdf)
    info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True, check=True).stdout
    pages = int(info.split('Pages:')[1].splitlines()[0])
    assert pages == 8
    visual = json.loads((OUT / 'visual_review.json').read_text())
    assert visual['passed'] and visual['pdf_sha256'] == e.sha(pdf) and visual['pages_inspected'] == list(range(1, 9))
    extracted = Path('/tmp/ga2o3_phase24_report.txt').read_text()
    assert all(t in extracted for t in ('32940', '0.593217', '0.347842', '0.016661', '0.018368'))
    e.write(OUT / 'rendering_check.json', dict(passed=True, pages=8, numeric_claims_verified=571,
        report_sha256=e.sha(report), pdf_sha256=e.sha(pdf), no_render_warnings=True,
        math_control_characters_absent=True, paired_validation_distinction_present=True))
    paths = ['extraction/extraction_audit.json', 'chemical_learning/audit.json', 'paired_validation/audit.json',
        'property_prediction/audit/contract_audit.json', 'paired_property_prediction/audit.json',
        'process_gradients/gates.json', 'process_gradients/tied_parameter_gates.json', 'inventory_scope/audit.json',
        'delivery/CLI_smoke.json', 'delivery/paired_CLI_smoke.json', 'parent_verification.json']
    audits = {p: json.loads((OUT / p).read_text()) for p in paths}
    assert all(d['passed'] for d in audits.values())
    assert audits[paths[3]]['requests'] == 26352 and audits[paths[4]]['requests'] == 6588
    assert audits[paths[1]]['fit_records'] == 1188 and audits[paths[2]]['fit_records'] == 2346
    assert audits['parent_verification.json']['entries'] == 43717
    goal = json.loads((OUT / 'system_goal_checked.json').read_text())
    assert goal['goal']['status'] == 'active'
    direction = dict(created_utc=e.now(), system_goal_status='active', goal_complete=False,
        direction='trainable ML+Sandbox predicts doped-material properties; theory serves predictive development',
        primary='actual incorporated chemistry+traceable inputs->nH; compatible mu/derived sigma and supported ranking in scope',
        original29_5_anchors_common18_unchanged=True, chemical_layers=12, chemical_stacks=2, chemical_DOIs=1,
        chemical_fit_records=3534, original_chemical_models=12, paired_selected_chemical_models=3,
        chemical_checks=14786, paired_chemical_checks=30062, zero_anchor_gradient_checks=37, tied_gradient_cases=6,
        requests=32940, numeric_requests=32940, refusals=0, summary_rows=5490,
        original_property_checks=272668, paired_property_checks=57735,
        full5_methods=34, related_source_excluded_methods=27,
        new_Hall_training_labels=0, new_shared_property_training=0,
        chemistry_scope='same DOI/two SIMS stacks; nominal source T; independent chemical supervision, not same-Hall measured dose',
        conditional_chemical_signal='known-rate paired outer Arr .026868 vs RateAffine .016661; nested selected policy .018368 dex',
        chemical_selection_correction='whole-stack holdout removes a rate level, so extra rate effect is unidentifiable in that task',
        stable_new_property_gain_established=False, nonlinear_chemical_ML_unique_gain_established=False,
        retained_property_failure='paired full5 Original .593217 vs Median2 .347842; chemistry gain does not close Hall deficit',
        inventory_hypothesis_rejected='same-Ga structural pools dominate the negative charge',
        independent_blind_validation=False, independent_new_lab=False, new_DFT_MD=0, new_material_measurements=0,
        GPU_used=False, GPU_authorized='GPU1 only',
        next='qualify Leedy uniform chemical/Hall conditions in a separate dataset version, then train matched shared dose/process inventory relationships and controls',
        next_theory='docs/ML_SANDBOX_HIGH_DOSE_SHARED_LEARNING_NEXT_20261018.md', continuation='CONTINUE_PHASE24_HERE.md')
    e.write(OUT / 'goal_direction_check.json', direction)
    for name in ('progress.json', 'goal_requirements_audit.json'):
        p = BASE / name
        v = json.loads(p.read_text())
        v.update(current_phase=24, goal_complete=False, active_jobs=[], latest_utc=e.now(), checked_utc=e.now(),
            protected_files_checked=43717, changed_protected_files=0)
        v['phase24'] = dict(direction, status='complete_bounded_prediction_checkpoint', frozen=False)
        v['phase24_running'] = dict(status='finished', goal_complete=False)
        v['current_execution'] = 'phase24 complete; independent chemistry learning and 32940 actual coupled property forecasts; no stable property increment'
        v['current_turn'] = 'progress: process chemistry trained and independently differentiated; validation task corrected; predictive failures retained'
        v['next_concrete_work'] = [direction['next'], 'retain actual dopant property prediction as the primary endpoint and all strong controls']
        if name == 'progress.json':
            v.update(status='active', stage='phase24_chemical_process_prediction_checkpoint',
                current_report=str(pdf.relative_to(ROOT)), latest_report_directory=str(OUT.relative_to(ROOT)),
                continuation=str((BASE / 'CONTINUE_PHASE24_HERE.md').relative_to(ROOT)), next_theory=direction['next_theory'], next=[direction['next']])
            item = 'phase24:3534 chemical fits/32940 coupled forecasts; known-rate conditional chemical gain but no stable property gain; analytic chain/tied-parameter gradients verified'
            if item not in v['completed']:
                v['completed'].append(item)
        else:
            v.update(goal_status='active', objective=goal['goal']['objective'], sufficient_to_mark_goal_complete=False)
            item = dict(requirement='trained process-to-chemical relation actually coupled to dopant-property Sandbox',
                evidence=['phase24_chemical_process/paired_validation/audit.json', 'phase24_chemical_process/paired_property_prediction/audit.json',
                    'phase24_chemical_process/process_gradients/tied_parameter_gates.json'],
                scope='known-rate chemical signal on two stacks; no new Hall-trained relation or stable property gain; nominal dose proxies retained')
            if not any(x['requirement'] == item['requirement'] for x in v['implemented_and_verified']):
                v['implemented_and_verified'].append(item)
        e.write(p, v)
    print('phase24 verified:571 claims/8 pages;3534 fits/32940 forecasts;goal active', flush=True)


if __name__ == '__main__':
    main()
