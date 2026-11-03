"""Additional exposed-source, zero-anchor low-dose check;main design unchanged."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE = e.BASE / 'phase27_charge_feasible'
SOURCE = BASE / 'independent_input_audit'
OUT = SOURCE / 'lowdose_transfer'
BUNDLE = BASE / 'delivery/predictor_bundle'
CONTEXTS = ('cluster::full', 'cluster::without:10.1063/5.0027870')
DOSES = (3e15, 3.5e15, 4e15)
OXYGEN = {'old_effective': 1e-5, 'ideal_feed_proxy': 15. / 760. * 1200. / (1200. + 1500.)}


def declare():
    qual = json.loads((SOURCE / 'qualification.json').read_text())
    assert qual['condition']['observed_muH_cm2Vs'] is None
    assert e.sha(ROOT / qual['source']['local_pdf']) == qual['source']['pdf_sha256']
    assert (BASE / 'prediction/execution_started.json').exists()
    assert not OUT.exists() or not (OUT / 'execution_started.json').exists()
    inputs = [SOURCE / 'qualification.json', ROOT / qual['source']['local_pdf'],
        BASE / 'pre_evaluation_protocol.json', BASE / 'training_protocol.json',
        BASE / 'training_complete.json', BASE / 'prediction/shared_choices.json',
        e.BASE / 'phase25_shared_process/data/conditions33.csv']
    code = [Path(__file__), Path(__file__).with_name('phase27_lowdose_audit.py'),
        Path(__file__).with_name('phase27_api_runtime_v2.py'),
        ROOT / 'src/sandbox/feasible_property_predictor_v2.py',
        ROOT / 'src/sandbox/feasible_property_predictor.py']
    value = dict(created_utc=e.now(), purpose='one exposed published low-dose chemical/Hall condition stress test',
        main_outer_execution_already_started=True, declaration_is_before_this_transfer_only=True,
        source_already_read_and_previously_cited=True, independent_blind_validation=False,
        no_changed_main_protocol_or_selection=True, main_rows_still9504=True,
        contexts=CONTEXTS, variants=48, requests=96, point_rows=576,
        doses_cm3=DOSES, oxygen_scenarios=OXYGEN, central_C_cm3=3.5e15,
        oxygen_proxy_assumptions='ideal O2/Ar feed fraction times reported15Torr;precursor/minor-flow corrections ignored;not measured surface equilibrium chemical potential',
        condition_id=qual['condition']['sample_id'], observed_nH_cm3=3.96e15,
        physical_conditions=1, observed_mu_labels=0, independent_sigma_labels=0,
        anchors=0, no_source_or_shared_refit=True,
        H_C='below-detection statements;quantitative bounds not audited;unknown-zero modeled scenario,not measured absence',
        C_range='printed representative plateau range,not statistical CI;three sampled values do not enclose every continuous prediction',
        institution_exclusion='remove same-UCSB older Sn DOI;not proof of shared samples or complete lab-independence',
        qualification='keep all48 including refusals;no partial-coverage precision claim;outside primary Si dose support flagged',
        metrics='predeclared central-dose error and all three sampled responses per oxygen scenario;no best oxygen/model chosen from label',
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in inputs},
        no_new_DFT_MD=True, no_new_material_measurements=True, goal_complete=False)
    target = OUT / 'protocol_before_transfer.json'
    if target.exists():
        old = json.loads(target.read_text())
        for k in ('code_hashes', 'input_hashes', 'contexts', 'doses_cm3', 'oxygen_scenarios'):
            assert old[k] == e.clean(value[k]), k
        return old
    e.write(target, value)
    return value


def worker(i):
    run = subprocess.run([sys.executable, str(Path(__file__).with_name('phase27_api_runtime_v2.py')),
        str(OUT / f'tasks_{i}.json'), str(OUT / f'outputs_{i}.json')], cwd=ROOT, capture_output=True, text=True)
    (OUT / f'worker_{i}.log').write_text(run.stdout + '\n' + run.stderr)
    run.check_returncode()
    return json.loads((OUT / f'outputs_{i}.json').read_text())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args(); protocol = declare()
    if args.declare_only:
        print('Declared additional low-dose96requests/576points,onephysicalcondition;not main preregistration', flush=True); return
    assert json.loads((BASE / 'delivery/cohort_replay/audit.json').read_text())['passed']
    for kind in ('code_hashes', 'input_hashes'):
        for path, digest in protocol[kind].items(): assert e.sha(ROOT / path) == digest, path
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    primary = read(e.BASE / 'phase25_shared_process/data/conditions33.csv')
    minimum = float(primary[primary.element.eq('Si')].incorporated_cm3.min())
    assert minimum > max(DOSES)
    tasks = []; refs = {}; nodes = []; case_info = []
    for oxygen, value in OXYGEN.items():
        for dose in DOSES:
            nodes.append(dict(element='Si', incorporated_cm3=dose, Tobs_K=300., growth_T_C=950.,
                H_known=False, H_cm3=0., carbon_known=False, carbon_cm3=0., scenario_pO2_atm=value,
                oxygen_input_kind=oxygen + ';declared conditional endpoint oxygen,not measured surface equilibrium'))
            case_info.append(dict(oxygen_scenario=oxygen, C_cm3=dose,
                central_dose=dose == protocol['central_C_cm3'], outside_primary_Si_dose_support=True))
    for context in CONTEXTS:
        presets = sorted(n for n, s in manifest['presets'].items() if s['context'] == context)
        assert len(presets) == 48
        for preset in presets:
            spec = manifest['presets'][preset]; key = 'q' + str(len(tasks))
            tasks.append(dict(id=key, input=dict(preset=preset, inputs=nodes, anchors=[])))
            refs[key] = dict(context=context, variant=spec['variant'], preset=preset,
                serving_qualified=spec['serving_qualified'], model_key=spec['model_key'])
    assert len(tasks) == 96
    e.write(OUT / 'execution_started.json', dict(created_utc=e.now(), requests=96, points=576,
        bundle_sha256=e.sha(BUNDLE / 'manifest.json'), minimum_primary_Si_cm3=minimum,
        shared_or_source_fits=0, goal_complete=False))
    e.write(OUT / 'request_reference.json', refs)
    for i in range(4): e.write(OUT / f'tasks_{i}.json', dict(bundle=str(BUNDLE.resolve()), queries=tasks[i::4]))
    with ThreadPoolExecutor(max_workers=4) as pool: packets = list(pool.map(worker, range(4)))
    rows = []
    for packet in packets:
        assert not packet['blocked_paths']
        for response in packet['rows']:
            ref = refs[response['id']]; result = response['result']
            for j, case in enumerate(case_info):
                point = result['predictions'][j] if result is not None else dict(numeric=False, valid=False)
                valid = bool(point['valid']); numeric = bool(point['numeric'])
                row = dict(id=response['id'], **ref, **case, physical_condition_id=protocol['condition_id'],
                    observed_log_nH=float(np.log10(protocol['observed_nH_cm3'])),
                    numeric=numeric, valid=valid, error=response['error'], point_error=point.get('error'),
                    predicted=None, abs_error_dex=None, predicted_log_mu=None, predicted_log_sigma=None,
                    unaccepted_numeric_continuation=point.get('continuation_log10_nH'),
                    warnings=point.get('warnings', []))
                if valid:
                    row.update(predicted=point['log10_nH_cm3'],
                        abs_error_dex=abs(point['log10_nH_cm3'] - row['observed_log_nH']),
                        predicted_log_mu=point['log10_muH_cm2Vs'], predicted_log_sigma=point['log10_sigma_S_cm'])
                rows.append(row)
    assert len(rows) == 576
    e.csvwrite(OUT / 'predictions.csv', rows)
    summaries = []
    for context in CONTEXTS:
        variants = sorted({r['variant'] for r in rows if r['context'] == context})
        for variant in variants:
            for oxygen in OXYGEN:
                group = [r for r in rows if r['context'] == context and r['variant'] == variant and r['oxygen_scenario'] == oxygen]
                assert len(group) == 3
                central = next(r for r in group if r['central_dose']); complete = all(r['numeric'] and r['valid'] for r in group)
                summaries.append(dict(context=context, variant=variant, oxygen_scenario=oxygen, sampled_doses=3,
                    independent_physical_conditions=1, coverage_complete=complete,
                    central_nH_abs_error=central['abs_error_dex'],
                    sampled_prediction_min=min(r['predicted'] for r in group) if complete else None,
                    sampled_prediction_max=max(r['predicted'] for r in group) if complete else None,
                    sampled_abs_error_min=min(r['abs_error_dex'] for r in group) if complete else None,
                    sampled_abs_error_max=max(r['abs_error_dex'] for r in group) if complete else None,
                    grade='conditional one-source stress,sampled chemical/oxygen sensitivity;not interval coverage or blind validation'))
    e.csvwrite(OUT / 'summary.csv', summaries)
    e.write(OUT / 'execution_complete.json', dict(created_utc=e.now(), requests=96, rows=len(rows),
        summary_rows=len(summaries), numeric=sum(r['numeric'] for r in rows), valid=sum(r['valid'] for r in rows),
        physical_conditions=1, new_fits=0, new_mu_truth=0, goal_complete=False))
    print('Actual low-dose frozen transfer', len(rows), 'rows;onephysicalcondition;all48retained', flush=True)


if __name__ == '__main__': main()
