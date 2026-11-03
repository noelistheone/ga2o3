"""Independently check additional low-dose input/label isolation and all scores."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE = e.BASE / 'phase27_charge_feasible'
OUT = BASE / 'independent_input_audit/lowdose_transfer'


def main():
    protocol = json.loads((OUT / 'protocol_before_transfer.json').read_text())
    for kind in ('code_hashes', 'input_hashes'):
        for path, digest in protocol[kind].items(): assert e.sha(ROOT / path) == digest, path
    bundle = BASE / 'delivery/predictor_bundle'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    start = json.loads((OUT / 'execution_started.json').read_text())
    assert e.sha(bundle / 'manifest.json') == start['bundle_sha256']
    for item in manifest['files']:
        path = bundle / item['path']; assert e.sha(path) == item['sha256']
        fit = json.loads(path.read_text())
        assert '10.1063/5.0230413' not in fit['train_families']
    qual = json.loads((BASE / 'independent_input_audit/qualification.json').read_text())
    assert qual['condition']['calibration_sample_is_TableI_A'] is False
    assert qual['condition']['observed_muH_cm2Vs'] is None
    primary = read(e.BASE / 'phase25_shared_process/data/conditions33.csv')
    assert not primary.family.eq('10.1063/5.0230413').any()
    frame = read(OUT / 'predictions.csv'); summary = read(OUT / 'summary.csv')
    checks = []
    def check(name, error, tolerance=0.):
        checks.append(dict(name=name, error=float(error), tolerance=tolerance, passed=bool(error <= tolerance)))
    check('all_point_rows', abs(len(frame) - 576)); check('all_summary_rows', abs(len(summary) - 192))
    check('all48methods', abs(frame.variant.nunique() - 48))
    check('one_physical_condition', abs(frame.physical_condition_id.nunique() - 1))
    check('no_fake_mu_labels', float('observed_log_mu' in frame or 'observed_muH_cm2Vs' in frame))
    target = np.log10(3.96e15)
    for i in range(4):
        tasks = json.loads((OUT / f'tasks_{i}.json').read_text())
        outputs = json.loads((OUT / f'outputs_{i}.json').read_text())
        check(f'{i}/no_forbidden_table_reads', len(outputs['blocked_paths']))
        check(f'{i}/request_count', abs(len(tasks['queries']) - 24))
        actual = {r['id']: r for r in outputs['rows']}
        check(f'{i}/response_count', abs(len(actual) - 24))
        for request in tasks['queries']:
            assert set(request) == {'id', 'input'}
            spec = manifest['presets'][request['input']['preset']]
            check(request['id'] + '/zero_anchors', len(request['input']['anchors']))
            assert set(request['input']) == {'preset', 'inputs', 'anchors'}
            inputs = request['input']['inputs']; check(request['id'] + '/six_cases', abs(len(inputs) - 6))
            result = actual[request['id']]['result']
            q = frame[frame.id.eq(request['id'])]
            check(request['id'] + '/six_stored_cases', abs(len(q) - 6))
            for j, inp in enumerate(inputs):
                assert set(inp) == {'element', 'incorporated_cm3', 'Tobs_K', 'growth_T_C', 'H_known', 'H_cm3',
                    'carbon_known', 'carbon_cm3', 'scenario_pO2_atm', 'oxygen_input_kind'}
                row = q[(q.C_cm3 == inp['incorporated_cm3']) & q.oxygen_scenario.eq('old_effective' if inp['scenario_pO2_atm'] == 1e-5 else 'ideal_feed_proxy')]
                assert len(row) == 1; row = row.iloc[0]
                check(request['id'] + f'/{j}/observed_pair', abs(row.observed_log_nH - target), 1e-13)
                check(request['id'] + f'/{j}/outside_Si_support', float(inp['incorporated_cm3'] >= primary[primary.element.eq('Si')].incorporated_cm3.min()))
                point = result['predictions'][j] if result is not None else dict(valid=False, numeric=False)
                check(request['id'] + f'/{j}/numeric', float(bool(point['numeric']) != bool(row.numeric)))
                check(request['id'] + f'/{j}/valid', float(bool(point['valid']) != bool(row.valid)))
                check(request['id'] + f'/{j}/qualified_preserved', float(bool(spec['serving_qualified']) != bool(row.serving_qualified)))
                if row.valid:
                    check(request['id'] + f'/{j}/stored_prediction', abs(row.predicted - point['log10_nH_cm3']), 1e-13)
                    check(request['id'] + f'/{j}/absolute_error', abs(row.abs_error_dex - abs(row.predicted - target)), 1e-13)
                    check(request['id'] + f'/{j}/derived_sigma', abs(row.predicted_log_sigma -
                        (np.log10(1.602176634e-19) + row.predicted + row.predicted_log_mu)), 1e-10)
                else: check(request['id'] + f'/{j}/no_invalid_score', float(np.isfinite(row.abs_error_dex)))
    for r in summary.itertuples():
        group = frame[frame.context.eq(r.context) & frame.variant.eq(r.variant) & frame.oxygen_scenario.eq(r.oxygen_scenario)]
        assert len(group) == 3
        complete = bool(group.numeric.all() and group.valid.all())
        check(r.context + '/' + r.variant + '/' + r.oxygen_scenario + '/coverage', float(complete != bool(r.coverage_complete)))
        central = group[group.C_cm3.eq(3.5e15)].iloc[0]
        if central.valid: check(r.context + '/' + r.variant + '/' + r.oxygen_scenario + '/central', abs(r.central_nH_abs_error - central.abs_error_dex), 1e-13)
        else: check(r.context + '/' + r.variant + '/' + r.oxygen_scenario + '/missing_central', float(np.isfinite(r.central_nH_abs_error)))
        if complete:
            for column, value in [('sampled_prediction_min', group.predicted.min()), ('sampled_prediction_max', group.predicted.max()),
                ('sampled_abs_error_min', group.abs_error_dex.min()), ('sampled_abs_error_max', group.abs_error_dex.max())]:
                check(r.context + '/' + r.variant + '/' + r.oxygen_scenario + '/' + column, abs(getattr(r, column) - value), 1e-13)
        else:
            check(r.context + '/' + r.variant + '/' + r.oxygen_scenario + '/no_partial_range',
                float(any(np.isfinite(getattr(r, column)) for column in ('sampled_prediction_min', 'sampled_prediction_max', 'sampled_abs_error_min', 'sampled_abs_error_max'))))
    result = dict(passed=all(r['passed'] for r in checks), checks=len(checks), rows=checks,
        requests=96, point_rows=len(frame), numeric=int(frame.numeric.sum()), valid=int(frame.valid.sum()),
        physical_conditions=1, observed_mu_labels=0, observed_sigma_labels=0, all48_methods_retained=True,
        no_shared_or_source_refit=True, no_query_tables_read=True, no_main_cohort_or_scores_changed=True,
        after_main_outer_started=True, not_independent_blind_validation=True,
        outside_primary_Si_dose_support=True, sampled_ranges_not_continuum_or_confidence=True,
        code_hashes={str(Path(__file__).relative_to(ROOT)): e.sha(Path(__file__))},
        input_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in (OUT / 'protocol_before_transfer.json',
            BASE / 'independent_input_audit/qualification.json', OUT / 'predictions.csv', OUT / 'summary.csv', bundle / 'manifest.json')},
        goal_complete=False)
    e.write(OUT / 'audit.json', result)
    print('Low-dose onecondition/all48 audit', result['passed'], len(checks), flush=True)
    if not result['passed']: raise RuntimeError('Retain failed low-dose audit before a versioned repair')


if __name__ == '__main__': main()
