"""Annotate every frozen main prediction without filtering or changing scores."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE = e.BASE / 'phase27_charge_feasible'
OUT = BASE / 'input_scope_annotations'
DATA = e.BASE / 'phase25_shared_process/data'


def declare():
    inputs = [DATA / 'conditions33.csv', DATA / 'episodes33.csv',
        BASE / 'pre_evaluation_protocol.json', BASE / 'carbon_scope/addendum_before_outer.json',
        BASE / 'carbon_scope/audit.json']
    protocol = dict(created_utc=e.now(), purpose='all-row input scope,not new accuracy selection',
        expected_rows=9504, expected_conditions=33, expected_known_C_conditions=1, expected_known_C_rows=288,
        known_C_interpretation='sampleD reported surface accumulation and average,not uniform bulk',
        no_row_exclusion=True, no_refit=True, no_changed_prediction_or_metric=True,
        code_hashes={str(Path(__file__).relative_to(ROOT)): e.sha(Path(__file__))},
        input_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in inputs}, goal_complete=False)
    dest = OUT / 'protocol_before.json'
    if dest.exists():
        old = json.loads(dest.read_text())
        for field in ('code_hashes', 'input_hashes'): assert old[field] == protocol[field]
        return old
    e.write(dest, protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args(); protocol = declare()
    if args.declare_only:
        print('Declared all9504 input-scope annotations,main numerics unchanged', flush=True); return
    assert not (OUT / 'audit.json').exists(), 'Preserve the first annotation outcome'
    for group in ('code_hashes', 'input_hashes'):
        for name, h in protocol[group].items(): assert e.sha(ROOT / name) == h, name
    assert json.loads((BASE / 'prediction/audit.json').read_text())['passed']
    predictions = read(BASE / 'prediction/combined_predictions.csv')
    inputs = read(DATA / 'conditions33.csv').set_index('sample_id')
    roles = read(DATA / 'episodes33.csv').set_index(['family', 'budget', 'sample_id']).role
    bundle = BASE / 'delivery/predictor_bundle'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    fits, rows = {}, []
    for r in predictions.itertuples():
        source = inputs.loc[r.sample_id]
        preset = r.scheme + '::without:' + r.held_unit + '::' + r.variant
        spec = manifest['presets'][preset]; key = spec['model_key']
        assert key == r.fit_key
        if key not in fits: fits[key] = json.loads((bundle / 'models' / (key + '.json')).read_text())
        fit = fits[key]; actual = fit.get('evaluation_method', fit['method'])
        assert roles[r.family, int(r.budget), r.sample_id] == r.role
        known = bool(source.carbon_known and source.carbon_cm3 > 0)
        physical = actual in ('FixedPhysical', 'ParentReference', 'GrowthReference') or actual.startswith(('Inventory', 'Output', 'Share', 'Feasible'))
        endpoint = actual.startswith('Envelope')
        state_elements = [source.element] + (['H'] if source.H_known else []) if physical or endpoint else []
        rows.append(dict(scheme=r.scheme, variant=r.variant, budget=int(r.budget), family=r.family,
            sample_id=r.sample_id, role=r.role, common_query=bool(r.common_query), numeric=bool(r.numeric), valid=bool(r.valid),
            actual_evaluation_method=actual, physical_midpoint_model=physical, physical_endpoint_only=endpoint,
            modeled_state_budget_elements=state_elements, reported_H_known=bool(source.H_known),
            reported_C_known=bool(source.carbon_known), reported_C_cm3=float(source.carbon_cm3),
            reported_C_unmodeled_in_physical_state=known,
            carbon_available_to_direct_feature_model=bool(known and fit['method'] == 'DirectRidge'),
            scope='restricted modeled-state/observation validity,not complete composition or accuracy certification'))
    assert len(rows) == protocol['expected_rows']
    assert predictions.sample_id.nunique() == protocol['expected_conditions']
    known = [r for r in rows if r['reported_C_unmodeled_in_physical_state']]
    assert len(known) == protocol['expected_known_C_rows']
    assert len({r['sample_id'] for r in known}) == 1
    e.csvwrite(OUT / 'all_main_rows.csv', rows)
    e.csvwrite(OUT / 'reported_C_rows.csv', known)
    e.write(OUT / 'audit.json', dict(passed=True, created_utc=e.now(), rows=len(rows), conditions=33,
        known_C_rows=len(known), known_C_conditions=1, no_deleted_or_modified_predictions=True,
        no_model_refit_or_new_measurements=True, no_scores_recomputed_or_selected=True,
        role_and_model_identity_verified_for_all_rows=True,
        main_predictions_sha256=e.sha(BASE / 'prediction/combined_predictions.csv'),
        annotation_sha256=e.sha(OUT / 'all_main_rows.csv'), protocol_sha256=e.sha(OUT / 'protocol_before.json'),
        code_sha256=e.sha(Path(__file__)), goal_complete=False))
    print('All main input-scope rows annotated', len(rows), 'known-C rows', len(known), flush=True)


if __name__ == '__main__': main()
