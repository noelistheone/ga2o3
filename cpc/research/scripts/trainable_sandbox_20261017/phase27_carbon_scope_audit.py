"""Actual input/provenance audit and metadata-wrapper numerical identity."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from sandbox.context_saved_model import INPUT_NAMES
from sandbox.charge_feasible_saved_model import SavedFeasibleModel
from sandbox.feasible_property_predictor_v2 import FeasiblePropertyPredictorV2
from sandbox.feasible_property_predictor_v3 import FeasiblePropertyPredictorV3
import phase27_training as t
import phase2_energy as e

OUT = t.OUT / 'carbon_scope'


def main():
    assert not (OUT / 'audit.json').exists()
    affected = t.TABLE[t.TABLE.carbon_known.eq(True) & (t.TABLE.carbon_cm3 > 0)]
    assert len(affected) == 1
    row = next(affected.itertuples())
    inputs = {name: e.clean(getattr(row, name)) for name in INPUT_NAMES}
    source = Path(row.source_file)
    text = source.read_text()
    assert '7.52' in text and 'upper-surface' in text
    snapshots = json.loads((t.OUT / 'saved_model_smoke/audit.json').read_text())['fit_snapshots']
    code = [Path(__file__), ROOT / 'src/sandbox/feasible_property_predictor_v3.py',
        ROOT / 'scripts/trainable_sandbox_20261017/predict_feasible_properties_v2.py',
        ROOT / 'docs/ML_SANDBOX_REPORTED_CARBON_SCOPE_CORRECTION_20261018.md']
    e.write(OUT / 'protocol_before.json', dict(created_utc=e.now(),
        purpose='reported carbon scope and metadata-only numerical identity;not carbon physical attribution',
        affected_condition=row.sample_id, reported_C_cm3=float(row.carbon_cm3),
        source_text_sha256=e.sha(source), source_locator='cached journal main text page2,lines94--100',
        interpretation='reported surface aggregation and average;not a uniform bulk budget',
        snapshots=snapshots, numeric_tolerance=2e-12, no_primary_or_training_change=True,
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in
            [t.parent.OUT / 'data/conditions33.csv', t.parent.OUT / 'data/episodes33.csv',
             t.OUT / 'interface_smoke/bundle/manifest.json']}))
    checks, records = [], []
    def check(name, error, tol=0.):
        checks.append(dict(name=name, error=float(error), tolerance=tol, passed=bool(error <= tol)))
    for spec in snapshots:
        fit = json.loads((ROOT / spec['fit_path']).read_text()); model = SavedFeasibleModel(fit)
        a = model.point(SimpleNamespace(**inputs), require_observation=False)
        zero = dict(inputs, carbon_known=False, carbon_cm3=0.)
        b = model.point(SimpleNamespace(**zero), require_observation=False)
        error = max(abs(np.r_[a.log_nH, a.log_muH, a.log_sigma] - np.r_[b.log_nH, b.log_muH, b.log_sigma]))
        check(fit['key'] + '/Cnot_in_state', error, 2e-12)
        records.append(dict(method=spec['method'], reg=spec['reg'], fit_key=fit['key'],
            actual_evaluation_method=fit['evaluation_method'], condition=row.sample_id,
            reported_C_cm3=float(row.carbon_cm3), three_property_C_toggle_difference=float(error),
            this_is_not_a_C_free_material_measurement=True))
    bundle = t.OUT / 'interface_smoke/bundle'
    old, new = FeasiblePropertyPredictorV2(bundle), FeasiblePropertyPredictorV3(bundle)
    for preset in old.manifest['presets']:
        if preset == 'invalid_reference': continue
        request = dict(preset=preset, inputs=[inputs], anchors=[])
        a, b = old.predict(**request), new.predict(**request)
        x, y = a['predictions'][0], b['predictions'][0]
        check(preset + '/unchanged_numeric_flags', float((x['numeric'], x['valid']) != (y['numeric'], y['valid'])))
        if x['valid']:
            check(preset + '/unchanged_numerics', max(abs(x[k] - y[k]) for k in
                ['log10_nH_cm3', 'log10_muH_cm2Vs', 'log10_sigma_S_cm']), 2e-12)
        check(preset + '/reported_C_disclosed', float(not y['reported_carbon_unmodeled_in_physical_state']))
        check(preset + '/warning_present', float(not any('carbon' in w for w in y['warnings'])))
    # Direct feature availability is different from a physical C atom budget.
    oldbundle = e.BASE / 'phase26_pool_share/delivery/predictor_bundle'
    model = FeasiblePropertyPredictorV3(oldbundle)
    a = model.predict(preset='DOI::full::DirectRidge', inputs=[inputs], anchors=[])
    check('Ridge/carbon_feature_marked', float(not a['predictions'][0]['carbon_available_to_direct_feature_model']))
    check('Ridge/physical_C_still_not_inferred', float(not a['predictions'][0]['reported_carbon_unmodeled_in_physical_state']))
    e.csvwrite(OUT / 'fixed_parameter_C_toggle.csv', records)
    e.csvwrite(OUT / 'affected_roles.csv', t.parent.ROLES[t.parent.ROLES.sample_id.eq(row.sample_id)].to_dict('records'))
    value = dict(passed=all(c['passed'] for c in checks), created_utc=e.now(), checks=len(checks), rows=checks,
        actual_conditions_with_reported_C=len(affected), current_conditions=33,
        source_surface_aggregation_not_uniform_bulk=True, fixed_snapshot_pairs=len(snapshots),
        model_numerics_or_training_changed=False, all_original_conditions_and_roles_retained=True,
        raw_valid_flag_not_full_composition_certificate=True, no_C_physical_effect_or_accuracy_claim=True,
        metadata_wrapper_not_full9504replay=True, final_V2_main_replay_separate=True,
        code_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in code}, goal_complete=False)
    e.write(OUT / 'audit.json', value)
    print('Actual reported-C scope audit', value['passed'], len(checks), 'snapshot pairs', len(snapshots), flush=True)
    if not value['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
