"""Retrospective read-only diagnosis of saved process-share validation failures.

This is a new diagnostic, not a repair of the frozen experiment.  It never
retrains, changes a prediction, or uses a held-out label to select a model.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from csv_io_v2 import read_result_csv as read
from sandbox import thermo
from sandbox.pool_share_saved_model import SavedPoolModel
from sandbox.pool_share_controls import PoolReferenceModel
from sandbox.coherent_fermi import published_coherent_profile
import phase26_training as t
import phase2_energy as e

OUT = t.OUT / 'inner_domain_diagnosis'
PROTOCOL = OUT / 'protocol_before_replay.json'


def census():
    table = read(t.OUT / 'prediction/inner_zero_predictions.csv')
    subset = table[table.method.eq('ShareEffectiveProcess') & table.reg.notna()]
    subset = subset.drop_duplicates(['fit_key', 'sample_id'])
    bad = subset[~subset.numeric.eq(True) | ~subset.valid.eq(True)]
    cases = bad[['fit_key', 'sample_id', 'scheme', 'reg']].sort_values(['fit_key', 'sample_id'])
    return subset, bad, cases.to_dict('records')


def declare():
    allpoints, bad, cases = census()
    deps = [Path(__file__), t.OUT / 'prediction/inner_zero_predictions.csv',
            t.OUT / 'training_protocol.json', t.parent.OUT / 'data/conditions33.csv',
            ROOT / 'src/sandbox/pool_share_model.py', ROOT / 'src/sandbox/absolute_pool_share.py',
            ROOT / 'src/sandbox/fixed_inventory_charge.py', ROOT / 'src/sandbox/pool_share_saved_model.py',
            ROOT / 'src/sandbox/pool_share_controls.py', ROOT / 'src/sandbox/coherent_fermi.py']
    for key in sorted(bad.fit_key.unique()):
        path = t.OUT / 'fits' / (key + '.json')
        fit = json.loads(path.read_text())
        _, old = t.inherited_reference(fit['scheme'], fit['train_units'])
        deps.extend([path, old])
    inputs = {str(p.relative_to(ROOT)): e.sha(p) for p in dict.fromkeys(deps)}
    protocol = dict(created_utc=e.now(), scope='retrospective diagnosis; no accuracy selection or fitting',
        unique_finite_process_inner_points=len(allpoints), selected_failed_points=len(cases),
        selected_nonnumeric_points=int((~bad.numeric.eq(True)).sum()), cases=cases,
        comparisons=['saved learned share', 'reference at the same learned Tf/background',
                     'inherited reference trained on exactly the same units'],
        theory='At fixed EF, Q is affine in the absolute pool fraction; cold Q_EF<0 gives the conditional free-electron threshold.',
        free_electron_floor_cm3=1e13, hall_floor_is_separate=True,
        replay_tolerance_dex=2e-6, charge_algebra_relative_tolerance=3e-8,
        no_observed_labels_used=True, no_training=True, no_original_rows_modified=True,
        no_new_experimental_measurements=True, no_new_DFT_MD=True, files=inputs)
    if PROTOCOL.exists():
        old = json.loads(PROTOCOL.read_text())
        assert {k: v for k, v in old.items() if k != 'created_utc'} == {k: v for k, v in protocol.items() if k != 'created_utc'}
    else:
        e.write(PROTOCOL, protocol)
    return json.loads(PROTOCOL.read_text())


def charge_mean(layer, pars, ef, T):
    H, _, _, profile = layer._reference(T, pars)
    entropy, _ = layer._entropy(T)
    A = H - 3 * thermo.KB_EV * T * entropy * layer.db.sum_dm
    logw = -(A + layer.q * ef) / (thermo.KB_EV * T) + layer.logg
    maxima = np.full(layer.db.nD, -np.inf)
    np.maximum.at(maxima, layer.parent, logw)
    weights = np.exp(logw - maxima[layer.parent])
    pi = weights / np.bincount(layer.parent, weights=weights, minlength=layer.db.nD)[layer.parent]
    ell, _ = layer.fraction(pars)
    for plus, neutral in layer.pairs:
        difference = logw[neutral] - logw[plus]
        pi[neutral] = ell * expit(difference)
        pi[plus] = (1 - ell) + ell * expit(-difference)
    mean = np.bincount(layer.parent, weights=pi * layer.q, minlength=layer.db.nD)
    return mean, profile.carriers(ef, T)[0]


def threshold(row, w, operator, point):
    layer, solver = operator.engine_for(row).layer_for(row)
    inventory = solver.last_inventory
    cold = solver.last_prediction
    T = point.diagnostics['Tf_K']
    pars = layer.parameters(T=T, Nd=10 ** w[1], weights=[0.],
        doses=(row.incorporated_cm3, row.H_cm3) if row.H_known else (row.incorporated_cm3,),
        pO2=row.scenario_pO2_atm, Tobs=row.Tobs_K, binding_eV=.0366, screen_s=0.)
    reference = solver.reference(pars)
    c0 = reference.totals
    cx, isolated = solver.complex, solver.receivers
    U = float(solver.dopant_count[cx] @ c0[cx])
    V = float(c0[isolated].sum())
    W = U + V
    fi = float(inventory.diagnostics['isolated_atom_fraction_after'])
    ef_floor = brentq(lambda ef: np.log10(operator.profile.carriers(ef, row.Tobs_K)[0][0]) - 13., -5., 12.)
    mean, carriers = charge_mean(layer, pars, ef_floor, row.Tobs_K)
    q_complex = float((c0[cx] / U) @ mean[cx])
    q_isolated = float((c0[isolated] / V) @ mean[isolated])
    other = ~(cx | isolated)
    Qother = float(c0[other] @ mean[other] + 10 ** w[1] + carriers[1:].sum() - carriers[0])
    slope = W * (q_isolated - q_complex)
    assert slope > 0, 'This diagnosis claims the threshold only for positive contrast'
    fcrit = -(Qother + W * q_complex) / slope
    Qaffine = Qother + W * q_complex + slope * fi
    Qdirect = float(inventory.totals @ mean + 10 ** w[1] + carriers[1:].sum() - carriers[0])
    scale = max(abs(Qother), abs(W * q_complex), abs(slope), 1.)
    mean_root, _ = charge_mean(layer, pars, cold.ef_eV, row.Tobs_K)
    root_mean_error = float(max(abs(mean_root - np.bincount(layer.parent,
        weights=cold.probabilities * layer.q, minlength=layer.db.nD))))
    return dict(isolated_fraction=fi, reference_isolated_fraction=V / W,
        critical_free_floor_fraction=float(fcrit), fraction_minus_free_critical=float(fi - fcrit),
        critical_fraction_feasible=bool(0 <= fcrit < 1), q_per_complex_atom=q_complex,
        q_per_isolated_atom=q_isolated, cold_EF_eV=float(cold.ef_eV), free_floor_EF_eV=float(ef_floor),
        charge_affine_relative_error=abs(Qaffine - Qdirect) / scale, root_mean_charge_error=root_mean_error,
        below_free_floor_by_threshold=bool(fi < fcrit),
        below_free_floor_actual=bool(point.diagnostics['free_n_cm3'] < 1e13),
        free_threshold_agrees=bool((fi < fcrit) == (point.diagnostics['free_n_cm3'] < 1e13)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    protocol = declare()
    if args.declare_only:
        print('Declared saved inner-domain diagnostic', len(protocol['cases']), flush=True)
        return
    assert not (OUT / 'audit.json').exists(), 'Keep the first diagnostic outcome; do not overwrite'
    for p, h in protocol['files'].items():
        assert e.sha(ROOT / p) == h, p
    allpoints, _, _ = census()
    oldrows = {(r.fit_key, r.sample_id): r for r in allpoints.itertuples()}
    inputs = {r.sample_id: r for r in t.parent.TABLE.itertuples()}
    reference = PoolReferenceModel(temperature_mode='effective', profile=published_coherent_profile())
    models, records = {}, []
    for i, case in enumerate(protocol['cases']):
        key, sample = case['fit_key'], case['sample_id']
        if key not in models:
            fit = json.loads((t.OUT / 'fits' / (key + '.json')).read_text())
            inherited, _ = t.inherited_reference(fit['scheme'], fit['train_units'])
            models[key] = (fit, inherited, SavedPoolModel(fit))
        fit, inherited, saved = models[key]
        row = inputs[sample]
        record = dict(case, evaluation_numeric=False, error=None)
        try:
            p = saved.point(row, source_kind='field', shift=0., require_observation=False)
            dg = p.diagnostics
            record.update(evaluation_numeric=True, valid=p.valid, log10_nH=p.log_nH,
                replay_error_dex=abs(p.log_nH - oldrows[(key, sample)].predicted),
                log10_free_n=float(np.log10(dg['free_n_cm3'])), log10_free_p=float(np.log10(dg['free_p_cm3'])),
                hole_dominant=not dg['single_electron_Hall_valid'], hall_below_floor=p.log_nH < 13.,
                free_below_floor=dg['free_n_cm3'] < 1e13, hall_factor=dg['hall_factor'],
                source_response=p.source_derivative, Tf_K=dg['Tf_K'],
                hot_root_residual=dg['hot_root_residual'], cold_Q_EF=dg['cold_Q_EF_cm3_per_eV'])
            record.update(threshold(row, np.asarray(fit['parameters']), saved.operator, p))
            same = reference.point(row, np.asarray(fit['parameters'])[:2], require_observation=False)
            base = reference.point(row, np.asarray(inherited['parameters'])[:2], require_observation=False)
            record.update(same_base_reference_valid=same.diagnostics['observation_domain_valid'],
                same_base_reference_nH=float(same.values[0]),
                inherited_reference_valid=base.diagnostics['observation_domain_valid'],
                inherited_reference_nH=float(base.values[0]))
        except Exception as exc:
            record['error'] = type(exc).__name__ + ': ' + str(exc)
        records.append(record)
        if (i + 1) % 25 == 0:
            print('inner-domain replay', i + 1, '/', len(protocol['cases']), flush=True)
    e.csvwrite(OUT / 'point_diagnoses.csv', records)
    numerical = [r for r in records if r['evaluation_numeric'] and r['error'] is None]
    passed = (len(numerical) == len(records) and
        max((r['replay_error_dex'] for r in numerical), default=float('inf')) <= protocol['replay_tolerance_dex'] and
        all(r['free_threshold_agrees'] and r['charge_affine_relative_error'] < protocol['charge_algebra_relative_tolerance'] and
            r['root_mean_charge_error'] < protocol['charge_algebra_relative_tolerance'] for r in numerical))
    def count(name):
        return sum(bool(r[name]) for r in numerical)
    audit = dict(passed=bool(passed), created_utc=e.now(), cases=len(records), completed=len(numerical),
        errors=[{'fit_key': r['fit_key'], 'sample_id': r['sample_id'], 'error': r['error']} for r in records if r['error']],
        saved_invalid=count('hall_below_floor'), hole_dominant=count('hole_dominant'),
        free_below_floor=count('free_below_floor'),
        same_learned_base_reference_valid=count('same_base_reference_valid'),
        inherited_reference_valid=count('inherited_reference_valid'),
        critical_free_fraction_feasible=count('critical_fraction_feasible'),
        maximum_replay_error_dex=max((r['replay_error_dex'] for r in numerical), default=None),
        all_positive_charge_contrast=bool(numerical and all(r['q_per_isolated_atom'] > r['q_per_complex_atom'] for r in numerical)),
        no_new_fits_or_labels=True, no_original_result_changes=True, no_accuracy_improvement_claim=True,
        free_floor_not_Hall_floor=True, protocol_sha256=e.sha(PROTOCOL),
        records_sha256=e.sha(OUT / 'point_diagnoses.csv'), code_sha256=e.sha(Path(__file__)), goal_complete=False)
    e.write(OUT / 'audit.json', audit)
    print('Saved inner-domain diagnostic', {k: v for k, v in audit.items() if k != 'errors'}, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
