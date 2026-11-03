"""Read all completed held-out results without refitting or selecting a winner."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE = e.BASE / 'phase25_shared_process'
OUT = BASE / 'analysis'
INTERNAL = ('InventoryConstant', 'InventoryDose', 'InventoryProcess',
            'InventoryConstant_internal', 'InventoryDose_internal', 'InventoryProcess_internal')


def main():
    assert json.loads((BASE / 'prediction/audit.json').read_text())['passed']
    pred = read(BASE / 'prediction/predictions.csv')
    family = read(BASE / 'prediction/per_family.csv')
    choices = json.loads((BASE / 'prediction/selections.json').read_text())
    comparisons = []
    for scheme in ('DOI', 'cluster'):
        for budget in (0, 1, 2):
            d = family[family.scheme.eq(scheme) & family.budget.eq(budget) & family.scope.eq('common20')]
            for internal in INTERNAL:
                basis = internal.split('_')[0].replace('Inventory', '')
                controls = ['FixedPhysical', 'Output' + basis, 'MedianActivation', 'DoseLine',
                            'DirectRidge', 'DoseLineAffine', 'RidgeAffine', 'PhysicalAffine']
                if basis in ('Dose', 'Process'):
                    controls.append('Direct' + basis)
                a = d[d.variant.eq(internal)].set_index('family')
                for control in controls:
                    b = d[d.variant.eq(control)].set_index('family')
                    joined = a[['nH_MAE', 'related_cluster']].join(b.nH_MAE.rename('control'), how='inner')
                    complete = len(joined) == 6
                    delta = joined.nH_MAE - joined.control
                    grouped = delta.groupby(joined.related_cluster).mean()
                    comparisons.append(dict(scheme=scheme, budget=budget, internal=internal, control=control,
                        full_coverage=complete, DOIs=len(joined), related_groups=len(grouped),
                        delta_DOI_macro=float(delta.mean()) if complete else None,
                        delta_group_macro=float(grouped.mean()) if complete else None,
                        better_DOIs=int((delta < -1e-8).sum()), tied_DOIs=int((abs(delta) <= 1e-8).sum()),
                        worse_DOIs=int((delta > 1e-8).sum()), better_groups=int((grouped < -1e-8).sum()),
                        maximum_DOI_deterioration=float(delta.max()) if complete else None,
                        leave_one_group_delta_min=float(min(grouped.drop(g).mean() for g in grouped.index)) if complete else None,
                        leave_one_group_delta_max=float(max(grouped.drop(g).mean() for g in grouped.index)) if complete else None,
                        analysis_scope='retrospective descriptive comparisons;not a CI or multiplicity-adjusted hypothesis test'))
    e.csvwrite(OUT / 'matched_comparisons.csv', comparisons)
    diagnostics = []
    for c in choices:
        if c['budget'] != 0:
            continue
        f = json.loads((ROOT / c['fit_path']).read_text())
        singular = f.get('weighted_prediction_jacobian_singular_values', [])
        finite = [v for v in singular if v > 0]
        diagnostics.append(dict(scheme=c['scheme'], held_unit=c['held_unit'], variant=c['variant'],
            method=c['method'], shared_reg=c['reg'], model_key=c['fit_key'],
            parameters=f.get('parameters'), jacobian_rank=f.get('prediction_jacobian_numeric_rank'),
            declared_parameter_count=len(f['parameters']) if f.get('parameters') is not None else None,
            condition_number=float(max(finite) / min(finite)) if finite else None,
            rms_response_change=f.get('rms_response_change'),
            numerical_rank_not_structural_identifiability=True))
    e.csvwrite(OUT / 'selected_fit_diagnostics.csv', diagnostics)
    policies = []
    for c in choices:
        policies.append(dict(scheme=c['scheme'], held_unit=c['held_unit'], variant=c['variant'],
            budget=c['budget'], shared_reg=c['reg'], source_lambda=c['selected_source']['lam'],
            no_source_update=c['selected_source']['lam'] is None,
            final_context=c['held_unit'] is None))
    e.csvwrite(OUT / 'selected_policies.csv', policies)
    costs = []
    all_errors = []
    for i in range(4):
        packet = json.loads((BASE / f'prediction/outputs_{i}.json').read_text())
        assert not packet['denied_reads']
        for r in packet['records']:
            solutions = [s for ep in r['episodes'] for s in ep['solutions']]
            costs.append(dict(task_id=r['id'], kind=r['kind'], physical_point_evaluations=r['actual_physical_point_calls'],
                source_vectors=r['actual_source_vectors'], seconds=r['seconds'],
                point_failures=len(r['point_failures']), candidate_solutions=len(solutions),
                invalid_solutions=sum(not s['valid'] for s in solutions)))
            all_errors.extend(dict(task_id=r['id'], kind=r['kind'], **err) for err in r['point_failures'])
    e.csvwrite(OUT / 'source_costs.csv', costs)
    e.write(OUT / 'source_trial_failures.json', all_errors)
    train = json.loads((BASE / 'training_audit.json').read_text())
    stats = dict(passed=True, comparisons=len(comparisons), selected_fit_rows=len(diagnostics),
        source_tasks=len(costs), actual_source_point_evaluations=sum(r['physical_point_evaluations'] for r in costs),
        actual_source_vectors=sum(r['source_vectors'] for r in costs),
        point_failure_records=len(all_errors), invalid_candidate_solutions=sum(r['invalid_solutions'] for r in costs),
        training_fit_records=train['fit_records'], training_optimizer_starts=train['actual_optimizer_starts'],
        initial_not_stopped=train['initial_not_stopped'], selected_not_stopped=train['selected_not_stopped'],
        main_rows=len(pred), main_numeric=int(pred.numeric.sum()), main_valid=int(pred.valid.sum()),
        new_fits=0, new_measurements=0, outer_selected_method=False,
        source_calls_scope='includes empirical point calls;not all calls are nonlinear physical solves',
        goal_complete=False, code_sha256=e.sha(Path(__file__)))
    e.write(OUT / 'audit.json', stats)
    print('phase25 analysis', stats, flush=True)


if __name__ == '__main__':
    main()
