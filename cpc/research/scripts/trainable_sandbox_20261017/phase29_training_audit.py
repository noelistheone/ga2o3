"""Independent metadata, supervision and budget audit of phase29 actual fits."""
from pathlib import Path
import sys,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase29_learning as l
import phase29_training_v2 as t
import phase29_data as data
import phase2_energy as e


def main():
    out=data.OUT;target=out/'training_audit.json'
    assert not target.exists(),'preserve prior audit before versioned repair'
    protocol=json.loads((out/'training_protocol.json').read_text())
    complete=json.loads((out/'training_complete.json').read_text())
    code=[Path(__file__)]
    e.write(out/'training_audit_protocol_before.json',dict(created_utc=e.now(),
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        training_protocol_sha256=e.sha(out/'training_protocol.json'),
        training_complete_sha256=e.sha(out/'training_complete.json'),
        score_tolerance=1e-11,no_new_training=True,no_outer_prediction=True))
    checks=[];records=[];cases=data.load_cases()
    def check(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw))
    for hashes in ('code_hashes','input_hashes'):
        for path,digest in protocol[hashes].items():check('training_dependency_unchanged',e.sha(ROOT/path)==digest,path=path)
    for units in l.contexts():
        table=cases[cases.related_cluster.isin(units)].copy()
        wn=l.weights(table);wp=l.weights(table,paired_only=True);mask,rho=l.observed_pair(table)
        for method in l.METHODS:
            for config in l.configs(method):
                key,path=t.fit_path(units,method,config)
                check('declared_fit_exists',path.exists(),key=key)
                if not path.exists():continue
                fit=json.loads(path.read_text());records.append(fit)
                check('fit_key_context_and_config',fit['key']==key and fit['train_units']==sorted(units)
                    and fit['method']==method and fit['config']==config,key=key)
                check('only_training_condition_labels',fit['training_condition_ids']==sorted(table.sample_id.unique())
                    and fit['training_case_ids']==table.case_id.tolist()
                    and fit['train_families']==sorted(table.family.unique()),key=key)
                check('scenario_rows_not_independent_conditions',fit['scenario_rows']==len(table)
                    and fit['physical_reported_condition_count']==table.sample_id.nunique()
                    and fit['paired_reported_conditions']==table.loc[mask,'sample_id'].nunique(),key=key)
                check('nH_training_weights',np.max(abs(np.asarray(fit['training_nH_case_weights'])-wn))<1e-14,key=key)
                check('rho_training_weights',np.max(abs(np.asarray(fit['training_rho_case_weights'])-wp))<1e-14,key=key)
                if method in l.NONLINEAR:
                    trials=fit['trials'];finite=[r for r in trials if r.get('loss') is not None]
                    initial=[r for r in trials if r['stage']=='initial']
                    refined=[r for r in trials if r['stage']=='uniform_refinement']
                    expected_refined=(2+int(any(r.get('loss') is not None for r in initial))
                        if any(not r['success'] for r in initial) else 0)
                    check('uniform_optimizer_budget',len(initial)==2 and all(r['cap']==120 for r in initial)
                        and len(refined)==expected_refined and all(r['cap']==360 for r in refined)
                        and fit['new_optimizer_starts']==len(trials),key=key)
                    if finite:
                        best=min(finite,key=lambda r:r['loss'])
                        check('minimum_finite_loss_chosen_without_success_cherry_pick',fit['parameters']==best['parameters']
                            and abs(fit['selected_loss']-best['loss'])<1e-12
                            and fit['selected_success']==best['success'],key=key)
                        values=np.asarray(fit['selected_training_values'],float)
                        check('nH_training_score',abs(fit['training_nH_MAE']-wn@abs(values[:,0]-table.target.to_numpy(float)))<1e-11,key=key)
                        check('rho_training_score',abs(fit['training_rho_MAE']-wp[mask]@abs(-values[mask,2]-rho[mask]))<1e-11,key=key)
                        check('separate_data_and_regularized_ranks',fit['observation_jacobian_rank']<=len(fit['parameters'])
                            and fit['regularized_jacobian_rank']<=len(fit['parameters']),key=key)
                        check('selection_replay_count_in_actual_budget',fit['objective_forward_calls']==fit['optimization_objective_forward_calls']+fit['selection_replay_forward_calls']
                            and fit['physical_point_calls']==fit['optimization_physical_point_calls']+fit['selection_replay_physical_point_calls'],key=key)
                    else:check('failed_fit_retained',fit['parameters'] is None and not fit['selected_success'],key=key)
                else:
                    values=(l.direct_values(fit,table) if method=='DirectMultiRidge'
                        else l.empirical_values(method,fit,table))
                    check('closedform_training_values_replay',np.max(abs(values-np.asarray(fit['selected_training_values'])))<1e-11,key=key)
                    check('closedform_budget',fit['new_optimizer_starts']==0 and fit['new_linear_solves']==
                        (2 if method=='DirectMultiRidge' else int(method=='DoseLine')),key=key)
    check('all_planned_records_retained',len(records)==complete['fit_records']==1392)
    check('actual_start_count',sum(r['new_optimizer_starts'] for r in records)==complete['actual_optimizer_starts'])
    check('actual_linear_solve_count',sum(r['new_linear_solves'] for r in records)==complete['new_linear_solves'])
    result=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,
        fit_records=len(records),nonlinear_fit_records=sum(r['method'] in l.NONLINEAR for r in records),
        selected_nonstopped=sum(not r['selected_success'] for r in records),
        invalid_training_fits=sum(not r['training_valid'] for r in records),
        actual_optimizer_starts=sum(r['new_optimizer_starts'] for r in records),
        actual_objective_forward_calls=sum(r.get('objective_forward_calls',0) for r in records),
        actual_physical_point_calls=sum(r.get('physical_point_calls',0) for r in records),
        summed_worker_fit_seconds=sum(r['seconds'] for r in records),
        no_new_training_or_outer_prediction=True,goal_complete=False)
    e.write(target,result)
    print({k:result[k] for k in ('passed','checks','fit_records','selected_nonstopped','invalid_training_fits')},flush=True)
    if not result['passed']:raise RuntimeError('Preserve audit failure before versioned correction')


if __name__=='__main__':main()
