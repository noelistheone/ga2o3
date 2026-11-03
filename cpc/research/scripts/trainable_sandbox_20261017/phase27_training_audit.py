"""Check complete split/budget/objective records, retaining invalid candidates."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase27_training as t
import phase2_energy as e


def main():
    done=json.loads((t.OUT/'training_complete.json').read_text());protocol=json.loads((t.OUT/'training_protocol.json').read_text())
    for key in ('code_hashes','input_hashes','inherited_parent_fit_sha256'):
        for name,h in protocol[key].items():assert e.sha(ROOT/name)==h,name
    checks=[];records=[]
    def check(name,x,tol=0.):checks.append(dict(name=name,error=float(x),tolerance=tol,passed=bool(x<=tol)))
    for scheme in ('DOI','cluster'):
        for units in t.parent.contexts(scheme):
            table=t.parent.subset(scheme,units);wg=t.parent.weights(table,scheme)
            for method in t.METHODS:
                for reg in t.configs(method):
                    key,path=t.path_for(scheme,units,method,reg);r=json.loads(path.read_text());records.append(r)
                    check(key+'/key',float(r['key']!=key))
                    check(key+'/split',float(r['scheme']!=scheme or r['train_units']!=sorted(units)))
                    check(key+'/families',float(r['train_families']!=sorted(table.family.unique())))
                    check(key+'/training_label_ids',float(r['training_obs_ids']!=table.obs_id.tolist()))
                    check(key+'/method_and_reg',float(r['method']!=method or r['reg']!=reg))
                    actual=r['evaluation_method'];is_ref=method.endswith('Reference');fallback=not is_ref and reg is None
                    check(key+'/fallback_rule',float(bool(r.get('reference_fallback'))!=fallback))
                    if fallback:
                        refmethod=t.refmethod(method)
                        if refmethod=='ParentReference':ref,rp=t.inherited_reference(scheme,units)
                        else:_,rp=t.path_for(scheme,units,refmethod,None);ref=json.loads(rp.read_text())
                        check(key+'/actual_reference_model',float(actual!=refmethod))
                        check(key+'/reference_path',float(r['reference_fit_path']!=str(rp.relative_to(ROOT))))
                        check(key+'/reference_checksum',float(r['reference_fit_sha256']!=e.sha(rp)))
                        check(key+'/reference_parameters',np.max(abs(np.asarray(r['parameters'])-ref['parameters'])))
                        check(key+'/no_new_reference_optimizer',r['new_optimizer_starts'])
                    else:
                        check(key+'/actual_learner',float(actual!=method))
                        initial=[x for x in r['trials'] if x['stage']=='initial'];refined=[x for x in r['trials'] if x['stage']=='uniform_refinement']
                        check(key+'/initial_count',abs(len(initial)-2));check(key+'/initial_caps',sum(x['cap']!=120 for x in initial))
                        check(key+'/refinement_trigger',float(bool(refined)!=any(not x['success'] for x in initial)))
                        finite_initial=[x for x in initial if x['loss'] is not None]
                        check(key+'/refinement_count',abs(len(refined)-(2+bool(finite_initial) if refined else 0)))
                        check(key+'/refinement_caps',sum(x['cap']!=360 for x in refined))
                        check(key+'/actual_optimizer_starts',abs(r['new_optimizer_starts']-len(r['trials'])))
                        finite=[x for x in r['trials'] if x['loss'] is not None]
                        if finite:
                            best=min(finite,key=lambda x:x['loss'])
                            check(key+'/selected_minimum_trial',np.max(abs(np.asarray(r['parameters'])-best['parameters'])))
                            check(key+'/selected_stop_flag',float(r['selected_success']!=best['success']))
                            mse=r['training_nH_MSE'];total=mse+(0. if reg is None else reg*r['rms_response_change']**2)
                            # The nominal-growth reference has a measured
                            # cancellation floor. This checks metadata-level
                            # objective consistency; the independent121gates
                            # validate the derivative, not a global minimum.
                            check(key+'/penalized_loss',abs(total-r['selected_loss'])/(1+abs(total)),2e-6)
                        else:check(key+'/failed_trials_not_servable',float(r['parameters'] is not None or r['selected_success'] or r['training_valid']))
                    if r['parameters'] is not None:
                        check(key+'/actual_parameter_count',abs(len(r['parameters'])-len(t.model(actual).parameter_names)))
                        v=np.asarray(r['selected_training_values']);mse=float((v-table.target.to_numpy(float))**2@wg)
                        mae=float(abs(v-table.target.to_numpy(float))@wg)
                        check(key+'/stored_mse',abs(mse-r['training_nH_MSE']),1e-11)
                        check(key+'/stored_mae',abs(mae-r['training_nH_MAE']),1e-11)
                        check(key+'/domain_flag',float(r['training_valid']!=all(r['selected_training_valid'])))
    check('all_planned_records',abs(len(records)-912));check('reported_count',abs(done['fit_records']-len(records)))
    check('no_unplanned_fits',abs(len(list((t.OUT/'fits').glob('*.json')))-len(records)))
    starts=sum(r['new_optimizer_starts'] for r in records);check('reported_optimizer_starts',abs(starts-done['actual_optimizer_starts']))
    value=dict(passed=all(x['passed'] for x in checks),checks=len(checks),rows=checks,fit_records=len(records),
        actual_optimizer_starts=starts,actual_nonlinear_fits=sum(bool(r['trials']) for r in records),
        initial_not_stopped=sum(not a['success'] for r in records for a in r['trials'] if a['stage']=='initial'),
        selected_not_stopped=sum(not r['selected_success'] for r in records),training_invalid=sum(not r['training_valid'] for r in records),
        invalid_candidates_retained=True,invalid_is_not_integrity_failure=True,
        no_physical_truth_global_minimum_or_prediction_gain_certified=True,goal_complete=False,
        code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))})
    e.write(t.OUT/'training_audit.json',value);print('phase27 training audit',value['passed'],len(checks),len(records),flush=True)
    if not value['passed']:raise RuntimeError('Preserve complete training audit failure before repair')


if __name__=='__main__':main()
