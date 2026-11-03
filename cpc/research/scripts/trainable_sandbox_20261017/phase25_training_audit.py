"""Check all saved training splits, objectives and declared optimizer budgets."""
from pathlib import Path
import sys,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase25_training as t
import phase2_energy as e


def main():
    done=json.loads((t.OUT/'training_complete.json').read_text())
    protocol=json.loads((t.OUT/'training_protocol.json').read_text())
    for key in ('code_hashes','input_hashes'):
        for name,h in protocol[key].items():assert e.sha(ROOT/name)==h,name
    plan=json.loads((t.OUT/'training_started.json').read_text())
    records=[];checks=[]
    def check(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
    for scheme,train,method,reg in plan['jobs']:
        key,path=t.fit_path(scheme,train,method,reg)
        f=json.loads(path.read_text());records.append(f);tr=t.subset(scheme,train)
        check(key+'/train_ids',float(set(f['training_obs_ids'])!=set(tr.obs_id)),0.)
        check(key+'/families',float(f['train_families']!=sorted(tr.family.unique())),0.)
        check(key+'/split',float(f['train_units']!=sorted(train) or f['scheme']!=scheme),0.)
        check(key+'/key',float(key!=f['key']),0.)
        if method in t.SPECS:
            n=2+dict(none=0,constant=1,dose=2,process=4)[t.SPECS[method]['basis']]
            if f['parameters'] is not None:check(key+'/parameter_count',abs(len(f['parameters'])-n),0.)
            if f.get('exact_reference_fallback'):
                refmethod='DoseLine' if t.SPECS[method]['kind']=='direct' else 'FixedPhysical'
                _,rp=t.fit_path(scheme,train,refmethod,None);ref=json.loads(rp.read_text())
                pars=ref['coefficients'] if refmethod=='DoseLine' else ref['parameters']
                check(key+'/reference_parameters',np.max(np.abs(np.asarray(f['parameters'])-np.r_[pars,np.zeros(n-2)])),0.)
                check(key+'/no_extra_optimizer',f['new_optimizer_starts'],0.)
            else:
                initial=[r for r in f['trials'] if r['stage']=='initial'];refined=[r for r in f['trials'] if r['stage']=='uniform_refinement']
                check(key+'/initial_budget',abs(len(initial)-2)+sum(r['cap']!=120 for r in initial),0.)
                check(key+'/refinement_trigger',float(bool(refined)!=any(not r['success'] for r in initial)),0.)
                check(key+'/refinement_caps',sum(r['cap']!=360 for r in refined),0.)
                check(key+'/actual_starts',abs(f['new_optimizer_starts']-len(f['trials'])),0.)
                finite=[r for r in f['trials'] if r['loss'] is not None]
                if finite:
                    best=min(finite,key=lambda r:r['loss'])
                    check(key+'/selected_trial',np.max(np.abs(np.asarray(f['parameters'])-best['parameters'])),0.)
                    v=np.asarray(f['selected_training_values']);wg=t.weights(tr,scheme)
                    mse=float(((v-tr.target.to_numpy(float))**2)@wg)
                    check(key+'/weighted_observed_loss',abs(mse-f['training_nH_MSE']),1e-10)
                    total=mse+(0. if reg is None else reg*f['rms_response_change']**2)
                    check(key+'/penalized_objective',abs(total-f['selected_loss']),1e-8)
                    check(key+'/domain_flag',float(f['training_valid']!=all(f['selected_training_valid'])),0.)
        elif method=='DirectRidge':
            wg=t.weights(tr,scheme);x=t.direct_features(tr);mean=np.average(x,axis=0,weights=wg)
            scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=wg));scale[scale<1e-12]=1.
            check(key+'/training_only_mean',np.max(np.abs(mean-f['center'])),1e-12)
            check(key+'/training_only_scale',np.max(np.abs(scale-f['scale'])),1e-12)
    check('all_planned_records',abs(len(records)-1596),0.)
    check('completion_count',abs(len(records)-done['fit_records']),0.)
    check('no_unplanned_fit_files',abs(len(list((t.OUT/'fits').glob('*.json')))-len(records)),0.)
    value=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,fit_records=len(records),
        actual_optimizer_starts=sum(f['new_optimizer_starts'] for f in records),
        actual_nonlinear_fits=sum(bool(f['trials']) for f in records),
        initial_not_stopped=sum(not r['success'] for f in records for r in f['trials'] if r['stage']=='initial'),
        selected_not_stopped=sum(not f.get('selected_success',True) for f in records),
        training_invalid=sum(not f['training_valid'] for f in records),
        continuation_trial_states=sum(f.get('numerical_continuation_trial_states',0) for f in records),
        physical_truth_or_global_minima_certified=False,goal_complete=False,
        code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))})
    e.write(t.OUT/'training_audit.json',value)
    print('phase25 training audit',value['passed'],len(checks),'records',len(records),flush=True)
    if not value['passed']:raise RuntimeError('Retain failed split/objective/budget audit')


if __name__=='__main__':main()
