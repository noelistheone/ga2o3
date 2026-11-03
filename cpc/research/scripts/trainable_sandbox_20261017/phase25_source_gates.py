"""Check label-free saved-model replay and mean-loss source adaptation."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.context_saved_model import SavedContextModel,feature_vector,INPUT_NAMES
from sandbox.context_source import solve_context_source
import phase25_training as t
import phase2_energy as e


def inputs(record):
    return SimpleNamespace(**{k:record[k] for k in INPUT_NAMES})


def main():
    checks=[]
    def check(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
    d=t.TABLE.copy();X=np.array([feature_vector(inputs(r)) for r in d.to_dict('records')])
    check('runtime_features_no_DOI_needed',np.max(np.abs(X-t.direct_features(d))),0.)
    row=d[d.sample_id.eq(t.data.LEEDY+'::target-1')].iloc[0]
    r=inputs(row);one=d[d.sample_id.eq(row.sample_id)]
    for method in t.METHODS:
        f=dict(method=method,training_valid=True,selected_success=True)
        if method in t.SPECS:
            spec=t.SPECS[method];n=dict(none=0,constant=1,dose=2,process=4)[spec['basis']]
            f['parameters']=np.r_[18.,.8,np.array([.6,.4,-.2,.1])[:n]].tolist() if spec['kind']=='direct' else np.r_[.5,16.,np.array([.6,.4,-.2,.1])[:n]].tolist()
        elif method=='DirectRidge':
            f.update(coefficients=np.linspace(-.02,.02,10).tolist(),intercept=18.,center=np.zeros(10).tolist(),scale=np.ones(10).tolist())
        else:f.update(offset_dex=-.2,coefficients=[18.,.8] if method=='DoseLine' else None)
        saved=SavedContextModel(f)
        for z in (0.,.2,-.2):
            p=saved.point(r,source_kind='offset',shift=z,require_observation=False)
            expected=t.predict(f,one,source_kind='offset',shift=z,require_observation=False)
            check(method+f'/training_replay/{z}',abs(p.log_nH-expected[0][0]),2e-9)
            base=saved.point(r,source_kind='none',require_observation=False)
            q=saved.offset_point(r,base,z,require_observation=False)
            check(method+f'/offset_algebra/{z}',np.max(np.abs(np.array([p.log_nH,p.log_muH,p.log_sigma])-np.array([q.log_nH,q.log_muH,q.log_sigma]))),2e-9)
            check(method+f'/offset_validity/{z}',float(p.valid!=q.valid),0.)
    y0=np.array([16.,17.,19.]);y=np.array([16.2,17.4]);lams=[None,0.,.1,1.,10.]
    def linear(z):return y0+z,np.ones(3),y0+z>=13.
    result=solve_context_source(linear,[0,1],y,lams,linear=True)
    for sol in result['solutions']:
        lam=sol['lam'];expected=0. if lam is None else .3/(1+lam)
        check('mean_loss_lambda/'+str(lam),abs(sol['raw_shift']-expected),1e-10)
        check('mean_loss_value/'+str(lam),abs(sol['objective']-(np.mean((y0[:2]+expected-y)**2)+(0 if lam is None else lam*expected**2))),1e-10)
    # This is a conditional synthetic recovery test, not new measured data.
    f=dict(method='InventoryProcess',parameters=[.5,16.,.6,.4,-.2,.1],training_valid=True,selected_success=True)
    model=SavedContextModel(f)
    nodes=[inputs(r) for r in d[d.family.eq(t.data.LEEDY)].sort_values('incorporated_cm3').to_dict('records')]
    cache={}
    def actual(z):
        if z not in cache:
            a=[model.point(r,source_kind='field',shift=z,require_observation=False) for r in nodes]
            cache[z]=(np.array([p.log_nH for p in a]),np.array([p.source_derivative for p in a]),np.array([p.valid for p in a]))
        return cache[z]
    target=actual(.3)[0][[0,2]]
    result=solve_context_source(actual,[0,2],target,[None,0.,.1],linear=False)
    zero=next(s for s in result['solutions'] if s['lam']==0.)
    check('conditional_synthetic_field_recovery',abs(zero['raw_shift']-.3),2e-5)
    check('conditional_synthetic_fit_residual',zero['objective'],1e-12)
    check('candidate_local_search_cap',max(len(s['searches']) for s in result['solutions'])-3,0.)
    paths=[Path(__file__),ROOT/'src/sandbox/context_saved_model.py',ROOT/'src/sandbox/context_source.py',ROOT/'src/sandbox/functional_source.py']
    value=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,query_labels_in_runtime_inputs=False,
        synthetic_recovery_is_material_validation=False,goal_complete=False,code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths})
    e.write(t.OUT/'source_gates.json',value)
    print('phase25 saved/source gates',value['passed'],len(checks),flush=True)
    if not value['passed']:raise RuntimeError('Preserve failed source gate before repair')


if __name__=='__main__':main()
