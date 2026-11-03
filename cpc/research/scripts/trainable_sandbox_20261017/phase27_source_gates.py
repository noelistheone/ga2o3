"""Verify the new label-free callback and unchanged finite-design source loss."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.charge_feasible_saved_model import SavedFeasibleModel as SavedPoolModel
from sandbox.context_saved_model import INPUT_NAMES
from sandbox.context_source import solve_context_source
import phase27_training as t
import phase2_energy as e


def node(r):return SimpleNamespace(**{k:r[k] for k in INPUT_NAMES})


def fake(method):
    s=t.SPECS[method];n=len(t.model(method).parameter_names)
    w=np.r_[.5,16.,np.array([1.4,.4,-.2,.1])[:n-2]]
    return dict(method=method,evaluation_method=method,parameters=w.tolist(),training_valid=True,selected_success=True)


def main():
    assert not (t.OUT/'source_gates.json').exists(),'Keep previous source-gate outputs intact'
    declare=[Path(__file__),ROOT/'src/sandbox/charge_feasible_saved_model.py',ROOT/'src/sandbox/context_source.py',
        ROOT/'src/sandbox/charge_feasible_model_v2.py',ROOT/'scripts/trainable_sandbox_20261017/phase27_training.py']
    e.write(t.OUT/'source_protocol_before.json',dict(created_utc=e.now(),purpose='savedcallbacks and unchanged finite-design source solve',
        families=list(t.METHODS),fallbacks=['FeasibleProcess','EnvelopeProcess'],synthetic_source_shift=.3,
        synthetic_anchors=[0,2],source_lambdas=[None,0.,.1],no_query_labels=True,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in declare}))
    checks=[]
    def check(name,x,tol):checks.append(dict(name=name,error=float(x),tolerance=tol,passed=bool(x<=tol)))
    table=t.TABLE[t.TABLE.sample_id.eq('10.1063/1.5047214::target-1')];row=node(table.iloc[0])
    for method in t.METHODS:
        fit=fake(method);saved=SavedPoolModel(fit);w=np.array(fit['parameters'])
        for z in (0.,.2,-.2):
            p=saved.point(row,source_kind='offset',shift=z,require_observation=False)
            base=saved.point(row,require_observation=False);q=saved.offset_point(row,base,z,require_observation=False)
            expected=t.forward(method,w,table)[0][0]+z
            check(method+'/'+str(z)+'/training_forward',abs(p.log_nH-expected),2e-8)
            check(method+'/'+str(z)+'/offset_properties',np.max(abs(np.array([p.log_nH,p.log_muH,p.log_sigma])-
                np.array([q.log_nH,q.log_muH,q.log_sigma]))),2e-8)
            check(method+'/'+str(z)+'/offset_domain',float(p.valid!=q.valid),0.)
    for display,actual,w in [('FeasibleProcess','ParentReference',[.5,16.]),('EnvelopeProcess','ParentReference',[.5,16.])]:
        saved=SavedPoolModel(dict(method=display,evaluation_method=actual,parameters=w,training_valid=True,selected_success=True,reference_fallback=True))
        p=saved.point(row,require_observation=False);expected=t.forward(actual,w,table)[0][0]
        check(display+'/actual_fallback_parameter_shape',abs(p.log_nH-expected),2e-8)
    for method in ('FeasibleProcess','EnvelopeProcess'):
        saved=SavedPoolModel(fake(method));rows=t.TABLE[t.TABLE.family.eq('10.1063/1.5047214')].sort_values('incorporated_cm3')
        nodes=[node(r) for r in rows.to_dict('records')];cache={}
        def forward(z):
            z=float(z)
            if z not in cache:
                p=[saved.point(r,source_kind='field',shift=z,require_observation=False) for r in nodes]
                cache[z]=(np.array([a.log_nH for a in p]),np.array([a.source_derivative for a in p]),np.array([a.valid for a in p]))
            return cache[z]
        target=forward(.3)[0][[0,2]]
        result=solve_context_source(forward,[0,2],target,[None,0.,.1],linear=False)
        fitted=next(r for r in result['solutions'] if r['lam']==0.)
        check(method+'/synthetic_source_recovery',abs(fitted['raw_shift']-.3),2e-5)
        check(method+'/synthetic_source_objective',fitted['objective'],1e-12)
        check(method+'/source_design_all_base_valid',float(not forward(0.)[2].all()),0.)
        check(method+'/point_ceiling',max(0,result['source_point_calls']-32768),0.)
    paths=[Path(__file__),ROOT/'src/sandbox/charge_feasible_saved_model.py',ROOT/'src/sandbox/charge_feasible_model_v2.py',
        ROOT/'src/sandbox/pool_share_controls.py',ROOT/'src/sandbox/context_source.py',ROOT/'src/sandbox/functional_source.py']
    value=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,
        synthetic_fit_flags_are_numerical_test_fixtures=True,synthetic_recovery_is_material_validation=False,
        new_shared_training=False,goal_complete=False,code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths})
    e.write(t.OUT/'source_gates.json',value);print('phase27 saved/source gates',value['passed'],len(checks),flush=True)
    if not value['passed']:raise RuntimeError('Preserve source-gate failure before repair')


if __name__=='__main__':main()
