"""Verify reference growth chains and complete training losses before fitting."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.context_saved_model import INPUT_NAMES
import phase27_training as t
import phase2_energy as e


def main():
    assert not (t.OUT/'learning_gates.json').exists(),'Keep the first learning-gate outcome'
    d=t.TABLE;table=d.loc[[d[d.element.eq('Si')&d.H_known.eq(True)].index[0],d[d.element.eq('Sn')].index[0],
        d[d.sample_id.eq('10.1063/1.5047214::target-1')].index[0]]].copy()
    declare_code=[Path(__file__),ROOT/'scripts/trainable_sandbox_20261017/phase27_training.py',
        ROOT/'src/sandbox/charge_feasible_model_v2.py',ROOT/'src/sandbox/charge_feasible_model.py',ROOT/'src/sandbox/charge_feasible_share.py']
    e.write(t.OUT/'learning_protocol_before.json',dict(created_utc=e.now(),
        purpose='complete objective/gradient and inherited-label-budget check;no fitting',
        fixed_parameters='Tf_logit.5,bg16,theta[1.4,.2,-.1,.1] truncated bybasis',
        synthetic_label_offsets=[.05,-.03,.10],gamma=.1,difference_step=1e-4,tolerance=3e-5,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in declare_code}))
    checks=[];weights=t.parent.weights(table,'DOI');synthetic_offset=np.array([.05,-.03,.10]);gamma=.1
    def check(name,x,tol):checks.append(dict(name=name,value=float(x),tolerance=tol,passed=bool(x<=tol)))
    for method in t.METHODS:
        m=t.model(method);n=len(m.parameter_names);s=t.SPECS[method]
        w=np.r_[.5,16.,np.array([1.4,.2,-.1,.1])[:n-2]]
        values,J,delta,dJ,valid=t.forward(method,w,table,penalty=True);target=values+synthetic_offset
        def loss(x):
            a,b,c,d,e=t.forward(method,x,table,penalty=True)
            return float((a-target)**2@weights+gamma*c**2@weights)
        gradient=2*J.T@((values-target)*weights)+2*gamma*dJ.T@(delta*weights)
        numeric=[]
        for j in range(n):
            h=np.zeros(n);h[j]=1e-4;numeric.append((loss(w+h)-loss(w-h))/.0002)
        check(method+'/complete_penalized_gradient',np.max(abs(np.array(numeric)-gradient)/(1+abs(gradient))),3e-5)
        row=SimpleNamespace(**{k:getattr(next(table.itertuples()),k) for k in INPUT_NAMES})
        p=m.point(row,w,require_observation=False);ana=p.input_jacobian[0]
        num=[]
        for name in p.input_names:
            plus=SimpleNamespace(**vars(row));minus=SimpleNamespace(**vars(row))
            if name=='growth_T_C':h=.02;plus.growth_T_C+=h;minus.growth_T_C-=h
            else:
                h=1e-4;k={'log10_dose_cm3':'incorporated_cm3','log10_Tobs_K':'Tobs_K','log10_H_cm3':'H_cm3'}[name]
                setattr(plus,k,getattr(plus,k)*10**h);setattr(minus,k,getattr(minus,k)*10**(-h))
            num.append((m.point(plus,w,require_observation=False).values[0]-m.point(minus,w,require_observation=False).values[0])/(2*h))
        check(method+'/input_chains',np.max(abs(np.array(num)-ana)/(1+abs(ana))),3e-5)
        lo,hi=t.bounds(method);check(method+'/interior_test_parameters',float(not ((w>lo)&(w<hi)).all()),0.)
    for scheme in ('DOI','cluster'):
        for units in t.parent.contexts(scheme):
            d=t.parent.subset(scheme,units);wg=t.parent.weights(d,scheme)
            check(scheme+'/'+t.parent.digest(units)+'/unit_weight',abs(wg.sum()-1),1e-12)
            ref,path=t.inherited_reference(scheme,units)
            check(scheme+'/'+t.parent.digest(units)+'/reference_training_ids',float(ref['training_obs_ids']!=d.obs_id.tolist()),0.)
    code=[Path(__file__),ROOT/'scripts/trainable_sandbox_20261017/phase27_training.py',ROOT/'src/sandbox/pool_share_controls.py',
        ROOT/'src/sandbox/charge_feasible_model_v2.py',ROOT/'src/sandbox/charge_feasible_model.py',ROOT/'src/sandbox/charge_feasible_share.py']
    value=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,methods=len(t.METHODS),
        maximum_derivative_error=max(c['value'] for c in checks if c['name'].endswith(('/input_chains','/complete_penalized_gradient'))),
        synthetic_targets_only_in_numerical_loss_check=True,new_supervised_fits=0,
        gradient_scope='complete response penalty includes reference gradient;synthetic loss targets,not a shared fit',goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code})
    e.write(t.OUT/'learning_gates.json',value);print('phase27 learning gates',value['passed'],len(checks),value['maximum_derivative_error'],flush=True)
    if not value['passed']:raise RuntimeError('Preserve learning gate failure before repair')


if __name__=='__main__':main()
