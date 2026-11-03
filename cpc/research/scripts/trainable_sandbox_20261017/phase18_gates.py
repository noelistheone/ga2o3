"""Gate the deployed log-odds state, input, source and training mappings."""
from pathlib import Path
from types import SimpleNamespace
from itertools import combinations
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.special import expit
import phase18_prediction as study
import phase2_energy as energy


def main():
    checks=[];errors=[];clock=time.perf_counter()
    def add(name,passed,**kwargs):checks.append(dict(name=name,passed=bool(passed),**kwargs))
    def derivative(name,fd,analytic,**kw):
        error=float(np.max(abs(fd-analytic)/(1+abs(fd))));errors.append(error);add(name,error<5e-6,error=error,**kw)
    def rich(fun,x,j,h):
        slopes=[]
        for step in (h,h/2):
            p=np.array(x,float);n=np.array(x,float);p[j]+=step;n[j]-=step
            slopes.append((fun(p)-fun(n))/(2*step))
        return (4*slopes[1]-slopes[0])/3
    cases=[dict(element='Si',incorporated_cm3=1e18,H_known=False,H_cm3=0.,Tobs_K=40.,growth_T_C=523.,thermal_cap_K=796.15),
        dict(element='Sn',incorporated_cm3=3e18,H_known=False,H_cm3=0.,Tobs_K=300.,growth_T_C=900.,thermal_cap_K=1173.15),
        dict(element='Si',incorporated_cm3=5.4e17,H_known=True,H_cm3=1.8e17,Tobs_K=150.,growth_T_C=830.,thermal_cap_K=1103.15)]
    for method,d in study.STRUCTURES.items():
        if 'empirical' in d:continue
        model=study.mapper(method);w=np.array(study.specification(method)[2][0]);w[-2:]=[-1.2,-.7] if d['closure']=='inventory' else [.3,.2]
        for case in cases:
            row=SimpleNamespace(**case);a=model.point(row,w)
            def combined(v,rr=row,kind='none',z=0.):
                q=model.point(rr,v,source_kind=kind,source_shift=z);return np.r_[q.values,q.transport_values]
            for j,name in enumerate(model.parameter_names):
                h=.04 if d['closure']=='affine_output' and j<model.ia else .002
                fd=rich(combined,w,j,h)
                derivative('actual_shared_state',fd[:3],a.jacobian[:,j],method=method,parameter=name,T=row.Tobs_K)
                derivative('actual_shared_transport',fd[3:],a.transport_jacobian[:,j],method=method,parameter=name,T=row.Tobs_K)
            fields=['incorporated_cm3','thermal_cap_K','Tobs_K']+(['H_cm3'] if row.H_known else [])
            iv=np.log10([getattr(row,f) for f in fields])
            def inputs(v):
                r=SimpleNamespace(**case)
                for field,value in zip(fields,v):setattr(r,field,10**value)
                return combined(w,rr=r)
            for j,name in enumerate(a.input_names):
                fd=rich(inputs,iv,j,.005 if d['closure']=='affine_output' else .002)
                derivative('actual_input_state',fd[:3],a.input_jacobian[:,j],method=method,input=name,T=row.Tobs_K)
                derivative('actual_input_transport',fd[3:],a.transport_input_jacobian[:,j],method=method,input=name,T=row.Tobs_K)
            for kind in (['offset','logodds'] if d['closure']=='inventory' else ['offset']):
                z=.03;p=model.point(row,w,source_kind=kind,source_shift=z)
                fd=rich(lambda v:combined(w,kind=kind,z=v[0]),[z],0,.002)
                derivative('actual_source_state',fd[:3],p.source_jacobian,method=method,kind=kind)
                derivative('actual_source_transport',fd[3:],p.transport_source_jacobian,method=method,kind=kind)
                zero=model.point(row,w,source_kind=kind,source_shift=0.)
                fresh=model.point(row,w)
                zero_error=float(np.max(abs(np.r_[zero.values-fresh.values,zero.transport_values-fresh.transport_values])))
                warm_error=float(np.max(abs(np.r_[fresh.values-a.values,fresh.transport_values-a.transport_values])))
                add('zero_source_shared_replay',zero_error<1e-9,error=zero_error,method=method,kind=kind)
                add('warm_path_numerical_replay',warm_error<2e-6,error=warm_error,method=method,kind=kind,
                    note='separate numerical-start drift;2e-6 frozenphase17 runtime criterion,not relaxation of zero-interface criterion')
                add('source_sigma_derivative_identity',abs(p.transport_source_jacobian[1]-p.source_jacobian[0]-p.transport_source_jacobian[0])<1e-10,method=method,kind=kind)
            add('sigma_shared_derivative_identity',max(abs(a.transport_jacobian[1]-a.jacobian[0]-a.transport_jacobian[0]))<1e-10,method=method)
            add('sigma_input_derivative_identity',max(abs(a.transport_input_jacobian[1]-a.input_jacobian[0]-a.transport_input_jacobian[0]))<1e-10,method=method)
            # A direct same-coordinate inner replay checks wrapper semantics.
            ww=w.copy();ww[-1]=0.
            v=np.r_[ww[:model.nbase],np.logaddexp(0.,ww[-2])] if d['closure']=='inventory' else ww[:model.nbase]
            b=model.point(row,ww);inner=model.inner.point(row,v)
            if d['closure']=='inventory':
                add('constant_allocation_inner_replay',max(abs(b.values-inner.values))<1e-10 and max(abs(b.transport_values-inner.transport_values))<1e-10,method=method)
                for g in (-20.,0.,6.):
                    pp=w.copy();pp[-2:]=[g,0.];p=model.point(row,pp);layer,solver=model.layer_for(row)
                    actual=layer.atoms.T@solver.last_prediction.populations
                    expected=np.array([row.incorporated_cm3,row.H_cm3] if row.H_known else [row.incorporated_cm3])
                    add('new_allocation_foreign_atoms',max(abs(actual/expected-1))<3e-8,method=method,g=g)
                    add('new_allocation_probability_charge',p.diagnostics['parent_probability_error']<2e-12 and p.diagnostics['cold_Q_EF_cm3_per_eV']<0,method=method,g=g)
                    add('new_allocation_positive',min(solver.last_inventory.totals)>=0 and p.diagnostics['foreign_atom_derivative_error']<3e-7,method=method,g=g)
            else:
                ww[-2:]=0.;b=model.point(row,ww)
                add('zero_affine_exact_inner_replay',max(abs(b.values-inner.values))<1e-10 and max(abs(b.transport_values-inner.transport_values))<1e-10,method=method)
    for method,d in study.STRUCTURES.items():
        if 'empirical' not in d:continue
        tr=study.training_data(study.FAMILIES,method);w=np.array(study.specification(method)[2][0]);values,J,Z=study.activation(w,tr,d['empirical'])
        for j in range(3):derivative('direct_activation_shared_gradient',rich(lambda v:study.activation(v,tr,d['empirical'])[0],w,j,.002),J[:,j],method=method,j=j)
        derivative('direct_activation_source_gradient',rich(lambda v:study.activation(w,tr,d['empirical'],source_shift=v[0])[0],[0.],0,.002),Z,method=method)
        zero=study.activation(w,tr,d['empirical'],source_shift=0.)[0]
        add('direct_source_zero_exact',np.array_equal(values,zero),method=method)
        if d['empirical']=='activation_product':
            old,_=study.prior.study.activation(w,tr)
            add('old_product_function_replay',max(abs(old-values))<1e-12,method=method)
    for fam in (f for n in (3,4) for f in combinations(study.FAMILIES,n)):
        for method in study.STRUCTURES:
            tr=study.training_data(fam,method)
            add('training_ownT_members_and_labels',set(tr.family)<=set(fam) and tr.endpoint.eq('nH').all() and tr.obs_id.is_unique and tr[tr.kind.eq('curve')].Tobs_K.le(275).all(),method=method)
            for kind,g in tr.groupby('kind'):
                weight=1. if kind=='RT' else .1
                add('training_DOI_block_weights',abs(g.loss_weight.sum()-weight)<1e-12 and np.ptp(g.groupby('family').loss_weight.sum().to_numpy())<1e-12,method=method,kind=kind)
    for g in (-20.,-2.,0.,2.,20.):
        S=np.logaddexp(0.,g);alpha=expit(g)
        add('allocation_logit_identity',abs(-np.expm1(-S)-alpha)<2e-16,g=g)
    add('new_slope_exceeds_old_shape_limit',3*.5/np.log(10.)>1/np.log(10.))
    paths=(Path(__file__),ROOT/'scripts/trainable_sandbox_20261017/phase18_prediction.py',ROOT/'src/sandbox/logodds_inventory_model.py',
        ROOT/'docs/ML_SANDBOX_LOGODDS_PREDICTION_20261018.md',ROOT/'src/sandbox/inventory_coherent_model.py',ROOT/'src/sandbox/fixed_inventory_charge.py')
    payload=dict(created_utc=energy.now(),passed=all(c['passed'] for c in checks),checks=checks,
        maximum_scaled_derivative_error=max(errors),seconds=time.perf_counter()-clock,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},
        scope='actualshared/input/source mapping andtrainingcontracts;notmaterial prediction validation',
        finite_difference='fourth-orderRichardson,5e-6criterion;known outputreference widerstencils fromfrozenphase17',
        initial_gate='2zero-interfacechecks mixed stale/warm hot roots;retained;consecutive identical-state zero remains1e-9,warm-path replay separately2e-6',goal_complete=False)
    energy.write(study.OUT/'numerical_gates.json',payload)
    print(json.dumps(dict(passed=payload['passed'],checks=len(checks),maximum_scaled_derivative_error=max(errors),seconds=payload['seconds'],
        failures=[c for c in checks if not c['passed']]),indent=2))
    if not payload['passed']:raise RuntimeError('New log-odds gate failed;do nottrain')


if __name__=='__main__':main()
