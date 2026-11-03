"""Actual-map and data/loss contract gates before joint property training."""
from pathlib import Path
from types import SimpleNamespace
from itertools import combinations
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.coherent_hall_model import CoherentHallModel
from sandbox.thermal_coherent_model import ThermalCoherentHallModel
from sandbox.coherent_fermi import published_coherent_profile
import phase16_prediction as study
import phase2_energy as energy


def main():
    checks=[];errors=[];clock=time.perf_counter()
    def add(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw))
    def derivative(name,fd,analytic,**kw):
        error=float(np.max(abs(fd-analytic)/(1+abs(fd))));errors.append(error)
        add(name,error<3e-6,error=error,**kw)
    def richardson(fun,x,j,h):
        slopes=[]
        for step in (h,h/2):
            p=np.array(x,float);n=np.array(x,float);p[j]+=step;n[j]-=step
            slopes.append((fun(p)-fun(n))/(2*step))
        return (4*slopes[1]-slopes[0])/3
    profile=published_coherent_profile()
    for ref in study.REFERENCES:
        definition=study.STRUCTURES[ref];old=CoherentHallModel(definition,profile=profile,pO2_atm=1e-5)
        effective=study.mapper(ref)
        cap_name=next(m for m,d in study.STRUCTURES.items() if d['reference_method']==ref and d['thermal_mode']=='record_cap' and d['gamma']==0)
        cap=study.mapper(cap_name)
        for element,C,H,T,upper in (('Si',1e18,0.,40.,796.15),('Sn',3e18,0.,300.,1173.15),('Si',5.4e17,1.8e17,150.,1103.15)):
            row=SimpleNamespace(element=element,incorporated_cm3=C,growth_T_C=upper-273.15,Tobs_K=T,H_known=H>0,H_cm3=H,thermal_cap_K=upper)
            w=np.asarray(study.core.specification(ref)[2][0]);a=effective.point(row,w);o=old.point(row,w)
            add('effective_frozen_reference_equal',max(abs(a.values-o.values).max(),abs(a.transport_values-o.transport_values).max())<1e-9,
                reference=ref,element=element,T=T)
            add('effective_reference_gradient_equal',max(abs(a.jacobian-o.jacobian).max(),abs(a.transport_jacobian-o.transport_jacobian).max())<1e-9,reference=ref)
            r=cap.point(row,w)
            add('capped_temperature_in_declared_interval',600<r.diagnostics['Tf_K']<upper,reference=ref,Tf=r.diagnostics['Tf_K'],cap=upper)
            for source in ('none','stock','Tf','offset'):
                z=0. if source=='none' else .2;r=cap.point(row,w,source_kind=source,source_shift=z)
                for j in range(len(w)):
                    def fun(v):
                        p=cap.point(row,v,source_kind=source,source_shift=z)
                        return np.r_[p.values,p.transport_values]
                    # Strong compensation loses finite-difference digits at tiny steps.
                    # Fourth-order stencils resolve the derivative without relaxing its gate.
                    fd=richardson(fun,w,j,.04 if j==2 else .08)
                    derivative('full_capped_parameter_chain',fd[:3],r.jacobian[:,j],ref=ref,source=source,j=j,T=T,element=element)
                    derivative('full_capped_transport_chain',fd[3:],r.transport_jacobian[:,j],ref=ref,source=source,j=j,T=T,element=element)
                def input_fun(v):
                    r2=SimpleNamespace(**vars(row));r2.thermal_cap_K=10**v[0]
                    p=cap.point(r2,w,source_kind=source,source_shift=z)
                    return np.r_[p.values,p.transport_values]
                fd=richardson(input_fun,[np.log10(upper)],0,.005)
                derivative('cap_loginput_carrier_chain',fd[:3],r.cap_jacobian,ref=ref,source=source,T=T,element=element)
                derivative('cap_loginput_transport_chain',fd[3:],r.transport_cap_jacobian,ref=ref,source=source,T=T,element=element)
                if source!='none':
                    def source_fun(v):
                        p=cap.point(row,w,source_kind=source,source_shift=v[0])
                        return np.r_[p.values,p.transport_values]
                    fd=richardson(source_fun,[z],0,.08)
                    derivative('capped_source_carrier_chain',fd[:3],r.source_jacobian,ref=ref,source=source,T=T,element=element)
                    derivative('capped_source_transport_chain',fd[3:],r.transport_source_jacobian,ref=ref,source=source,T=T,element=element)
                    current=cap.point(row,w);zero=cap.point(row,w,source_kind=source,source_shift=0.)
                    add('source_zero_update',max(abs(current.values-zero.values).max(),abs(current.transport_values-zero.transport_values).max())<1e-10,ref=ref,source=source)
            at_limit=SimpleNamespace(**vars(row));at_limit.thermal_cap_K=1950.
            r=cap.point(at_limit,w);o=old.point(at_limit,w)
            add('cap1950_recovers_old_map',max(abs(r.values-o.values).max(),abs(r.transport_values-o.transport_values).max())<1e-9,ref=ref)
            add('cap1950_recovers_old_gradient',max(abs(r.jacobian-o.jacobian).max(),abs(r.transport_jacobian-o.transport_jacobian).max())<1e-8,ref=ref)
            for invalid in (600.,599.,1951.,np.nan):
                bad=SimpleNamespace(**vars(row));bad.thermal_cap_K=invalid
                try:cap.point(bad,w);rejected=False
                except ValueError:rejected=True
                add('invalid_cap_rejected',rejected,cap=invalid)
    contexts=[f for n in (3,4) for f in combinations(study.core.FAMILIES,n)]
    no_mu=[]
    for families in contexts:
        for method,d in study.STRUCTURES.items():
            tr=study.training_data(families,method)
            add('training_members_and_ownT',set(tr.family)<=set(families) and tr.obs_id.is_unique and
                tr[tr.kind.eq('curve')].Tobs_K.le(275).all(),method=method,train_families=families)
            expected={'nH_RT':1.,'nH_curve':.1 if d['training']=='joint' else 0.,'muH_RT':d['gamma'],'muH_curve':.1*d['gamma']}
            for block,g in tr.groupby('loss_block'):
                family_weights=g.groupby('family').loss_weight.sum().to_numpy()
                sample_weights=g.groupby(['family','sample_id']).loss_weight.sum()
                ok=abs(g.loss_weight.sum()-expected[block])<1e-12 and np.ptp(family_weights)<1e-12
                for _,part in sample_weights.groupby(level=0):ok=ok and np.ptp(part.to_numpy())<1e-12
                add('block_family_sample_read_weights',ok,method=method,block=block,weight=float(g.loss_weight.sum()))
            if d['gamma']>0 and not tr.endpoint.eq('muH').any():no_mu.append(dict(method=method,train_families=families))
    add('declared_no_mu_training_context',len(no_mu)==8,no_mu_contexts=len(no_mu))
    # An objective derivative with multiple physical endpoints and H-containing curves.
    families=('10.1002/aelm.202300679','10.1007/s10853-015-9693-6','10.1007/s40843-022-2167-x')
    for method in ('Thermal_original_RT_record_cap_g0.1','Thermal_binding_joint_record_cap_g1'):
        tr=study.training_data(families,method);m=study.mapper(method);w=np.asarray(study.core.specification(method)[2][0])
        y=tr.target.to_numpy(float);weight=tr.loss_weight.to_numpy(float);values,jac=study.forward_targets(m,w,tr)
        exact=2*jac.T@(weight*(values-y))
        for j in range(len(w)):
            def objective(v):
                error=study.forward_targets(m,v,tr)[0]-y
                return np.array([weight@(error**2)])
            fd=richardson(objective,w,j,.02)
            derivative('joint_objective_actual_chain',fd,np.array([exact[j]]),method=method,j=j)
        add('sigma_not_independent_training_label',not tr.endpoint.eq('sigma').any())
    paths=(Path(__file__),Path(study.__file__),ROOT/'src/sandbox/thermal_coherent_model.py',
        ROOT/'docs/ML_SANDBOX_THERMAL_JOINT_20261018.md',ROOT/'scripts/trainable_sandbox_20261017/phase16_thermal_audit.py')
    result=dict(created_utc=energy.now(),passed=all(c['passed'] for c in checks),checks=checks,
        maximum_scaled_derivative_error=max(errors),seconds=time.perf_counter()-clock,no_mu_supervision_contexts=no_mu,
        finite_difference_method='fourth-order Richardson;parameterh.08(binding.04),logcaph.005,sourceh.08,objectiveh.02;unchanged3e-6gate',
        retained_initial_failure='945checks with2e-5two-point steps;strongcompensation amplifiedroundoff;no model/formula/tolerance changed before refinedstencil',
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},scope='actual mathematical/input/loss contracts;not material accuracy',goal_complete=False)
    energy.write(study.OUT/'numerical_gates.json',result)
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),maximum_scaled_derivative_error=max(errors),
        seconds=result['seconds'],failures=[c for c in checks if not c['passed']]),indent=2))
    if not result['passed']:raise RuntimeError('Gate failed; no training authorized by this protocol')


if __name__=='__main__':main()
