"""Pretraining numerical and observation-loss gates, not an accuracy study."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e
import phase29_data as data
import phase29_learning as l
from sandbox.pool_share_controls import PoolReferenceModel
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.paired_hall_observations import hall_pair_primitives
from sandbox.ionized_transport import E_CHARGE

OUT=data.OUT/'numerical_gates_v1'


def fd(fun,x,j,h):
    values=[]
    for step in (h,h/2):
        plus=np.array(x,float);minus=plus.copy()
        plus[j]+=step;minus[j]-=step
        values.append((fun(plus)-fun(minus))/(2*step))
    return (4*values[1]-values[0])/3


def parameters(method):
    s=l.SPECS[method]
    if s['kind']=='activation':return np.array([15.,-.2,.3,-.2,.1,-.15,-1.2])
    if s['kind']=='reference':return np.array([.4,14.,-1.2] if s['learn_scale'] else [.4,14.])
    coeff={'constant':[.3],'dose':[.3,-.2],'process':[.3,-.2,.1,-.15]}[s['basis']]
    return np.array([.4,14.,*coeff,*([-1.2] if s['learn_scale'] else [])])


def main():
    assert not (OUT/'protocol_before_gates.json').exists(),'preserve previous attempt before a versioned repair'
    code=[Path(__file__),Path(__file__).with_name('phase29_learning.py'),
        Path(__file__).with_name('phase29_training.py'),
        ROOT/'src/sandbox/reference_scattering.py',ROOT/'src/sandbox/activation_transport_control.py',
        ROOT/'src/sandbox/joint_inventory_scattering.py',ROOT/'src/sandbox/learnable_constant_tau.py',
        ROOT/'src/sandbox/paired_hall_observations.py',ROOT/'src/sandbox/pool_share_controls.py']
    inputs=[data.OUT/'data/qualification.json',data.OUT/'data/conditions35.csv',
        data.OUT/'data/scenarios49.csv',data.OUT/'data/episodes35.csv']
    e.write(OUT/'protocol_before_gates.json',dict(created_utc=e.now(),derivative_tolerance=3e-6,
        finite_difference='centered Richardson;parameter/source h.02,input log h.01,growth h.2',
        numerical_cases='one existing H-known Si plus two conditional Gogova Sn inputs',
        loss_target_scope='deterministic artificial labels for derivative checks;not used for training/selection',
        no_training_or_prediction_accuracy_claim=True,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs}))
    checks=[];errors=[];states=[];clock=time.perf_counter()
    def add(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw))
    def close(name,a,b,tol=3e-6,**kw):
        error=float(np.max(abs(np.asarray(a)-np.asarray(b))/(1+abs(np.asarray(b)))))
        errors.append(error);add(name,error<=tol,error=error,tolerance=tol,**kw)
    def output(p):return np.r_[p.values,p.transport_values]
    cases=data.load_cases();conditions=data.load_conditions();roles=data.load_roles()
    h=cases[cases.H_known.eq(True)].iloc[0]
    rows=[h,cases[cases.family.eq(data.GOGOVA)&cases.sample_label.eq('A')].iloc[2],
        cases[cases.family.eq(data.GOGOVA)&cases.sample_label.eq('B')].iloc[5]]
    sources=[SimpleNamespace(**r.to_dict()) for r in rows]
    original=PoolReferenceModel(temperature_mode='effective',profile=published_coherent_profile())
    ref=l.model('ReferenceTau')
    for row in sources:
        for kind,z in (('none',0.),('share',.2),('offset',.15)):
            a=ref.point(row,[.4,14.,0.],source_kind=kind,source_shift=z,require_observation=False)
            b=original.point(row,[.4,14.],source_kind=kind,source_shift=z,require_observation=False)
            close('reference205_zero_values',output(a),output(b),1e-10,case=row.case_id,source=kind)
            close('reference205_zero_parameter_chain',np.r_[a.jacobian[:,:2],a.transport_jacobian[:,:2]],
                np.r_[b.jacobian,b.transport_jacobian],1e-9,case=row.case_id,source=kind)
            close('reference205_zero_input_chain',np.r_[a.input_jacobian,a.transport_input_jacobian],
                np.r_[b.input_jacobian,b.transport_input_jacobian],1e-9,case=row.case_id,source=kind)
            close('reference205_zero_source_chain',np.r_[a.source_jacobian,a.transport_source_jacobian],
                np.r_[b.source_jacobian,b.transport_source_jacobian],1e-9,case=row.case_id,source=kind)
    # The joint module was already checked in phase28; the smaller reference
    # and matched empirical transport control are new in phase29.
    for method in ('ReferenceTau','ActivationProcessPair'):
        mapper=l.model(method);w=parameters(method)
        for row in sources:
            kinds=(('none',0.),('offset',.15))+((('share',.2),) if method=='ReferenceTau' else ())
            for kind,z in kinds:
                def point(v,r=row,kind=kind,z=z):
                    return mapper.point(r,v,source_kind=kind,source_shift=z,require_observation=False)
                p=point(w);J=np.r_[p.jacobian,p.transport_jacobian]
                states.append(dict(method=method,case_id=row.case_id,source=kind,
                    values=output(p).tolist(),diagnostics=p.diagnostics))
                for j in range(len(w)):
                    close('new_model_parameter_derivative',fd(lambda v:output(point(v)),w,j,.02),
                        J[:,j],method=method,case=row.case_id,source=kind,j=j)
                if kind!='none':
                    close('new_model_source_derivative',fd(lambda v:output(mapper.point(row,w,
                        source_kind=kind,source_shift=v[0],require_observation=False)),[z],0,.02),
                        np.r_[p.source_jacobian,p.transport_source_jacobian],method=method,case=row.case_id,source=kind)
                for j,name in enumerate(p.input_names):
                    attr={'log10_dose_cm3':'incorporated_cm3','log10_Tobs_K':'Tobs_K',
                        'log10_H_cm3':'H_cm3','growth_T_C':'growth_T_C'}[name]
                    value=float(getattr(row,attr));x=value if name=='growth_T_C' else np.log10(value)
                    def shifted(v):
                        r=SimpleNamespace(**vars(row));setattr(r,attr,float(v[0]) if name=='growth_T_C' else 10**float(v[0]))
                        return output(mapper.point(r,w,source_kind=kind,source_shift=z,require_observation=False))
                    close('new_model_input_derivative',fd(shifted,[x],0,.2 if name=='growth_T_C' else .01),
                        np.r_[p.input_jacobian[:,j],p.transport_input_jacobian[:,j]],
                        method=method,case=row.case_id,source=kind,input=name)
                close('new_model_sigma_identity',[p.transport_values[1]],
                    [np.log10(E_CHARGE)+p.values[0]+p.transport_values[0]],1e-12,method=method,source=kind)
                close('new_model_sigma_jacobian',p.transport_jacobian[1],
                    p.jacobian[0]+p.transport_jacobian[0],1e-12,method=method,source=kind)
    # Labels here are artificial, independent of observed held-out responses.
    synthetic=cases.copy()
    synthetic['target']=np.log10(synthetic.incorporated_cm3)-.37
    synthetic['mu_cm2Vs']=23.
    for method in l.NONLINEAR:
        w=parameters(method);lo,hi=l.bounds(method)
        add('gate_parameters_within_training_bounds',np.all(w>=lo)&np.all(w<=hi),method=method)
        config=dict(gamma=.1,alpha=.1);r,J,q=l.loss_arrays(method,w,synthetic,config)
        for j in range(len(w)):
            close('full_loss_gradient_including_reference_and_scale',fd(
                lambda v:np.array([np.linalg.norm(l.loss_arrays(method,v,synthetic,config)[0])**2]),w,j,.02),
                [2*J[:,j]@r],method=method,j=j)
        r0,J0,_=l.loss_arrays(method,w,synthetic,dict(gamma=0.,alpha=0.))
        pair,rho=l.observed_pair(synthetic);wp=l.weights(synthetic,paired_only=True)
        add('Hall_counted_once',len(r0)==len(cases) and np.array_equal(r[:len(cases)],r0),method=method)
        close('second_direction_is_only_primitive_rho',r[len(cases):len(cases)+int(pair.sum())],
            np.sqrt(.1*wp[pair])*(-q['values'][pair,2]-rho[pair]),1e-13,method=method)
    for units in l.contexts():
        sub=cases[cases.related_cluster.isin(units)]
        for paired_only in (False,True):
            w=l.weights(sub,paired_only=paired_only)
            t=sub.copy();t['weight']=w
            if paired_only:t=t[t.mobility_eligible.eq(True)]
            close('weight_sum_one',[w.sum()],[1.],1e-13,groups=len(units),paired=paired_only)
            byunit=t.groupby('related_cluster').weight.sum()
            close('related_groups_have_equal_total_weight',byunit.to_numpy(),np.repeat(1/len(byunit),len(byunit)),1e-13)
            for family,part in t.groupby('family'):
                bycondition=part.groupby('sample_id').weight.sum().to_numpy()
                close('scenario_rows_do_not_multiply_labels',bycondition,np.repeat(part.weight.sum()/len(bycondition),len(bycondition)),1e-13,
                    family=family,groups=len(units),paired=paired_only)
    close('scenario_weights_per_condition',cases.groupby('sample_id').scenario_weight.sum().to_numpy(),np.ones(35),1e-14)
    add('condition_inputs_unknown_remain_missing',conditions[conditions.family.eq(data.GOGOVA)].growth_T_C.isna().all())
    add('new_two_anchor_no_carrier_query',not ((roles.family.eq(data.GOGOVA))&roles.budget.eq(2)&roles.role.eq('query')).any())
    primary=set(cases.loc[cases.primary_common20,'sample_id'])
    add('original_common20_preserved',len(primary)==20 and not any(s.startswith(data.GOGOVA) for s in primary))
    for i,row in synthetic[synthetic.mobility_eligible.eq(True)].iterrows():
        pair=l.observed_pair(synthetic.loc[[i]])[1][0]
        close('rho_from_Hall_pair_coordinates',[pair],[hall_pair_primitives(row.target,np.log10(row.mu_cm2Vs))[1]],1e-13)
    direct=l.fit_direct(synthetic,.1);v=l.direct_values(direct,synthetic)
    close('direct_multi_derived_sigma',v[:,2],np.log10(E_CHARGE)+v[:,0]+v[:,1],1e-12)
    expected_center=np.average(l.direct_features(synthetic),axis=0,weights=l.weights(synthetic))
    close('direct_scaling_uses_only_supplied_training_conditions',direct['center'],expected_center,1e-13)
    add('two_direct_heads_only',direct['new_linear_solves']==2 and direct['two_heads_use_two_primitive_observables'])
    fits=len(l.contexts())*sum(len(l.configs(m)) for m in l.METHODS)
    nonlinear=len(l.contexts())*sum(len(l.configs(m)) for m in l.NONLINEAR)
    add('declared_training_record_counts',fits==1392 and nonlinear==1200,fit_records=fits,nonlinear=nonlinear)
    e.write(OUT/'numerical_states.json',states)
    result=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,
        maximum_scaled_difference=max(errors),seconds=time.perf_counter()-clock,
        actual_training=0,actual_prediction_validation=0,goal_complete=False)
    e.write(OUT/'gates.json',result)
    print(json.dumps({k:result[k] for k in ('passed','checks','maximum_scaled_difference','seconds')}),flush=True)
    if not result['passed']:raise RuntimeError('Preserve gates before a versioned correction')


if __name__=='__main__':main()
