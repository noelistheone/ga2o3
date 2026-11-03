"""Independent conservation, actual IFT/input chains and training contracts."""
from pathlib import Path
from types import SimpleNamespace
from itertools import combinations
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.thermal_coherent_model import ThermalCoherentHallModel
from sandbox.coherent_fermi import published_coherent_profile
import phase17_prediction as study
import phase2_energy as energy


def main():
    checks=[];errors=[];clock=time.perf_counter()
    def add(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw))
    def derivative(name,fd,analytic,tol=5e-6,**kw):
        error=float(np.max(abs(fd-analytic)/(1+abs(fd))));errors.append(error);add(name,error<tol,error=error,**kw)
    def rich(fun,x,j,h):
        slopes=[]
        for step in (h,h/2):
            p=np.array(x,float);n=np.array(x,float);p[j]+=step;n[j]-=step
            slopes.append((fun(p)-fun(n))/(2*step))
        return (4*slopes[1]-slopes[0])/3
    cases=[dict(element='Si',incorporated_cm3=1e18,H_known=False,H_cm3=0.,Tobs_K=40.,growth_T_C=523.,thermal_cap_K=796.15),
        dict(element='Sn',incorporated_cm3=3e18,H_known=False,H_cm3=0.,Tobs_K=300.,growth_T_C=900.,thermal_cap_K=1173.15),
        dict(element='Si',incorporated_cm3=5.4e17,H_known=True,H_cm3=1.8e17,Tobs_K=150.,growth_T_C=830.,thermal_cap_K=1103.15)]
    for ref in study.REFERENCES:
        model=study.mapper(ref);old=ThermalCoherentHallModel(dict(study.STRUCTURES[ref],thermal_mode='record_cap'),profile=model.profile,pO2_atm=1e-5)
        w=np.array(study.specification(ref)[2][0]);v=np.r_[6.,w]
        for data in cases:
            row=SimpleNamespace(**data);p=model.point(row,w);o=old.point(row,v)
            error=max(abs(p.values-o.values).max(),abs(p.transport_values-o.transport_values).max())
            add('zero_allocation_reference_values',error<2e-5,error=float(error),method=ref,element=row.element,T=row.Tobs_K)
            derivative('zero_reference_base_parameter_derivatives',p.jacobian,o.jacobian[:,1:],method=ref)
            derivative('zero_reference_transport_derivatives',p.transport_jacobian,o.transport_jacobian[:,1:],method=ref)
    rng=np.random.default_rng(1731)
    # Exact linear foreign-atom identity independent of equilibrium/targets.
    for data in (cases[0],cases[1],cases[2]):
        row=SimpleNamespace(**data);m=study.mapper('Stock_complex_constant_binding_joint');layer,solver=m.layer_for(row)
        A=solver.parent_atoms.T;r=np.zeros(layer.db.nD);r[solver.receivers]=[.31,.69]
        u=np.where(solver.complex,solver.dopant_count,0.)
        for alpha in (0.,.1,.5,1.):
            R=np.eye(layer.db.nD)-alpha*np.diag(solver.complex.astype(float))+alpha*np.outer(r,u)
            C=rng.uniform(.1,2.,layer.db.nD)
            add('linear_allocation_foreign_conservation',max(abs(A@R-A).ravel())<1e-13,element=row.element,alpha=alpha)
            add('linear_allocation_positive_and_atom_budget',min(R@C)>=0 and max(abs(A@(R@C)-A@C))<1e-12,alpha=alpha)
            if alpha==0:add('linear_allocation_exact_identity',np.array_equal(R,np.eye(layer.db.nD)))
    # Complete raw hot-reference -> allocation -> fixed-cold IFT.
    m=study.mapper('Stock_complex_constant_binding_joint');row=SimpleNamespace(**cases[2]);layer,solver=m.layer_for(row)
    p=layer.parameters(T=1101.49,Nd=1e16,weights=[0.],doses=(row.incorporated_cm3,row.H_cm3),pO2=1e-5,Tobs=150.,binding_eV=.03,screen_s=.2)
    nv=layer.np+2;chain=np.c_[np.eye(layer.np),np.zeros((layer.np,2))];dn=np.zeros(nv);dc=np.zeros(nv);dn[-2]=1;dc[-1]=1
    def raw(v):
        inventory=solver.allocate(v[:layer.np],chain,native_s=v[-2],complex_s=v[-1],native_s_jacobian=dn,complex_s_jacobian=dc)
        return solver.solve(v[:layer.np],inventory,chain)
    vector=np.r_[p,.3,.2];a=raw(vector)
    for j,name in enumerate((*layer.parameter_names,'native_s','complex_s')):
        if j==layer.ie+1 or j in (layer.iloc,layer.iloc+1):
            h=1e-5;plus=vector.copy();plus[j]+=h;fd=(raw(plus).values-a.values)/h
            derivative('raw_allocation_cold_IFT_one_sided',fd,a.jacobian[:,j],tol=3e-5,parameter=name)
        else:
            fd=rich(lambda v:raw(v).values,vector,j,.002)
            derivative('raw_allocation_cold_IFT',fd,a.jacobian[:,j],parameter=name)
    add('raw_foreign_mass_and_jacobian',a.diagnostics['foreign_atom_relative_error']<3e-8 and a.diagnostics['foreign_atom_derivative_error']<3e-7)
    inventory=solver.allocate(p,chain,native_s=.3,complex_s=.2,native_s_jacobian=dn,complex_s_jacobian=dc)
    for shift in (-.5,.5):
        modified=p.copy();modified[2]+=shift;b=solver.solve(modified,inventory,chain)
        add('fixed_inventory_common_parent_energy_gauge',max(abs(a.values-b.values))<1e-9,shift_eV=shift,error=float(max(abs(a.values-b.values))))
    add('conditional_charge_response_negative',a.diagnostics['cold_Q_EF_cm3_per_eV']<0)
    # Actual deployed theta/input/source map for every new physical structure.
    for method,definition in study.STRUCTURES.items():
        if 'empirical' in definition or method in study.REFERENCES:continue
        model=study.mapper(method);w=np.array(study.specification(method)[2][0]);w[model.ic0]=.3
        if model.ic1 is not None:w[model.ic1]=.2
        for data in cases:
            row=SimpleNamespace(**data);a=model.point(row,w)
            def values(v):
                q=model.point(row,v);return np.r_[q.values,q.transport_values]
            for j,name in enumerate(model.parameter_names):
                # Zero/output baseline has near-cancellation; a wider fourth-order
                # stencil is used for its shared physical coordinates.
                h=.04 if definition['inventory_kind']=='output' and j<model.ic0 else .002
                fd=rich(values,w,j,h)
                derivative('deployed_parameter_values',fd[:3],a.jacobian[:,j],method=method,parameter=name,T=row.Tobs_K)
                derivative('deployed_parameter_transport',fd[3:],a.transport_jacobian[:,j],method=method,parameter=name,T=row.Tobs_K)
            input_fields=['incorporated_cm3','thermal_cap_K','Tobs_K']+(['H_cm3'] if row.H_known else [])
            iv=np.log10([getattr(row,k) for k in input_fields])
            def input_forward(v):
                r=SimpleNamespace(**data)
                for field,val in zip(input_fields,v):setattr(r,field,10**val)
                q=model.point(r,w);return np.r_[q.values,q.transport_values]
            for j,name in enumerate(a.input_names):
                fd=rich(input_forward,iv,j,.005 if definition['inventory_kind']=='output' else .002)
                derivative('deployed_physical_input_values',fd[:3],a.input_jacobian[:,j],method=method,input=name,T=row.Tobs_K)
                derivative('deployed_physical_input_transport',fd[3:],a.transport_input_jacobian[:,j],method=method,input=name,T=row.Tobs_K)
            kinds=['offset']+([definition['inventory_kind']] if definition['inventory_kind'] in ('native','complex') else [])
            for kind in kinds:
                z=.01;h=.002;q=model.point(row,w,source_kind=kind,source_shift=z)
                fd=rich(lambda v:np.r_[model.point(row,w,source_kind=kind,source_shift=v[0]).values,
                    model.point(row,w,source_kind=kind,source_shift=v[0]).transport_values],[z],0,h)
                derivative('deployed_source_values',fd[:3],q.source_jacobian,method=method,kind=kind)
                derivative('deployed_source_transport',fd[3:],q.transport_source_jacobian,method=method,kind=kind)
                zero=model.point(row,w,source_kind=kind,source_shift=0.);base=model.point(row,w)
                add('deployed_source_zero',max(abs(zero.values-base.values).max(),abs(zero.transport_values-base.transport_values).max())<1e-9,method=method,kind=kind)
    # The strongest added simple control has an independent exact derivative.
    tr=study.training_data(study.FAMILIES,'activation_sigmoid_RT');w=np.array(study.specification('activation_sigmoid_RT')[2][0]);v,J=study.activation(w,tr)
    for j in range(3):derivative('direct_activation_gradient',rich(lambda x:study.activation(x,tr)[0],w,j,.002),J[:,j],j=j)
    contexts=[f for n in (3,4) for f in combinations(study.FAMILIES,n)]
    for fam in contexts:
        for method,d in study.STRUCTURES.items():
            tr=study.training_data(fam,method)
            add('training_members_ownT_no_mu',set(tr.family)<=set(fam) and tr.endpoint.eq('nH').all() and tr.obs_id.is_unique and tr[tr.kind.eq('curve')].Tobs_K.le(275).all(),method=method)
            for kind,g in tr.groupby('kind'):
                factor=1. if kind=='RT' else .1
                add('training_block_sourceequal',abs(g.loss_weight.sum()-factor)<1e-12 and np.ptp(g.groupby('family').loss_weight.sum().to_numpy())<1e-12,method=method,kind=kind)
    paths=(Path(__file__),ROOT/'scripts/trainable_sandbox_20261017/phase17_prediction.py',ROOT/'src/sandbox/fixed_inventory_charge.py',
        ROOT/'src/sandbox/inventory_coherent_model.py',ROOT/'docs/ML_SANDBOX_INVENTORY_LEARNING_20261018.md')
    result=dict(created_utc=energy.now(),passed=all(c['passed'] for c in checks),checks=checks,
        maximum_scaled_derivative_error=max(errors),seconds=time.perf_counter()-clock,
        finite_difference='fourth-orderRichardson;outputbaselinephysicalinputh.005 ratherthan.002 toresolve smallfreechargedifferences;unchanged5e-6testcriterion',
        retained_initial_gate='3outputdoseinputchecksfailed7.26e-6/7.24e-6/6.63e-6;no formula/analyticgradient/testthresholdchange;no trainingbeforepass',
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},
        scope='actualconservation/root/IFT/readout/input/source/labelcontracts;notmaterial accuracy',goal_complete=False)
    energy.write(study.OUT/'numerical_gates.json',result)
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),maximum_scaled_derivative_error=max(errors),
        seconds=result['seconds'],failures=[c for c in checks if not c['passed']]),indent=2))
    if not result['passed']:raise RuntimeError('Inventory gate failed;no training')


if __name__=='__main__':main()
