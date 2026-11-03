"""Numerical gates for a candidate joint state/scattering learning interface.

These are model/derivative and observation-algebra checks. They are not new
training, predictive validation, measured covariances or a Born certificate.
"""
from pathlib import Path
from types import SimpleNamespace
import sys,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.coherent_transport import fermi_transport
from sandbox.learnable_constant_tau import learnable_fermi_transport
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel
from sandbox.joint_inventory_scattering import JointInventoryScatteringModel,BASE_LOG_MU0
from sandbox.paired_hall_observations import (hall_pair_primitives,paired_hall_residual,
    PAIR_TO_PRIMITIVES,PRIMITIVES_TO_PAIR)
from sandbox.ionized_transport import E_CHARGE
import phase2_energy as e

OUT=e.BASE/'phase28_paired_transport/joint_scattering_core_v1'


def main():
    code=[Path(__file__),ROOT/'src/sandbox/learnable_constant_tau.py',
        ROOT/'src/sandbox/joint_inventory_scattering.py',ROOT/'src/sandbox/paired_hall_observations.py',
        ROOT/'src/sandbox/coherent_transport.py',ROOT/'src/sandbox/charge_feasible_share.py',
        ROOT/'src/sandbox/fixed_inventory_charge.py']
    data=e.BASE/'phase25_shared_process/data/conditions33.csv'
    protocol=dict(created_utc=e.now(),stage='candidate trainable joint interface numerical gates',
        theory='positive tau0 and full Fermi Hall moments;one global logmu0 shift shared across conditions',
        derivative_tolerance=3e-6,transport_u_tolerance=2e-7,
        no_prediction_or_training_claim=True,no_measured_covariance_claim=True,
        preserve_failures_before_any_versioned_repair=True,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(data.relative_to(ROOT)):e.sha(data)},goal_complete=False)
    target=OUT/'protocol_before_gates.json';assert not target.exists(),target
    e.write(target,protocol)
    checks=[];derivatives=[];clock=time.perf_counter()
    def add(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw))
    def close(name,actual,expected,tolerance,**kw):
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))/(1+np.abs(expected))))
        derivatives.append(error);add(name,error<=tolerance,error=error,tolerance=tolerance,**kw)
    def fd(fun,x,j,h):
        values=[]
        for step in (h,h/2):
            plus=np.array(x,float);minus=np.array(x,float);plus[j]+=step;minus[j]-=step
            values.append((fun(plus)-fun(minus))/(2*step))
        return (4*values[1]-values[0])/3
    def output(p):return np.r_[p.values,p.transport_values]
    def params(basis,shift):
        coeff={'constant':[.3],'dose':[.3,-.2],'process':[.3,-.2,.1,-.15]}[basis]
        return np.array([.4,14.,*coeff,shift])
    old=read(data);hrow=old[old.H_known.eq(True)].iloc[0]
    cases=[dict(name='old_reported_Si_H_condition',element='Si',incorporated_cm3=float(hrow.incorporated_cm3),
        H_known=True,H_cm3=float(hrow.H_cm3),growth_T_C=float(hrow.growth_T_C),Tobs_K=300.,scenario_pO2_atm=float(hrow.scenario_pO2_atm)),
        dict(name='RSC_A_conditional',element='Sn',incorporated_cm3=8e18,H_known=False,H_cm3=0.,growth_T_C=800.,Tobs_K=300.,scenario_pO2_atm=1e-5),
        dict(name='RSC_B_conditional',element='Sn',incorporated_cm3=3.6e18,H_known=False,H_cm3=0.,growth_T_C=825.,Tobs_K=300.,scenario_pO2_atm=1e-2)]
    profile=published_coherent_profile();transport_states=[];saved_states=[]
    for basis in ('constant','dose','process'):
        model=JointInventoryScatteringModel(basis=basis,profile=profile)
        parent=ChargeFeasibleModel(basis=basis,profile=profile)
        for source in cases:
            row=SimpleNamespace(**source);w0=params(basis,0.)
            a=model.point(row,w0,require_observation=False);b=parent.point(row,w0[:-1],require_observation=False)
            add('same205physical_values',np.max(abs(output(a)-output(b)))<1e-10,basis=basis,case=row.name)
            add('same205base_parameter_chain',np.max(abs(np.r_[a.jacobian[:,:-1],a.transport_jacobian[:,:-1]]-
                np.r_[b.jacobian,b.transport_jacobian]))<1e-9,basis=basis,case=row.name)
            add('same205input_chain',np.max(abs(np.r_[a.input_jacobian,a.transport_input_jacobian]-
                np.r_[b.input_jacobian,b.transport_input_jacobian]))<1e-9,basis=basis,case=row.name)
            if basis=='process':transport_states.append(dict(case=row.name,**a.diagnostics['coherent_transport']))
            variants=[('none',0.)]+([('share',.2),('offset',.15)] if basis=='process' else [])
            for kind,z in variants:
                w=params(basis,-1.2)
                def point(v,r=row,kind=kind,z=z):return model.point(r,v,source_kind=kind,source_shift=z,require_observation=False)
                a=point(w);J=np.r_[a.jacobian,a.transport_jacobian]
                saved_states.append(dict(basis=basis,case=row.name,source_kind=kind,valid=a.diagnostics['observation_domain_valid'],
                    outputs=output(a).tolist(),diagnostics=a.diagnostics))
                add('conditional_electron_readout_valid',a.diagnostics['observation_domain_valid'],basis=basis,case=row.name,source=kind)
                for j in range(len(w)):
                    close('full_joint_parameter',fd(lambda v:output(point(v)),w,j,.02),J[:,j],3e-6,basis=basis,case=row.name,source=kind,j=j)
                if kind!='none':
                    def sf(v):return output(model.point(row,w,source_kind=kind,source_shift=v[0],require_observation=False))
                    close('joint_source_chain',fd(sf,[z],0,.02),np.r_[a.source_jacobian,a.transport_source_jacobian],3e-6,basis=basis,case=row.name,source=kind)
                for j,name in enumerate(a.input_names):
                    original=float(getattr(row,'incorporated_cm3' if name=='log10_dose_cm3' else 'Tobs_K' if name=='log10_Tobs_K' else 'H_cm3' if name=='log10_H_cm3' else 'growth_T_C'))
                    x=original if name=='growth_T_C' else np.log10(original)
                    attr='incorporated_cm3' if name=='log10_dose_cm3' else 'Tobs_K' if name=='log10_Tobs_K' else 'H_cm3' if name=='log10_H_cm3' else 'growth_T_C'
                    def xf(v):
                        r=SimpleNamespace(**vars(row));setattr(r,attr,float(v[0]) if name=='growth_T_C' else 10**float(v[0]))
                        return output(model.point(r,w,source_kind=kind,source_shift=z,require_observation=False))
                    close('joint_input_chain',fd(xf,[x],0,.2 if name=='growth_T_C' else .01),
                        np.r_[a.input_jacobian[:,j],a.transport_input_jacobian[:,j]],3e-6,basis=basis,case=row.name,source=kind,input=name)
                add('sigma_value_identity',abs(a.transport_values[1]-(np.log10(E_CHARGE)+a.values[0]+a.transport_values[0]))<1e-10,basis=basis,case=row.name,source=kind)
                add('sigma_parameter_identity',np.max(abs(a.transport_jacobian[1]-a.jacobian[0]-a.transport_jacobian[0]))<1e-10,basis=basis,case=row.name,source=kind)
                add('sigma_input_identity',np.max(abs(a.transport_input_jacobian[1]-a.input_jacobian[0]-a.transport_input_jacobian[0]))<1e-10,basis=basis,case=row.name,source=kind)
    e.write(OUT/'actual_numerical_states.json',saved_states)
    for state in transport_states:
        for shift in (-2.,0.,2.):
            for hall in (True,False):
                u=BASE_LOG_MU0+shift;ln=np.log10(state['free_n_cm3']);lq=np.log10(state['Nq2_cm3'])
                def fun(v,order=256):
                    return learnable_fermi_transport(ln,lq,300.,log10_extra_mobility=v[0],Nc_ref_cm3=profile.Nc_ref,
                        T_ref_K=profile.T_ref,mass_ratio=profile.electron_mass_ratio,hall=hall,order=order)
                tr=fun([u]);d=fd(lambda v:fun(v).values,[u],0,.001)
                close('transport_new_mu0_partial',d,tr.jacobian[:,-1],2e-7,case=state['case'],shift=shift,hall=hall)
                close('transport_quadrature_refinement',tr.values,fun([u],512).values,2e-7,case=state['case'],shift=shift,hall=hall)
                add('new_scattering_monotonic_signs',d[0]<=1e-8 and d[1]>=-1e-8 and -1e-8<=d[2]<=1+1e-8,case=state['case'],shift=shift,hall=hall)
                add('new_scattering_jacobian_identity',abs(tr.jacobian[0,-1]+tr.jacobian[1,-1]-tr.jacobian[2,-1])<1e-12,case=state['case'],shift=shift,hall=hall)
    # Exact observation-coordinate algebra; specified covariance scenarios,
    # not estimated experimental uncertainty or synthetic accuracy evidence.
    primitive_to_triple=np.array([[-1.,0.],[1.,-1.],[0.,-1.]])
    add('two_observation_transforms_inverse',np.max(abs(PAIR_TO_PRIMITIVES@PRIMITIVES_TO_PAIR-np.eye(2)))==0.)
    for covariance in (np.diag([.03**2,.05**2]),np.array([[.04**2,.0004],[.0004,.05**2]])):
        pair_cov=PRIMITIVES_TO_PAIR@covariance@PRIMITIVES_TO_PAIR.T
        triple_cov=primitive_to_triple@covariance@primitive_to_triple.T
        add('derived_sigma_has_only_rank2',np.linalg.matrix_rank(triple_cov)==2)
        add('derived_sigma_null_vector',np.max(abs(triple_cov@np.array([1.,1.,-1.])))<1e-14)
        predicted=np.array([18.1,1.2]);observed=np.array([17.9,1.0]);J=np.array([[.2,.3],[-.1,.5]])
        residual,jac=paired_hall_residual(predicted,observed,J,primitive_covariance=covariance)
        direct=float((predicted-observed)@np.linalg.solve(pair_cov,predicted-observed))
        close('metric_coordinate_invariance',np.array([residual@residual]),np.array([direct]),1e-12)
        for j in range(2):
            def loss(v):
                r,_=paired_hall_residual(predicted+J@v,observed,J,primitive_covariance=covariance)
                return np.array([r@r])
            close('paired_objective_gradient',fd(loss,[0.,0.],j,1e-4),np.array([2*jac[:,j]@residual]),1e-8,j=j)
        primitive=hall_pair_primitives(*predicted)
        add('primitive_rho_matches_derived_sigma',abs(primitive[1]+np.log10(E_CHARGE)+predicted.sum())<1e-13)
    result=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,
        full_numerical_states=len(saved_states),actual_physical_condition_inputs=3,
        transport_derivative_states=len(transport_states)*6,
        maximum_scaled_difference=max(derivatives),seconds=time.perf_counter()-clock,
        actual_new_training=0,actual_new_predictive_validation=0,
        no_material_or_Born_certificate=True,no_measured_covariance=True,goal_complete=False)
    e.write(OUT/'gates.json',result)
    print(json.dumps({k:result[k] for k in ('passed','checks','full_numerical_states','maximum_scaled_difference','seconds')},ensure_ascii=False),flush=True)
    if not result['passed']:raise RuntimeError('preserve gate failure before a versioned correction')


if __name__=='__main__':main()
