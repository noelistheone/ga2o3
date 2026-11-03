"""Actual-state periodic Poisson, implicit derivative and independent-root gates."""
from pathlib import Path
from types import SimpleNamespace
import sys,time,numpy as np
from scipy.optimize import root
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from spatial_local_charge_v1 import FrozenChargeKernel
from spatial_parent_profile_v1 import parent_profile
from spatial_poisson_v1 import solve_poisson
from sandbox.joint_saved_model import SavedJointModel
from sandbox.ionized_transport import EPSILON0,E_CHARGE

OUT=c.BASE/'spatial_predictive_candidate/poisson_gates_v1'


def profiles(ref,cap,a,roles,N):
    stock=np.empty((len(ref),N));J=np.zeros((len(ref),N,4))
    for role in sorted(set(roles)):
        mask=roles==role;p=parent_profile(ref[mask],cap[mask],a*role,grid_size=N)
        stock[mask]=p.concentrations;J[mask,:,0]=p.amplitude_partial*role
        J[mask,:,3]=p.reference_partial*(np.log(10.)*ref[mask,None])
    return stock,J


def main():
    assert not (OUT/'protocol_before.json').exists()
    chosen=c.document(c.BASE/'spatial_predictive_candidate/local_charge_gates_v2/protocol_before.json')['selected_cases']
    plans=c.document(c.OLD/'prediction/plan.json')['plans'];table=c.cases();lookup={r.case_id:r for r in table.itertuples()}
    files=[Path(__file__),Path(__file__).with_name('spatial_poisson_v1.py'),
        Path(__file__).with_name('spatial_local_charge_v1.py'),Path(__file__).with_name('spatial_parent_profile_v1.py')]
    inputs=[c.OLD/'prediction/plan.json',c.OLD/'data/scenarios49.csv',
        c.BASE/'spatial_predictive_candidate/local_charge_gates_v2/gates.json',
        c.BASE/'spatial_predictive_candidate/profile_gates_v1/gates.json']
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes(files),
        input_hashes=c.file_hashes(inputs),selected_cases=chosen,N=16,modes=['common','structural_contrast'],
        amplitude_grid=[0.,.5,2.],period_nm_grid=[20.,200.],
        FD_coordinates=['amplitude','log10_background','log10_period','log10_all_mean_stocks'],
        FD_steps=[1e-4,2e-4,1e-4,2e-5],planned_cases=72,planned_FD_coordinates=48,
        no_fit_or_predictive_accuracy_claim=True,goal_complete=False))
    rows=[];derivatives=[];checks=[];start=time.perf_counter()
    for selected in chosen:
        row=lookup[selected['case_id']]
        plan=next(p for p in plans if p['variant']==selected['variant'] and p['outer_held']==row.related_cluster)
        mapper=SavedJointModel(plan['minimal_fit']);point=mapper.point(SimpleNamespace(**c.design.inputs(row)),require_observation=False)
        engine=mapper.engine.engine_for(row) if hasattr(mapper.engine,'engine_for') else mapper.engine.engines[row.scenario_pO2_atm]
        layer,solver=engine.layer_for(row);inv=solver.last_inventory;old=solver.last_prediction;ref=inv.totals.copy()
        Nd=old.diagnostics['Nd_cm3'];dm=layer.db.dm[layer.first];foreign=np.any(solver.parent_atoms>0,axis=1)
        cap=layer.db.prefactor[layer.first]*np.where(foreign,.1*(1-1e-6),1.)
        pars=layer.parameters(T=point.diagnostics['Tf_K'],Nd=Nd,weights=[0.],
            doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm,Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
        kernel=FrozenChargeKernel(layer,pars);N=16
        for mode in ['common','structural_contrast']:
            roles=np.ones(len(ref)) if mode=='common' else np.zeros(len(ref))
            if mode=='structural_contrast':
                roles[solver.receivers]=1.;roles[solver.complex]=-1.;roles[solver.native&(dm[:,0]<0)]=-1.
            def field(a,L,reference=ref,background=Nd,jacobian=False):
                stock,stockJ=profiles(reference,cap,a,roles,N)
                kw=dict(background_cm3=background,period_nm=L,initial_eV=old.ef_eV)
                if jacobian:kw.update(stock_jacobian=stockJ,
                    background_jacobian=np.array([0.,np.log(10.)*background,0.,0.]),log_period_jacobian=np.array([0.,0.,1.,0.]))
                return solve_poisson(kernel,stock,**kw)
            for a in [0.,.5,2.]:
                for L in [20.,200.]:
                    record=dict(variant=selected['variant'],case_id=row.case_id,mode=mode,amplitude=a,period_nm=L)
                    try:
                        result=field(a,L,jacobian=True)
                        uniform=float(max(abs(result.fermi_eV-old.ef_eV))) if a==0 else None
                        checks.extend([result.diagnostics['global_charge_relative_error']<2e-9,
                            result.diagnostics['implicit_linear_backward_error']<2e-11,
                            uniform is None or uniform<2e-9])
                        record.update(success=True,uniform_error=uniform,**result.diagnostics)
                        if a==.5 and L==200.:
                            stock,_=profiles(ref,cap,a,roles,N)
                            shifted=solve_poisson(kernel,np.roll(stock,3,axis=1),background_cm3=Nd,period_nm=L,initial_eV=old.ef_eV)
                            translation=float(max(abs(shifted.fermi_eV-np.roll(result.fermi_eV,3))))
                            checks.append(translation<2e-8);record['translation_error']=translation
                            for k,h in enumerate([1e-4,2e-4,1e-4,2e-5]):
                                entry=dict(variant=selected['variant'],case_id=row.case_id,mode=mode,coordinate=k)
                                try:
                                    def changed(delta):
                                        values=[a,Nd,L,ref]
                                        if k==0:values[0]+=delta
                                        elif k==1:values[1]*=10**delta
                                        elif k==2:values[2]*=10**delta
                                        else:values[3]=ref*10**delta
                                        return field(values[0],values[2],reference=values[3],background=values[1])
                                    plus=changed(h);minus=changed(-h)
                                    fd=(plus.fermi_eV-minus.fermi_eV)/(2*h)
                                    analytic=result.fermi_jacobian[:,k]
                                    eu=float(max(abs(fd-analytic))/(1+max(abs(fd))+max(abs(analytic))))
                                    lp=np.log10(np.c_[plus.local.carriers_cm3[:,0],plus.local.Nq2_cm3])
                                    lm=np.log10(np.c_[minus.local.carriers_cm3[:,0],minus.local.Nq2_cm3])
                                    fl=(lp-lm)/(2*h);al=result.log_state_jacobian[:,:,k]
                                    el=float(np.max(abs(fl-al)/(1+abs(fl)+abs(al))))
                                    checks.extend([eu<3e-6,el<5e-5]);entry.update(success=True,field_error=eu,log_state_error=el)
                                except Exception as exc:checks.append(False);entry.update(success=False,error=repr(exc))
                                derivatives.append(entry)
                            # Independent root: direct dimensional second differences,
                            # finite-difference Jacobian, no use of our Newton/IFT steps.
                            dx=L*1e-9/N;density=Nd+np.max(abs(kernel.q),axis=1)@stock
                            def residual(u):
                                second=(np.roll(u,1)-2*u+np.roll(u,-1))/(dx*dx)
                                Q=kernel.evaluate(u,stock,background_cm3=Nd).charge_cm3
                                return (-10.2*EPSILON0*second/(E_CHARGE*1e6)-Q)/density
                            alt=root(residual,np.full(N,old.ef_eV),method='hybr',options=dict(xtol=1e-10))
                            alt_error=float(max(abs(alt.x-result.fermi_eV)));alt_res=float(max(abs(residual(alt.x))))
                            checks.extend([alt_error<2e-6,alt_res<1e-7])
                            record.update(independent_root_status=bool(alt.success),independent_root_error=alt_error,independent_root_residual=alt_res)
                    except Exception as exc:checks.append(False);record.update(success=False,error=repr(exc))
                    rows.append(record)
        print('Poisson gate savedstate',selected['variant'],row.case_id,'cases',len(rows),flush=True)
    assert len(rows)==72
    c.e.csvwrite(OUT/'actual_field_cases.csv',rows);c.write(OUT/'derivatives.json',derivatives)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    successful=[r for r in rows if r['success']]
    c.write(OUT/'gates.json',dict(created_utc=c.e.now(),passed=all(checks),checks=len(checks),failures=sum(not bool(q) for q in checks),
        actual_saved_states=len(chosen),field_cases=len(rows),successful_field_cases=len(successful),
        derivative_coordinates=len(derivatives),successful_derivative_coordinates=sum(r['success'] for r in derivatives),
        maximum_field_FD_error=max((r['field_error'] for r in derivatives if r['success']),default=None),
        maximum_log_state_FD_error=max((r['log_state_error'] for r in derivatives if r['success']),default=None),
        seconds=time.perf_counter()-start,new_models_fitted=0,no_spatial_Hall_prediction_accuracy_claim=True,goal_complete=False))
    print('Poisson gates',len(checks),'failed',sum(not bool(q) for q in checks),flush=True)
    assert all(checks),'Preserve original field cases and use versioned repairs'


if __name__=='__main__':main()
