"""Compare new batched/readout code with original physics and actual fields."""
from pathlib import Path
from types import SimpleNamespace
import sys,time,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from spatial_transport_v1 import local_transport,laminate_readout,LN10
from spatial_local_charge_v1 import FrozenChargeKernel
from spatial_poisson_v1 import solve_poisson
from spatial_poisson_gates_v1 import profiles
from sandbox.coherent_transport import fermi_transport
from sandbox.coherent_fermi import reference,published_coherent_profile
from sandbox.joint_saved_model import SavedJointModel
from sandbox.ionized_transport import E_CHARGE
from sandbox.thermo import KB_EV

OUT=c.BASE/'spatial_predictive_candidate/transport_gates_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    files=[Path(__file__),Path(__file__).with_name('spatial_transport_v1.py'),
        Path(__file__).with_name('spatial_poisson_v1.py'),Path(__file__).with_name('spatial_local_charge_v1.py'),
        Path(__file__).with_name('spatial_parent_profile_v1.py'),Path(__file__).with_name('spatial_poisson_gates_v1.py')]
    chosen=c.document(c.BASE/'spatial_predictive_candidate/local_charge_gates_v2/protocol_before.json')['selected_cases']
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes(files),
        input_hashes=c.file_hashes([c.BASE/'spatial_predictive_candidate/poisson_gates_v1/gates.json',
            c.OLD/'prediction/plan.json',c.OLD/'data/scenarios49.csv']),selected_cases=chosen,
        actual_grid_size=16,actual_modes=['common','structural_contrast'],
        actual_amplitudes=[0.,2.],actual_period_nm=200.,fixed_local_extra_mobility=205.,
        no_fit_or_predictive_accuracy_claim=True,goal_complete=False))
    start=time.perf_counter();checks=[];errors=[];records=[];derivatives=[]
    p=published_coherent_profile();T=300.;eta=np.repeat([-100.,-20.,-8.,0.,2.,20.,50.,60.,100.,1000.],4)
    lNI=np.tile([14.,17.,19.,21.],10);ln=(np.log(p.Nc_ref)+reference().log(eta)[0])/LN10
    batch=local_transport(eta,ln,lNI,T,Nc_ref_cm3=p.Nc_ref,T_ref_K=p.T_ref,mass_ratio=p.electron_mass_ratio)
    for i in range(len(eta)):
        old=fermi_transport(ln[i],lNI[i],T,Nc_ref_cm3=p.Nc_ref,T_ref_K=p.T_ref,
            mass_ratio=p.electron_mass_ratio,extra_mobility_cm2Vs=205.)
        ev=float(max(abs(old.values-batch.values[i])));ej=float(np.max(abs(old.jacobian[:,:2]-batch.jacobian[i])))
        checks.extend([ev<2e-9,ej<2e-8]);records.append(dict(kind='scalar_comparison',eta=eta[i],log_NI=lNI[i],value_error=ev,jacobian_error=ej))
    high=local_transport(eta,ln,lNI,T,Nc_ref_cm3=p.Nc_ref,T_ref_K=p.T_ref,mass_ratio=p.electron_mass_ratio,order=512)
    quadrature_error=float(np.max(abs(high.values-batch.values)));checks.append(quadrature_error<2e-7)
    for k in range(2):
        h=2e-5;plus=[ln.copy(),lNI.copy()];minus=[ln.copy(),lNI.copy()];plus[k]+=h;minus[k]-=h
        def perturbed(values):
            ep=np.array([reference().eta_from_logF(x*LN10-np.log(p.Nc_ref)) for x in values[0]])
            return local_transport(ep,*values,T,Nc_ref_cm3=p.Nc_ref,T_ref_K=p.T_ref,mass_ratio=p.electron_mass_ratio)
        fd=(perturbed(plus).values-perturbed(minus).values)/(2*h)
        error=float(np.max(abs(fd-batch.jacobian[:,:,k])/(1+abs(fd)+abs(batch.jacobian[:,:,k]))))
        checks.append(error<2e-6);records.append(dict(kind='local_fd',coordinate=k,error=error))
    # Independent finite-field laminate constitutive equations, not log averages.
    rng=np.random.default_rng(20261021)
    for trial in range(12):
        N=16;v=np.column_stack([rng.uniform(14,20,N),rng.uniform(-1,2.3,N),np.zeros(N)])
        v[:,2]=np.log10(E_CHARGE)+v[:,0]+v[:,1]
        J=rng.normal(size=(N,3,3));J[:,2]=J[:,0]+J[:,1]
        out=laminate_readout(v,J);sigma=10**v[:,2];RH=1/(E_CHARGE*10**v[:,0]);b=1e-4
        rho=1/sigma;den=rho*rho+(RH*b)**2
        sxx=rho/den;sxy=-RH*b/den;syx=-sxy;syy=sxx
        Ey=-np.mean(syx/sxx)/np.mean(syy-syx*sxy/sxx)
        altRH=-Ey/b;outRH=1/(E_CHARGE*10**out.values[0])
        error=abs(altRH/outRH-1);checks.append(error<2e-12)
        for k in range(3):
            h=1e-5;fd=(laminate_readout(v+h*J[:,:,k],J).values-laminate_readout(v-h*J[:,:,k],J).values)/(2*h)
            ej=float(max(abs(fd-out.jacobian[:,k])));checks.append(ej<2e-8)
        uniform=laminate_readout(np.repeat(v[[0]],N,axis=0),np.repeat(J[[0]],N,axis=0))
        checks.extend([max(abs(uniform.values-v[0]))<2e-12,np.max(abs(uniform.jacobian-J[0]))<2e-12])
        records.append(dict(kind='finite_field_independent_laminate',trial=trial,RH_relative_error=error))
    plans=c.document(c.OLD/'prediction/plan.json')['plans'];lookup={r.case_id:r for r in c.cases().itertuples()}
    for selected in chosen:
        row=lookup[selected['case_id']];plan=next(v for v in plans if v['variant']==selected['variant'] and v['outer_held']==row.related_cluster)
        mapper=SavedJointModel(plan['minimal_fit']);oldpoint=mapper.point(SimpleNamespace(**c.design.inputs(row)),require_observation=False)
        engine=mapper.engine.engine_for(row) if hasattr(mapper.engine,'engine_for') else mapper.engine.engines[row.scenario_pO2_atm]
        layer,solver=engine.layer_for(row);ref=solver.last_inventory.totals.copy();cold=solver.last_prediction;Nd=cold.diagnostics['Nd_cm3']
        dm=layer.db.dm[layer.first];foreign=np.any(solver.parent_atoms>0,axis=1)
        cap=layer.db.prefactor[layer.first]*np.where(foreign,.1*(1-1e-6),1.)
        pars=layer.parameters(T=oldpoint.diagnostics['Tf_K'],Nd=Nd,weights=[0.],
            doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm,Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
        kernel=FrozenChargeKernel(layer,pars);profile=kernel.profile;_,Ec,*_=profile.bands(row.Tobs_K)
        for mode in ['common','structural_contrast']:
            roles=np.ones(len(ref)) if mode=='common' else np.zeros(len(ref))
            if mode=='structural_contrast':
                roles[solver.receivers]=1.;roles[solver.complex]=-1.;roles[solver.native&(dm[:,0]<0)]=-1.
            def run(a,L,reference_stocks=ref,background=Nd):
                stock,stockJ=profiles(reference_stocks,cap,a,roles,16)
                field=solve_poisson(kernel,stock,background_cm3=background,period_nm=L,initial_eV=cold.ef_eV,
                    stock_jacobian=stockJ,background_jacobian=np.array([0.,LN10*background,0.,0.]),log_period_jacobian=np.array([0.,0.,1.,0.]))
                local=local_transport((field.fermi_eV-Ec)/(KB_EV*row.Tobs_K),
                    np.log10(field.local.carriers_cm3[:,0]),np.log10(field.local.Nq2_cm3),row.Tobs_K,
                    Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,mass_ratio=profile.electron_mass_ratio)
                J=np.einsum('nij,njk->nik',local.jacobian,field.log_state_jacobian)
                return field,local,laminate_readout(local.values,J)
            for a in [0.,2.]:
                record=dict(kind='actual_full_chain',variant=selected['variant'],case_id=row.case_id,mode=mode,amplitude=a)
                try:
                    field,local,out=run(a,200.)
                    coldtr=fermi_transport(cold.values[0],cold.values[2],row.Tobs_K,Nc_ref_cm3=profile.Nc_ref,
                        T_ref_K=profile.T_ref,mass_ratio=profile.electron_mass_ratio,extra_mobility_cm2Vs=205.)
                    uniform_error=float(max(abs(out.values-coldtr.values))) if a==0 else None
                    checks.extend([abs(out.diagnostics['Hall_identity_error_dex'])<2e-12,
                        uniform_error is None or uniform_error<2e-8])
                    record.update(success=True,values=out.values.tolist(),uniform_error=uniform_error,
                        maximum_low_kell_moment1_fraction=max(local.diagnostics['local_moment1_fraction_kell_below1']),
                        **out.diagnostics)
                    if a==2.:
                        for k,h in enumerate([1e-4,2e-4,1e-4,2e-5]):
                            def changed(delta):
                                aa=a+delta if k==0 else a;dd=Nd*10**delta if k==1 else Nd
                                ll=200.*10**delta if k==2 else 200.;rr=ref*10**delta if k==3 else ref
                                return run(aa,ll,rr,dd)[2].values
                            fd=(changed(h)-changed(-h))/(2*h);analytic=out.jacobian[:,k]
                            er=float(max(abs(fd-analytic)/(1+abs(fd)+abs(analytic))))
                            checks.append(er<8e-5);derivatives.append(dict(variant=selected['variant'],case_id=row.case_id,mode=mode,coordinate=k,error=er))
                except Exception as exc:checks.append(False);record.update(success=False,error=repr(exc));errors.append(record)
                records.append(record)
        print('transport gate',selected['variant'],row.case_id,flush=True)
    c.write(OUT/'records.json',records);c.write(OUT/'derivatives.json',derivatives)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    result=dict(created_utc=c.e.now(),passed=all(checks),checks=len(checks),failures=sum(not bool(v) for v in checks),
        quadrature_256_512_max_error_dex=quadrature_error,actual_chain_cases=24,actual_chain_derivative_coordinates=len(derivatives),
        max_full_chain_FD_error=max((v['error'] for v in derivatives),default=None),exceptions=errors,
        seconds=time.perf_counter()-start,new_models_fitted=0,goal_complete=False)
    c.write(OUT/'gates.json',result)
    print('Spatial transport gates',len(checks),'failed',result['failures'],flush=True)
    assert all(checks),'Preserve records and use versioned repair'


if __name__=='__main__':main()
