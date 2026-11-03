"""Vector-background regression, charge conservation and full field derivatives."""
from pathlib import Path
from types import SimpleNamespace
import sys,time,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from spatial_local_charge_v1 import FrozenChargeKernel as ScalarKernel
from spatial_local_charge_v2 import FrozenChargeKernel
from spatial_poisson_v1 import solve_poisson as scalar_poisson
from spatial_poisson_v2 import solve_poisson
from spatial_parent_profile_v1 import parent_profile
from spatial_poisson_gates_v1 import profiles
from sandbox.joint_saved_model import SavedJointModel
from sandbox.kroger_db import N_SITE

OUT=c.BASE/'spatial_predictive_candidate/vector_background_gates_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    chosen=c.document(c.BASE/'spatial_predictive_candidate/local_charge_gates_v2/protocol_before.json')['selected_cases']
    code=[Path(__file__),*[Path(__file__).with_name(q) for q in ['spatial_local_charge_v2.py','spatial_poisson_v2.py','spatial_parent_profile_v1.py']]]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes(code),
        input_hashes=c.file_hashes([c.OLD/'prediction/plan.json',c.OLD/'data/scenarios49.csv']),selected_cases=chosen,
        grid_size=32,amplitudes=[0.,2.,16.],periods_nm=[200.,1000.],FD_coordinates=5,
        background_count_conserved_not_claimed_known_chemical_species=True,goal_complete=False))
    start=time.perf_counter();checks=[];records=[];derivs=[]
    lookup={r.case_id:r for r in c.cases().itertuples()};plans=c.document(c.OLD/'prediction/plan.json')['plans']
    for chosen_case in chosen:
        row=lookup[chosen_case['case_id']];plan=next(p for p in plans if p['variant']==chosen_case['variant'] and p['outer_held']==row.related_cluster)
        mapper=SavedJointModel(plan['minimal_fit']);oldp=mapper.point(SimpleNamespace(**c.design.inputs(row)),require_observation=False)
        engine=mapper.engine.engine_for(row) if hasattr(mapper.engine,'engine_for') else mapper.engine.engines[row.scenario_pO2_atm]
        layer,solver=engine.layer_for(row);ref=solver.last_inventory.totals.copy();cold=solver.last_prediction;Nd=cold.diagnostics['Nd_cm3']
        cap=layer.db.prefactor[layer.first]*np.where(np.any(solver.parent_atoms>0,axis=1),.1*(1-1e-6),1.)
        pars=layer.parameters(T=oldp.diagnostics['Tf_K'],Nd=Nd,weights=[0.],
            doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm,Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
        kernel=FrozenChargeKernel(layer,pars);oldkernel=ScalarKernel(layer,pars);roles=np.ones(len(ref));N=32
        for a in [0.,2.,16.]:
            for L in [200.,1000.]:
                entry=dict(variant=chosen_case['variant'],case_id=row.case_id,amplitude=a,period_nm=L)
                try:
                    def field(aa=a,bb=Nd,ll=L,rr=ref,abg=a):
                        stock,J4=profiles(rr,cap,aa,roles,N);stockJ=np.concatenate([J4,np.zeros((*stock.shape,1))],axis=2)
                        bg=parent_profile(np.array([bb]),np.array([N_SITE]),abg,grid_size=N)
                        background=bg.concentrations[0];Jbg=np.zeros((N,5));Jbg[:,1]=bg.reference_partial[0]*np.log(10.)*bb;Jbg[:,4]=bg.amplitude_partial[0]
                        return solve_poisson(kernel,stock,background_cm3=background,period_nm=ll,initial_eV=cold.ef_eV,
                            stock_jacobian=stockJ,background_jacobian=Jbg,log_period_jacobian=np.array([0.,0.,1.,0.,0.])),stock,background
                    out,stock,bg=field();checks.extend([abs(bg.mean()/Nd-1)<2e-10,out.diagnostics['global_charge_relative_error']<2e-8,
                        out.diagnostics['implicit_linear_backward_error']<2e-11])
                    # Backward-compatible scalar interface must retain old roots.
                    if a in [0.,2.]:
                        old=scalar_poisson(oldkernel,stock,background_cm3=Nd,period_nm=L,initial_eV=cold.ef_eV)
                        uniform=solve_poisson(kernel,stock,background_cm3=Nd,period_nm=L,initial_eV=cold.ef_eV)
                        error=float(max(abs(old.fermi_eV-uniform.fermi_eV)));checks.append(error<2e-9);entry['scalar_regression_error']=error
                    if a==2. and L==1000.:
                        for k,h in enumerate([1e-4,2e-4,1e-4,2e-5,1e-4]):
                            def changed(delta):
                                return field(aa=a+delta if k==0 else a,bb=Nd*10**delta if k==1 else Nd,
                                    ll=L*10**delta if k==2 else L,rr=ref*10**delta if k==3 else ref,abg=a+delta if k==4 else a)[0]
                            plus=changed(h);minus=changed(-h);fd=(plus.fermi_eV-minus.fermi_eV)/(2*h)
                            er=float(max(abs(fd-out.fermi_jacobian[:,k])/(1+abs(fd)+abs(out.fermi_jacobian[:,k]))));checks.append(er<3e-5)
                            derivs.append(dict(variant=chosen_case['variant'],case_id=row.case_id,coordinate=k,error=er))
                    entry.update(success=True,**out.diagnostics)
                except Exception as exc:checks.append(False);entry.update(success=False,error=repr(exc))
                records.append(entry)
        print('vector-background gates',chosen_case['variant'],row.case_id,flush=True)
    c.write(OUT/'records.json',records);c.write(OUT/'derivatives.json',derivs);c.verify_hashes(c.document(OUT/'protocol_before.json'))
    result=dict(created_utc=c.e.now(),passed=all(checks),checks=len(checks),failures=sum(not bool(q) for q in checks),
        states=len(records),successful_states=sum(r['success'] for r in records),derivative_coordinates=len(derivs),
        max_FD_error=max((r['error'] for r in derivs),default=None),seconds=time.perf_counter()-start,new_fits=0,goal_complete=False)
    c.write(OUT/'gates.json',result);print('Vector background',len(checks),'failed',result['failures'],flush=True)
    assert all(checks),'Preserve failures and use versioned repair'


if __name__=='__main__':main()
