"""Actual-state response of candidate chemical inventory fields, without fits."""
from pathlib import Path
from types import SimpleNamespace
import sys,time,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from capped_inventory_projection_v2 import project
from sandbox.joint_saved_model import SavedJointModel
from sandbox.fixed_inventory_charge import InventoryState
from sandbox.coherent_transport import fermi_transport

OUT=c.BASE/'capped_inventory_candidate/state_probe_v1'
FIELDS=[(0.,0.),(-1.,0.),(1.,0.),(-3.,0.),(3.,0.),(0.,-1.),(0.,1.),(0.,-3.),(0.,3.)]
MODES=('foreign_open_host','all_closed_host')


def main():
    assert not (OUT/'protocol_before.json').exists()
    gates=c.BASE/'capped_inventory_candidate/gates_v2/gates.json'
    assert c.document(gates)['passed']
    plans=c.document(c.OLD/'prediction/plan.json')['plans']
    selected=[p for p in plans if p['outer_held'] is not None and p['variant'] in
              ('Reference205_internal','JointProcess_nHonly_internal')]
    code=[Path(__file__),Path(__file__).with_name('capped_inventory_projection_v2.py'),Path(c.__file__)]
    code+=list((c.ROOT/'src/sandbox').glob('*.py'))
    inputs=[gates,c.OLD/'prediction/plan.json',c.OLD/'data/scenarios49.csv']+[c.ROOT/p['fit_path'] for p in selected]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes(code),
        input_hashes=c.file_hashes(inputs),fields=FIELDS,modes=MODES,
        fields_definition=['all native parent count','native Ga-deficit with no O change'],
        planned_base_states=50,planned_field_solves=900,planned_FD_coordinates=200,
        candidate_fields_dimensionless_not_measured_formation_energy=True,
        no_outcome_labels_for_field_choices=True,no_parameters_optimized=True,
        full_upstream_trained_model_not_implemented=True,goal_complete=False))
    rows=[];derivatives=[];start=time.monotonic();zero_errors=[];cold_calls=0;projection_calls=0
    cases=c.cases();paired=cases[cases.mobility_eligible.eq(True)]
    for plan in selected:
        mapper=SavedJointModel(plan['minimal_fit'])
        for row in paired[paired.related_cluster.eq(plan['outer_held'])].itertuples():
            old=mapper.point(SimpleNamespace(**c.design.inputs(row)),require_observation=False)
            engine=(mapper.engine.engine_for(row) if hasattr(mapper.engine,'engine_for') else mapper.engine.engines[row.scenario_pO2_atm])
            layer,solver=engine.layer_for(row);inventory=solver.last_inventory;original=solver.last_prediction
            dm=layer.db.dm[layer.first];foreign=solver.parent_atoms.T
            isforeign=np.any(foreign>0,axis=0);caps=layer.db.prefactor[layer.first]*np.where(isforeign,.1*(1-1e-6),1.)
            D=np.c_[solver.native.astype(float),(solver.native&(dm[:,0]<0)&(dm[:,1]==0)).astype(float)]
            pars=layer.parameters(T=old.diagnostics['Tf_K'],Nd=original.diagnostics['Nd_cm3'],weights=[0.],
                doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
                pO2=row.scenario_pO2_atm,Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
            chain=np.zeros((layer.np,2));zeroJ=np.zeros((len(inventory.totals),2))
            profile=engine.profile
            mu0=205. if plan['minimal_fit']['method']=='Reference205' else 205.*10.**float(plan['minimal_fit']['parameters'][-1])
            for mode in MODES:
                B=foreign if mode=='foreign_open_host' else np.vstack([dm[:,:2].T,foreign])
                def point(coordinates):
                    nonlocal cold_calls,projection_calls
                    projection_calls+=1
                    p=project(inventory.totals,caps,B,D@coordinates,reference_jacobian=zeroJ,field_jacobian=D)
                    inv=InventoryState(p.concentrations,p.jacobian,dict(inventory.diagnostics,
                        candidate_projection_scope=mode,statistical_field_not_measured_free_energy=True))
                    cold_calls+=1;z=solver.solve(pars,inv,chain)
                    tr=fermi_transport(z.values[0],z.values[2],row.Tobs_K,Nc_ref_cm3=profile.Nc_ref,
                        T_ref_K=profile.T_ref,mass_ratio=profile.electron_mass_ratio,extra_mobility_cm2Vs=mu0,hall=True)
                    J=tr.jacobian[:,:2]@z.jacobian[[0,2]]
                    return p,z,tr,J
                base=None
                for fields in FIELDS:
                    record=dict(variant=plan['variant'],case_id=row.case_id,sample_id=row.sample_id,
                        element=row.element,mode=mode,native_field=fields[0],Ga_deficit_field=fields[1],
                        original_log_nH=old.log_nH,original_log_mu=old.log_muH,
                        original_free_n_cm3=original.diagnostics['free_n_cm3'],
                        original_Nq2_cm3=original.diagnostics['charge_squared_scatter_budget_cm3'])
                    try:
                        p,z,tr,J=point(np.array(fields))
                        record.update(success=True,valid_Hall=bool(tr.values[0]>=13 and z.diagnostics['single_electron_Hall_valid']),
                            log_nH=float(tr.values[0]),log_mu=float(tr.values[1]),log_sigma=float(tr.values[2]),
                            delta_log_nH=float(tr.values[0]-old.log_nH),delta_log_mu=float(tr.values[1]-old.log_muH),
                            free_n_cm3=z.diagnostics['free_n_cm3'],Nq2_cm3=z.diagnostics['charge_squared_scatter_budget_cm3'],
                            logn_sensitivity_native=float(J[0,0]),logmu_sensitivity_native=float(J[1,0]),
                            logn_sensitivity_Ga_deficit=float(J[0,1]),logmu_sensitivity_Ga_deficit=float(J[1,1]),
                            projection_iterations=p.diagnostics['iterations'],
                            projection_budget_error=p.diagnostics['chemistry_relative_error'],
                            projection_derivative_backward_error=p.diagnostics['derivative_chemistry_relative_error'])
                        if fields==(0.,0.):
                            err=float(max(abs(tr.values[:2]-np.array([old.log_nH,old.log_muH]))));zero_errors.append(err)
                            record['zero_reconstruction_error']=err;base=(tr.values.copy(),J.copy())
                    except Exception as exc:
                        record.update(success=False,error=f'{type(exc).__name__}: {exc}')
                    rows.append(record)
                if base is not None:
                    for k in range(2):
                        entry=dict(variant=plan['variant'],case_id=row.case_id,mode=mode,field_coordinate=k)
                        try:
                            h=2e-5;step=np.eye(2)[k]*h
                            plus=point(step)[2].values;minus=point(-step)[2].values
                            fd=(plus-minus)/(2*h);analytic=base[1][:,k]
                            err=float(max(abs(fd-analytic)/(1+abs(fd)+abs(analytic))))
                            entry.update(success=True,error=err,passed=bool(err<3e-5),
                                finite_difference=fd.tolist(),analytic=analytic.tolist())
                        except Exception as exc:entry.update(success=False,passed=False,failure=repr(exc))
                        derivatives.append(entry)
            print('cap actual-state response',plan['variant'],row.case_id,'basecases',len(rows),flush=True)
    assert len(rows)==900
    c.e.csvwrite(OUT/'field_responses.csv',rows);c.write(OUT/'actual_state_derivatives.json',derivatives)
    summary=[]
    for variant in sorted({r['variant'] for r in rows}):
        for mode in MODES:
            good=[r for r in rows if r['variant']==variant and r['mode']==mode and r['success']]
            summary.append(dict(variant=variant,mode=mode,planned_field_cases=225,successful_cases=len(good),
                valid_Hall_cases=sum(r['valid_Hall'] for r in good),
                min_delta_log_nH=min((r['delta_log_nH'] for r in good),default=None),
                max_delta_log_nH=max((r['delta_log_nH'] for r in good),default=None),
                min_delta_log_mu=min((r['delta_log_mu'] for r in good),default=None),
                max_delta_log_mu=max((r['delta_log_mu'] for r in good),default=None)))
    c.e.csvwrite(OUT/'summary.csv',summary)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'audit.json',dict(created_utc=c.e.now(),field_cases=len(rows),
        successful_field_cases=sum(r['success'] for r in rows),zero_reconstructions=len(zero_errors),
        max_zero_reconstruction_error=max(zero_errors,default=None),
        derivative_coordinates=len(derivatives),passed_derivative_coordinates=sum(r['passed'] for r in derivatives),
        actual_projection_calls=projection_calls,actual_cold_solve_calls=cold_calls,
        failed_cases_preserved=True,seconds=time.monotonic()-start,new_fits=0,
        no_predictive_accuracy_or_microphysical_identification_claim=True,goal_complete=False))
    for r in summary:print(r,flush=True)


if __name__=='__main__':main()
