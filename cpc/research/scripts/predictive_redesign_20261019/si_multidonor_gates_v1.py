"""Gauge/stoichiometry gates and fixed-parameter actual-state overlay probe."""
from pathlib import Path
import sys,time,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from sandbox.kroger_db import load,NPZ,DefectDB
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel
from sandbox.coherent_fermi import published_coherent_profile
from si_multidonor_overlay_v1 import make_overlay,pure_mask

OUT=c.BASE/'si_multidonor_overlay_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    original=load();fit,path=c.fit(c.units(),'ShareProcess205',dict(gamma=1.,alpha=.1))
    scenarios=[dict(anchor='existing_pair',attachment_shift_eV=(d,d),prefactor_scale=1.) for d in [-.2,0.,.2]]
    scenarios += [dict(anchor='isolated_species',attachment_shift_eV=(0.,0.),prefactor_scale=g) for g in [.25,1.,4.]]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes([Path(__file__),Path(__file__).with_name('si_multidonor_overlay_v1.py')]),
        input_hashes=c.file_hashes([NPZ,path,c.BASE/'literature_v3/lany2018_nlr.pdf',c.OLD/'data/scenarios49.csv']),
        fixed_shared_stock_fit=fit['key'],scenarios=scenarios,planned_new_state_evaluations=49*7,
        no_refitting=True,no_heldout_or_predictive_gain_claim=True,full_charge_diagrams_not_invented=True,goal_complete=False))
    start=time.perf_counter();checks=[];details=[];metas=[];db0hash=c.e.sha(NPZ);rng=np.random.default_rng(20261021)
    profile=published_coherent_profile();oldmapper=ChargeFeasibleModel(basis='process',profile=profile,db=original)
    originals={}
    for row in c.cases().itertuples():
        p=oldmapper.point(row,fit['parameters'],require_observation=False)
        originals[row.case_id]=(np.r_[p.values[0],p.transport_values],p)
    for scenario_id,scenario in enumerate(scenarios):
        overlay,meta=make_overlay(original=original,**scenario);metas.append(meta)
        for name in ['dHo','charge','dm','cs_ID','prefactor','sum_dm']:
            checks.append(np.array_equal(getattr(original,name),getattr(overlay,name)[:original.nCS]))
        q=np.array([1,-2,-1,0]);dm=np.zeros((4,len(original.elements)),int)
        dm[:,0]=[-1,-2,-3,-4];dm[:,original.col('Si')]=[1,1,2,3]
        nu=np.array([[-1,-1],[-1,0],[1,-1],[0,1]]) # D+C1->C2, C2+D->C3
        checks.extend([np.all(dm.T@nu==0),np.all(q@nu==0)])
        checks.extend([np.array_equal((q*q)@nu,[-4,-2])])
        # Changing both host/foreign chemical reference and charge-energy
        # zero must commute with the attachment-energy construction.
        for repeat in range(12):
            atom_gauge=rng.normal(size=len(original.elements));charge_gauge=float(rng.normal())
            shifted=DefectDB(**{**original.__dict__,'dHo':original.dHo+original.dm@atom_gauge+original.charge*charge_gauge})
            transformed,_=make_overlay(original=shifted,**scenario)
            expected=overlay.dHo+overlay.dm@atom_gauge+overlay.charge*charge_gauge
            error=float(max(abs(transformed.dHo-expected)));checks.append(error<3e-13)
        mapper=ChargeFeasibleModel(basis='process',profile=profile,db=overlay)
        for row in c.cases().itertuples():
            record=dict(scenario=scenario_id,case_id=row.case_id,element=row.element,anchor=scenario['anchor'],
                attachment_shift_eV=scenario['attachment_shift_eV'],prefactor_scale=scenario['prefactor_scale'])
            try:
                p=mapper.point(row,fit['parameters'],require_observation=False);v=np.r_[p.values[0],p.transport_values]
                layer,solver=mapper.engine_for(row).layer_for(row);stock=solver.last_inventory.totals;cold=solver.last_prediction
                qstates=cold.populations;mask=np.array([i in meta['new_parent_ids'] for i in layer.original_parent_ids])
                added_atoms=float(layer.atoms[mask,0]@qstates[mask]) if mask.any() else 0.
                neutral_added=float(qstates[mask&(layer.q==0)].sum())
                delta=v-originals[row.case_id][0]
                if row.element=='Sn':checks.append(float(max(abs(delta)))<2e-9)
                checks.extend([cold.diagnostics['parent_probability_error']<2e-12,
                    cold.diagnostics['charge_relative_residual']<2e-8,
                    abs(cold.diagnostics['achieved_concentrations_cm3'][row.element]/row.incorporated_cm3-1)<3e-8])
                _,Ec,*_=profile.bands(300.)
                record.update(success=True,values=v.tolist(),old_values=originals[row.case_id][0].tolist(),
                    output_change_dex=delta.tolist(),observation_valid=p.diagnostics['observation_domain_valid'],
                    new_complex_dopant_atom_fraction=added_atoms/row.incorporated_cm3,
                    new_neutral_complex_dopant_atom_fraction=3*neutral_added/row.incorporated_cm3,
                    new_free_n_cm3=cold.diagnostics['free_n_cm3'],new_Nq2_cm3=cold.diagnostics['charge_squared_scatter_budget_cm3'],
                    EF_minus_Ec_eV=float(cold.ef_eV-Ec),
                    charge_scope_note='n-type fixed-charge template;unreported alternative CTLs not certified',
                    learned_parameters_unchanged=True)
            except Exception as exc:record.update(success=False,error=repr(exc));checks.append(False)
            details.append(record)
        print('Si overlay scenario',scenario_id,'done',flush=True)
    checks.append(c.e.sha(NPZ)==db0hash)
    c.write(OUT/'overlay_metadata.json',metas);c.write(OUT/'actual_state_probe.json',details)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'gates.json',dict(created_utc=c.e.now(),passed=all(checks),checks=len(checks),failures=sum(not bool(v) for v in checks),
        actual_states=len(details),successful_states=sum(v['success'] for v in details),new_fits=0,
        original_state_evaluations=49,new_overlay_state_evaluations=294,seconds=time.perf_counter()-start,
        conditional_cross_method_template=True,no_material_or_predictive_gain_claim=True,goal_complete=False))
    print('Si overlay gates',len(checks),'failed',sum(not bool(v) for v in checks),flush=True)
    assert all(checks),'Preserve failures and use versioned repair'


if __name__=='__main__':main()
