"""Read actual neutral native/foreign occupations, without inventing centers."""
from pathlib import Path
from types import SimpleNamespace
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from sandbox.joint_saved_model import SavedJointModel

OUT=c.BASE/'native_neutral_state_probe_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    plans=c.document(c.OLD/'prediction/plan.json')['plans']
    selected=[p for p in plans if p['outer_held'] is not None and p['variant'] in
              ('Reference205_internal','JointProcess_nHonly_internal')]
    old=c.read(c.BASE/'phase31_rate_redesign/probe_v2/saved_state_features.csv').set_index(['variant','case_id'])
    inputs=[c.OLD/'prediction/plan.json',c.OLD/'data/scenarios49.csv',
            c.BASE/'phase31_rate_redesign/probe_v2/saved_state_features.csv']
    inputs += [c.ROOT/p['fit_path'] for p in selected]
    code=[Path(__file__),Path(c.__file__)]+list((c.ROOT/'src/sandbox').glob('*.py'))
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),
        code_hashes=c.file_hashes(code),input_hashes=c.file_hashes(set(inputs)),
        reason='test availability of actual neutral-native populations after reading additional-neutral-scattering literature',
        original_complex_probe_did_not_test_all_native_neutral_parents=True,
        no_new_fit_or_population_changes=True,no_scattering_cross_section_inferred=True,
        actual_neutral_occupation_not_measured_scattering_truth=True,goal_complete=False))
    cases=c.cases();paired=cases[cases.mobility_eligible.eq(True)]
    rows=[];checks=[];differences=[]
    for plan in selected:
        mapper=SavedJointModel(plan['minimal_fit'])
        for row in paired[paired.related_cluster.eq(plan['outer_held'])].itertuples():
            point=mapper.point(SimpleNamespace(**c.design.inputs(row)),require_observation=False)
            backend=(mapper.engine.engine_for(row) if hasattr(mapper.engine,'engine_for')
                     else mapper.engine.engines[row.scenario_pO2_atm])
            layer,solver=backend.layer_for(row);inventory=solver.last_inventory;cold=solver.last_prediction
            dm=layer.db.dm[layer.first];native=solver.native;comp=solver.complex;donor=solver.receivers
            other=~(native|comp|donor)
            vo=native&(dm[:,0]==0)&(dm[:,1]==-1)
            vga=native&(dm[:,0]==-1)&(dm[:,1]==0)
            hrelated=dm[:,layer.db.col('H')]>0
            mixedH=hrelated&(dm[:,layer.db.col(row.element)]>0)
            zero=layer.q==0
            def population(mask):return float(cold.populations[zero&mask[layer.parent]].sum())
            nn=population(native);nc=population(comp);nd=population(donor);no=population(other)
            total=float(cold.populations[zero].sum())
            checks.append(abs(nn+nc+nd+no-total)/(1+total)<1e-12)
            achieved=np.bincount(layer.parent,weights=cold.populations,minlength=layer.db.nD)
            checks.append(np.max(abs(achieved-inventory.totals)/(1+inventory.totals))<1e-12)
            oldrow=old.loc[(plan['variant'],row.case_id)]
            err=max(abs(point.log_nH-oldrow.predicted_log_nH),abs(point.log_muH-oldrow.predicted_log_mu))
            differences.append(err);checks.append(err<1e-10)
            rows.append(dict(variant=plan['variant'],case_id=row.case_id,sample_id=row.sample_id,
                family=row.family,related_cluster=row.related_cluster,element=row.element,dose_cm3=row.incorporated_cm3,
                predicted_log_nH=point.log_nH,predicted_log_mu=point.log_muH,
                neutral_all_cm3=total,neutral_native_cm3=nn,neutral_complex_cm3=nc,
                isolated_primary_donor_neutral_cm3=nd,neutral_other_foreign_cm3=no,
                neutral_single_O_vacancy_cm3=population(vo),neutral_single_Ga_vacancy_cm3=population(vga),
                neutral_other_native_cm3=population(native&~(vo|vga)),
                neutral_H_related_cm3=population(hrelated),neutral_mixed_primary_H_cm3=population(mixedH),
                native_parent_total_cm3=float(inventory.totals[native].sum()),
                free_n_cm3=cold.diagnostics['free_n_cm3'],Nq2_cm3=cold.diagnostics['charge_squared_scatter_budget_cm3'],
                observed_log_nH=row.target,observed_mu_cm2Vs=row.mu_cm2Vs,
                prediction_valid=point.valid,unmodeled_scattering_not_identified_by_occupation=True))
    assert len(rows)==50
    c.e.csvwrite(OUT/'native_and_foreign_neutral_states.csv',rows)
    summary=[]
    for variant in sorted({r['variant'] for r in rows}):
        for element in ['Si','Sn']:
            q=[r for r in rows if r['variant']==variant and r['element']==element]
            summary.append(dict(variant=variant,element=element,scenario_states=len(q),
                physical_conditions=len({r['sample_id'] for r in q}),
                neutral_native_min=min(r['neutral_native_cm3'] for r in q),
                neutral_native_max=max(r['neutral_native_cm3'] for r in q),
                neutral_single_OV_min=min(r['neutral_single_O_vacancy_cm3'] for r in q),
                neutral_single_OV_max=max(r['neutral_single_O_vacancy_cm3'] for r in q)))
    c.e.csvwrite(OUT/'summary.csv',summary)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'audit.json',dict(passed=all(checks),checks=len(checks),actual_saved_state_evaluations=50,
        max_old_property_difference=max(differences),new_fits=0,all_parameter_files_unchanged=True,
        neutral_partition_and_parent_conservation_verified=True,
        no_causal_species_cross_section_or_predictive_gain_claim=True,goal_complete=False))
    print('Native-neutral actual saved-state probe',len(checks),'checks;failed',sum(not q for q in checks),flush=True)
    assert all(checks),'Preserve first probe failure before versioned repair'
    for r in summary:print(r,flush=True)


if __name__=='__main__':main()
