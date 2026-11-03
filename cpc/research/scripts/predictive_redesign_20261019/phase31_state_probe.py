"""Inspect saved predicted defect states before choosing a new rate relation."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from sandbox.joint_saved_model import SavedJointModel

OUT=c.BASE/'phase31_rate_redesign/probe_v1'


def main():
    assert not (OUT/'protocol_before.json').exists(),'Preserve original state probe'
    plans=c.document(c.OLD/'prediction/plan.json')['plans']
    names=('Reference205_internal','JointProcess_nHonly_internal')
    selected=[p for p in plans if p['outer_held'] is not None and p['variant'] in names]
    cases=c.cases();paired=cases[cases.mobility_eligible.eq(True)]
    files=[c.OLD/'prediction/plan.json',c.OLD/'data/scenarios49.csv']
    files += [c.ROOT/p['fit_path'] for p in selected]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),
        purpose='posthoc saved-state inspection to design falsifiable constitutive relation',
        input_hashes=c.file_hashes(set(files)),
        code_hashes=c.file_hashes([Path(__file__),Path(c.__file__),
            c.ROOT/'src/sandbox/joint_saved_model.py']),
        no_new_fit=True,no_parameter_changes=True,retrospective_development=True,
        neutral_complex_population_not_measured_scattering_centers=True,goal_complete=False))
    rows=[]
    for plan in selected:
        mapper=SavedJointModel(plan['minimal_fit'])
        for row in paired[paired.related_cluster.eq(plan['outer_held'])].itertuples():
            point=mapper.point(SimpleNamespace(**c.design.inputs(row)),require_observation=False)
            layer,solver=mapper.engine.engine_for(row).layer_for(row)
            inventory=solver.last_inventory;cold=solver.last_prediction
            complex_states=solver.complex[layer.parent]
            neutral=complex_states & (layer.q==0)
            cq=float(inventory.totals[solver.complex].sum())
            nn=float(cold.populations[neutral].sum())
            rows.append(dict(variant=plan['variant'],case_id=row.case_id,sample_id=row.sample_id,
                family=row.family,related_cluster=row.related_cluster,element=row.element,
                dose_cm3=float(row.incorporated_cm3),
                predicted_log_nH=point.log_nH,predicted_log_mu=point.log_muH,
                free_n_cm3=cold.diagnostics['free_n_cm3'],
                Nq2_cm3=cold.diagnostics['charge_squared_scatter_budget_cm3'],
                complex_center_cm3=cq,neutral_complex_center_cm3=nn,
                neutral_fraction=nn/cq if cq>0 else None,
                cold_EF_eV=cold.ef_eV,complex_parents=int(solver.complex.sum()),
                neutral_complex_charge_rows=int(neutral.sum()),
                observed_log_nH=float(row.target),observed_mu_cm2Vs=float(row.mu_cm2Vs),
                prediction_valid=point.valid))
    assert len(rows)==50
    assert all(0<=r['neutral_complex_center_cm3']<=r['complex_center_cm3']*(1+1e-12) for r in rows)
    c.e.csvwrite(OUT/'saved_state_features.csv',rows)
    c.write(OUT/'audit.json',dict(passed=True,created_utc=c.e.now(),saved_state_evaluations=len(rows),
        new_fits=0,all_parameter_files_unchanged=True,
        neutral_population_is_a_conditional_model_quantity=True,
        no_material_causal_or_prediction_gain_claim=True,
        evidence_sha256=c.file_hashes([OUT/'saved_state_features.csv',OUT/'protocol_before.json']),
        goal_complete=False))
    for r in rows:
        if r['variant']=='Reference205_internal' and 'scenario0' in r['case_id']:
            print(r['sample_id'],'complex',f"{r['complex_center_cm3']:.3e}",
                'neutral',f"{r['neutral_complex_center_cm3']:.3e}",'fraction',
                f"{r['neutral_fraction']:.3e}",flush=True)


if __name__=='__main__':main()
