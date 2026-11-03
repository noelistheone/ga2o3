"""Reconcile actual background/defect scatter budgets; no model changes."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.joint_saved_model import SavedJointModel
from sandbox.context_saved_model import INPUT_NAMES
import phase29_data as data
import phase2_energy as e

OUT=data.OUT/'scatter_budget_notation_audit_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    parent=data.OUT/'saved_source_gates_v1/packet.json'
    code=[Path(__file__),ROOT/'src/sandbox/fixed_inventory_charge.py',
        ROOT/'docs/ML_SANDBOX_SCATTER_BUDGET_CORRECTION_PHASE29_20261019.md',
        ROOT/'docs/ML_SANDBOX_METHOD_PHASE29_V2_20261019.md']
    e.write(OUT/'protocol_before.json',dict(created_utc=e.now(),purpose='background included in actual total Nq2;notation correction only',
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(parent.relative_to(ROOT)):e.sha(parent)},
        artificial_background_coordinates=[14.,17.5],reconciliation_tolerance=1e-12,
        no_training_or_heldout_prediction=True,no_material_accuracy_claim=True,goal_complete=False))
    packet=json.loads(parent.read_text());cases=data.load_cases()
    H=cases[cases.H_known.eq(True)].iloc[0]
    inputs=[SimpleNamespace(**{k:getattr(H,k) for k in INPUT_NAMES})]
    inputs += [SimpleNamespace(**packet['tasks'][0]['nodes'][i]) for i in (0,13)]
    checks=[];states=[]
    def add(name,error):checks.append(dict(name=name,passed=bool(error<=1e-12),error=float(error)))
    for name in ('Reference205_offset','ReferenceTau_internal','JointProcess_internal'):
        task=next(t for t in packet['tasks'] if t['id']==name)
        for background in (14.,17.5):
            fit=dict(task['fit']);fit['parameters']=list(fit['parameters']);fit['parameters'][1]=background
            mapper=SavedJointModel(fit)
            for row in inputs:
                p=mapper.point(row,require_observation=False)
                engine=mapper.engine.engine_for(row) if hasattr(mapper.engine,'engine_for') else mapper.engine.engines[row.scenario_pO2_atm]
                layer,solver=engine.layer_for(row);cold=solver.last_prediction
                explicit_q2=float((layer.q**2)@cold.populations);explicit_abs=float(abs(layer.q)@cold.populations)
                bg=float(cold.diagnostics['Nd_cm3']);total=explicit_q2+bg
                observed=float(p.diagnostics['coherent_transport']['Nq2_cm3'])
                add('total_scatter_budget_includes_background',abs(observed/total-1))
                add('cold_log_scatter_budget_includes_background',abs(cold.values[2]-np.log10(total)))
                add('cold_absolute_charge_includes_background',abs(cold.values[1]-np.log10(explicit_abs+bg)))
                states.append(dict(method=fit['method'],element=row.element,H_known=bool(row.H_known),
                    incorporated_cm3=float(row.incorporated_cm3),background_cm3=bg,
                    explicit_defect_q2_cm3=explicit_q2,total_Nq2_cm3=observed,
                    background_fraction_of_total=float(bg/total),artificial_numerical_parameters_not_trained=True))
    protocol=json.loads((data.OUT/'post_prediction_protocol_before_wait.json').read_text())
    for name,digest in protocol['bound_files'].items():
        add('bound_implementation_unchanged',0. if e.sha(ROOT/name)==digest else np.inf)
    e.write(OUT/'numerical_states.json',states)
    value=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,
        numerical_states=len(states),background_is_assumed_singly_ionized=True,
        no_code_model_parameter_or_prediction_changes=True,new_training=0,
        raw_frozen_formula_omission_preserved_with_explicit_erratum=True,goal_complete=False)
    e.write(OUT/'audit.json',value)
    print('Scatter budget notation audit',value['passed'],len(checks),'states',len(states),flush=True)
    if not value['passed']:raise RuntimeError('Preserve scatter-budget reconciliation failure')


if __name__=='__main__':main()
