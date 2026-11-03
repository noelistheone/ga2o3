"""Update mutable continuation/goal status without changing frozen science."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy

BASE=energy.BASE;OUT=BASE/'phase16_thermal_joint'


def main():
    checks=json.loads((OUT/'goal_direction_check.json').read_text())
    assert checks['system_goal_status']=='active' and not checks['goal_complete']
    audit=json.loads((OUT/'validation/contract_audit.json').read_text())
    rt=json.loads((OUT/'research_runtime/verification.json').read_text())
    assert audit['passed'] and rt['passed']
    phase=dict(complete_bounded_checkpoint=True,goal_complete=False,
        thermal_specimens=29,point_or_cohort_records=18,interval_only=11,complete_actual_peak_histories=0,
        final_numerical_checks=945,retained_initial_failed_checks=76,new_fits=150,new_optimizer_starts=300,
        inherited_fits=30,all_final_records=180,new_start_nonconverged=0,state_failures=0,
        inner_scores=3600,RT_saved_rows=1044,nT_saved_rows=5364,missing_queries=0,
        original29_5DOI_anchors_common18_preserved=True,independent_bundles=180,independent_rows=8280,
        independent_max_state_error=rt['maximum_state_error'],independent_max_transport_error=rt['maximum_transport_error'],
        muRT_pairs=5,muRT_DOIs=2,muT_reads=54,muT_specimens=3,muT_DOIs=1,independent_sigma_labels=0,
        simple_mu_control_fits=135,post_primary_shared_state_mu_heads=60,post_primary_shared_state_mu_head_rows=708,
        actual_mu_states_scope_checked=708,majority_conductivity_kl_below1=338,
        all90_capped_fits_at_T_and_background_upper=True,
        cap_carrier0_original=2.8688898,cap_carrier0_EC=3.1974728,
        effective_original_gamma_point1_carrier=[.270200,.258748,.265949],
        effective_EC_gamma1_nH0=.345127,effective_EC_gamma1_muT0=.153771,effective_EC_gamma1_muRT0=1.082635,
        universal_prediction_gain_established=False,complete_actual_thermal_history_claimed=False,
        new_DFT_MD=0,new_material_measurements=0,GPU_used=False,
        next='implement/gate shared non-equilibrium native inventory closure under explicit thermal scenarios;notimplemented;positive extra_mu0 downstream')
    # Use exact table values for claims rather than the rounded prose defaults.
    from csv_io_v2 import read_result_csv as read
    n=read(OUT/'validation/primary_summary.csv')
    for name,method in [('cap_carrier0_original','Thermal_original_RT_record_cap_g0'),('cap_carrier0_EC','Thermal_binding_joint_record_cap_g0')]:
        phase[name]=float(n[n.method.eq(method)&n.budget.eq(0)&n.scope.eq('all_query')].MAE.iloc[0])
    next_steps=[
        'read docs/ML_SANDBOX_INVENTORY_CLOSURE_NEXT_20261018.md;new inventory/charge layer not implemented',
        'prove and gate positive native stock factor,Si/Sn/Hinventoryconservation,conditionaluniquecharge root and fullIFT/zero replay',
        'fix thermal scenario andsmallsharednative-survival parameters before newtraining;no fabricated actualTmax or microscopic rates',
        'same29/5/anchors/common18,strongoldbaselines,same-capacity output/directML andmatchedn-mu label budgets',
        'retain transport scope;weakscattering invalidity is not curedbypositive rates;learnedextra_mu0 downstream']
    report=str((OUT/'ML与Sandbox_温度闭合与联合性质预测.pdf').relative_to(ROOT))
    for filename in ('progress.json','goal_requirements_audit.json'):
        path=BASE/filename;data=json.loads(path.read_text());data['phase16']=phase
        data['previous_goal_turn']='progress';data['current_turn']='progress';data['active_jobs']=[]
        if filename=='progress.json':
            data.update(status='active',goal_complete=False,stage='phase16_complete_bounded_prediction_checkpoint',current_phase=16,
                next=next_steps,latest_utc=energy.now(),current_report=report,latest_report_directory=str(OUT.relative_to(ROOT)),
                current_predictor=str((OUT/'research_runtime').relative_to(ROOT)),
                continuation=str((BASE/'CONTINUE_PHASE16_HERE.md').relative_to(ROOT)),
                next_theory='docs/ML_SANDBOX_INVENTORY_CLOSURE_NEXT_20261018.md',goal_direction_checked=True)
            if 'phase16_thermal_joint_prediction' not in data['completed']:data['completed'].append('phase16_thermal_joint_prediction')
        else:
            data.update(checked_utc=energy.now(),goal_status='active',objective=checks['system_goal_objective'],
                next_concrete_work=next_steps,sufficient_to_mark_goal_complete=False,
                current_execution='phase16 jointnH/mu thermalclosure completed;next sharednativeinventory predictor notimplemented')
            requirement=dict(requirement='thermal input closure and actual shared nH/mu prediction training',
                evidence=['phase16_thermal_joint/prediction_protocol.json','phase16_thermal_joint/validation/contract_audit.json',
                    'phase16_thermal_joint/research_runtime/verification.json','phase16_thermal_joint/property_evaluation/summary.csv',
                    'phase16_thermal_joint/transport_scope/contract_audit.json'],
                scope='same29/5prediction,temperature/mu tradeoffs retained;not actualthermalhistory or universalpropertygain')
            if not any(x['requirement']==requirement['requirement'] for x in data['implemented_and_verified']):data['implemented_and_verified'].append(requirement)
        manifest=BASE/'artifact_manifest_phase16.json'
        if manifest.exists():
            frozen=json.loads(manifest.read_text());data['phase16_artifact_count']=len(frozen['files']);data['phase16_manifest_sha256']=energy.sha(manifest)
            data['phase16'].update(frozen=True,artifact_count=len(frozen['files']),manifest_sha256=energy.sha(manifest))
        energy.write(path,data)
    print('Mutable status updated;goalactive,phase16prediction checkpoint,phase17inventory candidatepending')


if __name__=='__main__':main()
