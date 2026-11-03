"""Record a bounded prediction checkpoint while preserving the active goal."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
from csv_io_v2 import read_result_csv as read
BASE=energy.BASE;OUT=BASE/'phase17_inventory_learning'


def main():
    direction=json.loads((OUT/'goal_direction_check.json').read_text())
    assert direction['system_goal_status']=='active' and not direction['goal_complete']
    a=json.loads((OUT/'validation/contract_audit.json').read_text())
    r=json.loads((OUT/'research_runtime_v2/verification.json').read_text())
    q=json.loads((OUT/'property_evaluation/contract_audit.json').read_text())
    assert all(v['passed'] for v in (a,r,q))
    g=json.loads((OUT/'numerical_gates.json').read_text());assert g['passed']
    n=read(OUT/'validation/primary_summary.csv')
    def values(method):return [float(n[n.method.eq(method)&n.scope.eq('all_query')&n.budget.eq(b)].MAE.iloc[0]) for b in (0,1,2)]
    phase=dict(status='complete_bounded_prediction_checkpoint',goal_complete=False,
        actual_shared_inventory_learning=True,foreign_atom_conservation=True,Ga_O_exchange='open_endpoint',
        numerical_checks=len(g['checks']),maximum_derivative_error=g['maximum_scaled_derivative_error'],
        initial_failed_checks_retained=3,initial_runtime_serialization_failure_retained=True,
        base_methods=17,matched_direct_controls=8,family_selectors=6,reported_methods=31,
        nonlinear_new_fits=315,nonlinear_new_optimizer_starts=630,convex_new_fits=180,inherited_references=30,
        final_records=a['fit_records'],unconverged_starts=0,state_failures=a['state_failures'],
        inner_scores=a['inner_scores'],RT_saved_rows=a['RT_saved_rows'],nT_saved_rows=a['nT_saved_rows'],
        missing_primary_queries=a['missing_primary_queries'],original29_5DOI_anchors_common18_preserved=True,
        independent_bundles=r['saved_bundles'],independent_output_rows=r['independent_rows'],
        independent_carrier_error=r['maximum_carrier_error'],independent_transport_error=r['maximum_transport_error'],
        property_tables_read=False,ownT_mu_rows=q['ownT_mu_predictions'],fullchain_RT_query_rows=q['fullchain_RT_query_rows'],
        RT_mu_pairs=5,RT_mu_DOIs=2,muT_reads=54,muT_specimens=3,muT_DOIs=1,new_mu_supervision=False,
        independent_sigma_labels=0,transport_scope_states=q['transport_scope_states'],
        majority_sigma_kl_lt1_states=q['majority_sigma_kl_lt1_states'],material_Born_validity_established=False,
        selected_inventory_original=values('Inventory_selected_original_RT'),selected_inventory_EC=values('Inventory_selected_binding_joint'),
        direct_sigmoid=values('activation_sigmoid_RT'),universal_gain_established=False,
        new_DFT_MD=0,new_material_measurements=0,GPU_used=False,
        next='newversion log-odds inventory allocation and matched direct/output controls,oldstrongreferences retained;notimplemented')
    next_steps=[
        'read docs/ML_SANDBOX_LOG_ODDS_ALLOCATION_NEXT_20261018.md;candidate notimplemented',
        'implement newwrapper composing bound constantcomplex model;gate fullshared/input/source chain and conservation',
        'freeze same29/5/anchors/common18,parameter ranges/starts,andmatched directlogit/output/oldstrongcontrols beforetraining',
        'train actualpropertyprediction and evaluate common18,nH(T),ownTmu/sigma/scope;retainall failures',
        'noinvented phases/energies/thermalhistories;newindependentinput-richvalidation still needed for newdopant ranking']
    for name in ('progress.json','goal_requirements_audit.json'):
        path=BASE/name;data=json.loads(path.read_text());data['phase17']=phase.copy()
        data.update(previous_goal_turn='progress',current_turn='progress',active_jobs=[])
        if name=='progress.json':
            data.update(status='active',goal_complete=False,stage='phase17_complete_bounded_prediction_checkpoint',current_phase=17,
                next=next_steps,latest_utc=energy.now(),goal_direction_checked=True,
                current_report=str((OUT/'ML与Sandbox_守恒库存学习与跨来源预测.pdf').relative_to(ROOT)),
                latest_report_directory=str(OUT.relative_to(ROOT)),current_predictor=str((OUT/'research_runtime_v2').relative_to(ROOT)),
                continuation=str((BASE/'CONTINUE_PHASE17_HERE.md').relative_to(ROOT)),
                next_theory='docs/ML_SANDBOX_LOG_ODDS_ALLOCATION_NEXT_20261018.md')
            if 'phase17_inventory_learning_prediction' not in data['completed']:data['completed'].append('phase17_inventory_learning_prediction')
        else:
            data.update(checked_utc=energy.now(),goal_status='active',objective=direction['system_goal_objective'],
                next_concrete_work=next_steps,sufficient_to_mark_goal_complete=False,
                current_execution='phase17 sharedconservedinventory propertyprediction completed;nextlog-odds predictor notimplemented')
            item=dict(requirement='shared positive and foreign-atom-conserving inventory learning inside actual property prediction',
                evidence=['phase17_inventory_learning/numerical_gates.json','phase17_inventory_learning/validation/contract_audit.json',
                    'phase17_inventory_learning/research_runtime_v2/verification.json','phase17_inventory_learning/property_evaluation/contract_audit.json'],
                scope='31methods,same29/5prediction,savedreplay;nostableoverallgain,newmu/sigma validationabsent')
            if not any(x['requirement']==item['requirement'] for x in data['implemented_and_verified']):data['implemented_and_verified'].append(item)
        manifest=BASE/'artifact_manifest_phase17.json'
        if manifest.exists():
            frozen=json.loads(manifest.read_text());data['phase17_artifact_count']=len(frozen['files']);data['phase17_manifest_sha256']=energy.sha(manifest)
            data['phase17'].update(frozen=True,artifact_count=len(frozen['files']),manifest_sha256=energy.sha(manifest))
        energy.write(path,data)
    print('Phase17 status saved;goalactive,propertyprediction direction preserved,nextcandidate notimplemented')


if __name__=='__main__':main()
