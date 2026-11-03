"""Update mutable goal/continuation metadata after the actual prediction round."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase18_logodds_prediction'


def main():
    direction=json.loads((OUT/'goal_direction_check.json').read_text());assert direction['system_goal_status']=='active'
    a=json.loads((OUT/'validation/contract_audit.json').read_text());s=json.loads((OUT/'source_adaptation_v2/validation/contract_audit.json').read_text())
    r=json.loads((OUT/'research_runtime_v2/verification.json').read_text());q=json.loads((OUT/'property_evaluation/contract_audit.json').read_text())
    u=json.loads((OUT/'source_selected_update_audit.json').read_text());g=json.loads((OUT/'numerical_gates.json').read_text())
    assert all(x['passed'] for x in (a,s,r,q,u,g))
    phase=dict(status='complete_bounded_prediction_checkpoint',goal_complete=False,numerical_checks=len(g['checks']),initial_failed_zero_comparisons_retained=2,
        new_nonlinear_fits=105,new_optimizer_starts=210,inherited_references=15,fit_records=120,new_start_nonconverged=0,state_failures=0,
        base_methods=8,internal_source_variants=6,base_inner_scores=2400,source_inner_scores=1800,source_episodes=300,source_penalty_solutions=2100,
        source_forward_callbacks=26171,source_failed_probes=0,source_selected_boundary_hits=0,one_anchor_infinite_lambda=u['one_anchor_infinite_lambda'],
        RT_saved_rows=1218,legacy_percurve_Tanchor_rows=3576,source_RTspecimenbudget_curve_rows=2556,
        original29_5DOI_anchors_common18_preserved=True,independent_bundles=210,independent_output_rows=9660,
        independent_fixed_carrier_error=r['maximum_fixed_carrier_error'],independent_source_refit_transport_error=r['maximum_source_refit_transport_error'],
        property_tables_read=False,temperature_anchor_protocols_explicitly_separated=True,ownT_mu_rows=2478,sameRTspecimenbudget_nT_rows=5964,
        RT_fullchain_query_rows=154,RT_mu_pairs=5,RT_mu_DOIs=2,muT_reads=54,muT_specimens=3,muT_DOIs=1,new_mu_supervision=False,
        independent_sigma_labels=0,transport_scope_states=472,majority_sigma_kl_lt1_states=119,checked_predictor_domain_checks=24,
        material_Born_validity_established=False,universal_prediction_gain_established=False,confirmed_new_Hall_specimens=0,
        global_five_source_deployment_model=False,new_DFT_MD=0,new_material_measurements=0,GPU_used=False,
        next='fullgoal completion audit;train/export bestsupported full5source predictor/domain;independentactualdoseHall sourceaudit;no automaticcapacityexpansion')
    next_steps=['auditfullgoal requirement-by-requirement usingactualphase1-18evidence;retainfullscope',
        'train/export evidence-supported strong full5source models withunifiedAPI/domain;CVfoldsarenotfulldeployment',
        'auditcachedLeedy2018 DOI10.1063/1.5047214 actualSIMS/Hallpairing andsourceindependence;CV/detectionlimit/alphaphase notexactnHlabels',
        'ifnewlearningjustified,training-onlyregularizedincrement+exactstrongreferencefallback andmatcheddirect/outputbudgets']
    for name in ('progress.json','goal_requirements_audit.json'):
        path=BASE/name;data=json.loads(path.read_text());data['phase18']=phase.copy();data.update(previous_goal_turn='progress',current_turn='progress',active_jobs=[])
        if name=='progress.json':
            data.update(status='active',goal_complete=False,current_phase=18,stage='phase18_complete_bounded_prediction_checkpoint',next=next_steps,
                latest_utc=energy.now(),current_report=str((OUT/'ML与Sandbox_分配学习与来源泛化检验.pdf').relative_to(ROOT)),
                latest_report_directory=str(OUT.relative_to(ROOT)),current_predictor=str((OUT/'research_runtime_v2').relative_to(ROOT)),
                checked_predictor_api='src/sandbox/logodds_predictor_v2.py',continuation=str((BASE/'CONTINUE_PHASE18_HERE.md').relative_to(ROOT)),
                next_theory='docs/ML_SANDBOX_GENERALIZATION_NEXT_20261018.md',goal_direction_checked=True)
            if 'phase18_logodds_prediction' not in data['completed']:data['completed'].append('phase18_logodds_prediction')
        else:
            data.update(checked_utc=energy.now(),goal_status='active',objective=direction['system_goal_objective'],next_concrete_work=next_steps,
                sufficient_to_mark_goal_complete=False,current_execution='phase18actualtrainedlogitandinternaladaptation completed;full5sourcebestpredictor/domainandfullgoal auditnext')
            item=dict(requirement='trained shared log-odds inventory and same-anchor internal source adaptation plus independent property prediction',
                evidence=['phase18_logodds_prediction/validation/contract_audit.json','phase18_logodds_prediction/source_adaptation_v2/validation/contract_audit.json',
                    'phase18_logodds_prediction/research_runtime_v2/verification.json','phase18_logodds_prediction/property_evaluation/contract_audit.json',
                    'phase18_logodds_prediction/temperature_anchor_protocol_audit.json','phase18_logodds_prediction/source_selected_update_audit.json'],
                scope='14methods,same29/5,0-1-2;nooverallgain;oneanchor mostlyzero updates;CVfoldAPI only,notfull5source deployment')
            if not any(x['requirement']==item['requirement'] for x in data['implemented_and_verified']):data['implemented_and_verified'].append(item)
        manifest=BASE/'artifact_manifest_phase18.json'
        if manifest.exists():
            j=json.loads(manifest.read_text());data['phase18_artifact_count']=len(j['files']);data['phase18_manifest_sha256']=energy.sha(manifest)
            data['phase18'].update(frozen=True,artifact_count=len(j['files']),manifest_sha256=energy.sha(manifest))
        energy.write(path,data)
    print('Phase18 complete checkpoint recorded;full prediction goalactive,full5source deployment notyetdelivered')


if __name__=='__main__':main()
