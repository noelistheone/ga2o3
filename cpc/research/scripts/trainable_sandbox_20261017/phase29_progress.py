"""Refresh mutable execution pointers; never changes frozen scientific records."""
from pathlib import Path
import sys,json,shutil,os

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e


def alive(pid):
    path=Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split()[2] not in ('Z','X')


def passed(path):
    if not path.exists():return False
    try:return bool(json.loads(path.read_text()).get('passed',False))
    except json.JSONDecodeError:return None


def document_if_ready(path):
    if not path.exists():return {}
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return {}


def main():
    out=e.BASE/'phase29_joint_scattering';started=json.loads((out/'training_started.json').read_text())
    completion=document_if_ready(e.BASE/'ml_sandbox_goal_completion_receipt_20261019.json')
    goal_done=completion.get('goal',{}).get('status')=='complete'
    goal_state='complete' if goal_done else 'active'
    science=document_if_ready(out/'scientific_interpretation_v1.json')
    goal_audit=document_if_ready(out/'objective_audit_v1.json')
    records=[];inflight=[]
    for path in (out/'fits').glob('*.json'):
        try:records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:inflight.append(str(path.relative_to(ROOT)))
    live=alive(started['pid'])
    complete=(out/'training_complete.json').exists()
    prediction_started=document_if_ready(out/'prediction/execution_started.json')
    prediction_pid=prediction_started.get('pid')
    prediction_live=bool(prediction_pid and alive(prediction_pid))
    train_audit=document_if_ready(out/'training_audit.json')
    training_result=document_if_ready(out/'training_complete.json')
    api_started=document_if_ready(out/'api_replay.log_started.json')
    api_pid=api_started.get('pid');api_live=bool(api_pid and alive(api_pid))
    source_workers=[]
    for path in sorted((out/'prediction/runtime').glob('worker*_started.json')):
        worker=document_if_ready(path)
        if 'pid' not in worker:continue
        log=path.with_name(path.name.replace('_started.json','.log'))
        lines=log.read_text().splitlines() if log.exists() else []
        source_workers.append(dict(pid=worker['pid'],live=alive(worker['pid']),
            tasks=worker['packet_tasks'],done_events=sum(x.startswith('source29 ') for x in lines)))
    api_workers=[]
    for path in sorted((out/'api_replay').glob('worker*_started.json')):
        worker=document_if_ready(path)
        if 'pid' not in worker:continue
        log=path.with_name(path.name.replace('_started.json','.log'))
        lines=log.read_text().splitlines() if log.exists() else []
        api_workers.append(dict(pid=worker['pid'],live=alive(worker['pid']),
            requests=worker['requests'],done_events=sum(x.startswith('API29 ') and
                not x.startswith('API29 complete') for x in lines)))
    status=dict(created_utc=e.now(),goal_status=goal_state,goal_complete=goal_done,
        direction='通过可训练ML＋Sandbox预测掺杂性质;nH主要终点,随后同状态mu及相容sigma',
        theory_requirement='明确符号/单位/参考/假设/推导;独立导数和损失门槛;实际留出预测及物理适用性审计',
        reported_conditions=35,DOIs=7,related_groups=5,paired_reported_conditions=11,scenario_rows=49,
        original33_roles_common20_preserved=True,scenario_weights_per_condition=1.,
        numerical_gates_v1_failure_preserved=True,numerical_gates_v2_passed=True,numerical_checks=480,
        max_scaled_difference=2.074244391434981e-7,saved_source_checks=409,
        API_numerical_gates_passed=passed(out/'api_gates_v1/gates.json'),API_numerical_checks=646,
        API_numerical_maximum_log_difference=1.0582645870726992e-12,
        transport_scope_gates_passed=passed(out/'transport_scope_gates_v1/gates.json'),
        transport_scope_numerical_checks=96,transport_reconstruction_gate_max_error=1.3322676295501878e-15,
        training_pid=started['pid'],training_session=83942,workers=4,training_process_live=live,
        training_complete=complete,fit_records_available_now=len(records),planned_fit_records=1392,
        selected_nonstopped_so_far=sum(not r['selected_success'] for r in records),
        invalid_training_fits_so_far=sum(not r['training_valid'] for r in records),
        primary='JointProcess_internal',frozen_prediction_variants=48,
        actual_outer_prediction_started=(out/'prediction/execution_started.json').exists(),
        actual_prediction_pid=prediction_pid,actual_prediction_process_live=prediction_live,
        actual_prediction_complete=(out/'prediction/execution_complete.json').exists(),
        actual_source_prediction_workers=source_workers,
        actual_source_prediction_tasks=sum(w['tasks'] for w in source_workers),
        actual_completed_source_task_events=sum(w['done_events'] for w in source_workers),
        training_audit_passed=train_audit.get('passed'),
        actual_training_checks=train_audit.get('checks'),
        actual_optimizer_starts=training_result.get('actual_optimizer_starts'),
        actual_objective_forward_calls=train_audit.get('actual_objective_forward_calls'),
        actual_physical_point_calls=train_audit.get('actual_physical_point_calls'),
        shared_predictive_increment='not_established',inflight_incomplete_monitor_reads=inflight,
        pipeline_pid=2950397,pipeline_session=79673,
        pipeline_code_frozen=(out/'pipeline_protocol_before_wait.json').exists(),
        post_prediction_pid=2954829,post_prediction_session=78081,
        post_prediction_code_frozen=(out/'post_prediction_protocol_before_wait.json').exists(),
        actual_API_replay_passed=passed(out/'api_replay/audit.json'),
        actual_API_replay_pid=api_pid,actual_API_replay_process_live=api_live,
        actual_API_workers=api_workers,
        actual_completed_API_request_events=sum(w['done_events'] for w in api_workers),
        independent_prediction_audit_passed=passed(out/'prediction/audit.json'),
        actual_physical_scope_audit_passed=passed(out/'physical_scope/audit.json'),
        actual_information_audit_passed=passed(out/'information_audit/audit.json'),
        actual_post_prediction_checks_passed=passed(out/'post_prediction_independent_checks_complete.json'),
        fixed_state_scattering_attribution_passed=passed(out/'transport_attribution_v1/audit.json'),
        fixed_state_scattering_diagnostic_PDF_review_passed=passed(out/'transport_attribution_v1/visual_review_v1.json'),
        scenario_expanded_Jacobian_not_measured_information=True,
        scatter_budget_notation_audit_passed=passed(out/'scatter_budget_notation_audit_v1/audit.json'),
        scatter_budget_numerical_checks=97,scatter_budget_numerical_states=18,
        actual_total_scatter_budget_includes_singly_ionized_background=True,
        method_V2_math_visual_review_passed=passed(out/'report_preflight_v2/visual_review.json'),
        method_V2_math_pages=4,actual_fitted_gradient_audit_passed=passed(out/'fitted_gradient_audit_v1/audit.json'),
        report_pipeline_pid=2976967,report_pipeline_session=98750,
        report_code_frozen=(out/'report_protocol_before_wait.json').exists(),
        actual_results_report_ready_for_visual_review=(out/'report_ready_for_visual_review.json').exists(),
        actual_results_report_visual_review_passed=passed(out/'report_layout_v2/visual_review_v2.json'),
        actual_full_objective_audit_passed=passed(out/'objective_audit_v1.json'),
        all_research_requirements_evidenced=goal_audit.get('all_requirements_evidenced',False),
        predictive_increment_is_separate_from_research_delivery=True,
        additional_Sn_source_qualification_passed=passed(out/'additional_source_audit_v1/qualification.json'),
        additional_Sn_source_records_reviewed=3 if passed(out/'additional_source_audit_v1/qualification.json') else 0,
        additional_Sn_conditions_admitted=0,
        strict_goal_requirement_R6=science.get('requirements',{}).get('R6',{}).get('status','incomplete'),
        R6_scope='fullchain implementation andconditionalheldout evaluation;not accuratecross-elementvalidation;largeerrors andunknowninputsretained',
        GPU_used=False,GPU_authorized='GPU1only',new_DFT_MD=0,new_material_measurements=0,
        current_entry='results/trainable_sandbox_20261017/'+(
            'CONTINUE_PHASE29_HERE.md' if (e.BASE/'CONTINUE_PHASE29_HERE.md').exists()
            else 'CONTINUE_PHASE29_WORKING.md'))
    backup=out/'mutable_status_before_phase29'
    backup.mkdir(parents=True,exist_ok=True)
    for name in ('progress.json','goal_requirements_audit.json','README.md'):
        source=e.BASE/name;target=backup/name
        if not target.exists():shutil.copyfile(source,target)
    for name in ('progress.json','goal_requirements_audit.json'):
        path=e.BASE/name;value=json.loads(path.read_text())
        value['status']=goal_state;value['goal_complete']=goal_done
        value['previous_goal_turn']='originaltraining,prediction,postchecks andactualreport pipelinescompleted terminal0;no duplicatejobs'
        value['current_turn']='7056predictions/48variants audited;1056APIrequests/9408pointspass;17page654claimreport and3page45claimdiagnostic reviewed;layoutV2same scientificsource,onlyoneheading fixed;manualscientificreview preserves no stablejointgain andconditionalSi/Snscope'
        value['current_phase']='phase29_joint_scattering'
        if live and not complete:value['current_execution']='phase29 actual joint inventory/scattering training running'
        elif prediction_live:value['current_execution']='phase29 originalpipeline held-source prediction running;trainingaudit passed'
        elif api_live:value['current_execution']='phase29 predictions audited;actualsavedAPIreplay running inoriginalpostpipeline'
        elif goal_done:value['current_execution']='fullresearch deliverable completed;negativegain andmateriallimitations explicit;futureexperiments remainproposals'
        elif goal_audit.get('passed'):value['current_execution']='actualfullgoalaudit passed;finalfreeze andmanualcompletiondecision pending'
        elif passed(out/'report_layout_v2/visual_review_v2.json'):value['current_execution']='allactualreportpages reviewed;scientificgoalaudit pending'
        elif (out/'report_ready_for_visual_review.json').exists():value['current_execution']='phase29 reportready;manualallpagereview,scientificgoal auditandfreeze pending'
        elif complete:value['current_execution']='phase29 training complete;inspect original prediction/independent audit/report pipelines'
        else:value['current_execution']='phase29 training process notlive without completion;inspect original handles/logs,do not blindly restart'
        value['current_entry']=status['current_entry'];value['phase29_joint_scattering']=status
        value['active_jobs']=[dict(pid=started['pid'],session_id=83942,workers=4,
            job='phase29_training_v2',resource='CPU')] if live and not complete else []
        if prediction_live:
            value['active_jobs'].append(dict(pid=prediction_pid,parent_pid=2950397,
                parent_session_id=79673,job='phase29_prediction',resource='CPU',
                scope='training-onlynestedselection andfrozen48variant source-excludedprediction;no newtraining'))
        for worker in source_workers:
            if worker['live']:
                value['active_jobs'].append(dict(pid=worker['pid'],parent_pid=prediction_pid,
                    parent_session_id=79673,job='phase29_source_runtime',resource='CPU',
                    tasks=worker['tasks'],completed_events=worker['done_events']))
        if api_live:
            value['active_jobs'].append(dict(pid=api_pid,parent_pid=2954829,
                parent_session_id=78081,job='phase29_api_replay',resource='CPU',
                scope='1056freshAPIrequests;7056heldsource plus2352fulltraining-input points,thelatter notvalidation'))
        for worker in api_workers:
            if worker['live']:
                value['active_jobs'].append(dict(pid=worker['pid'],parent_pid=api_pid,
                    parent_session_id=78081,job='phase29_api_runtime',resource='CPU',
                    requests=worker['requests'],completed_events=worker['done_events']))
        if alive(2950397):
            value['active_jobs'].append(dict(pid=2950397,session_id=79673,
                job='phase29_pipeline_watch',resource='waitthenCPU',
                scope='existingtrainingthenauditand48variantheldsourceprediction;no training restart'))
        if alive(2954829):
            value['active_jobs'].append(dict(pid=2954829,session_id=78081,
                job='phase29_post_prediction_watch',resource='waitthenCPU',
                scope='independent prediction audit,delivery,actual1056APIreplay,physical andinformation audits'))
        if alive(2976967):
            value['active_jobs'].append(dict(pid=2976967,session_id=98750,
                job='phase29_report_watch',resource='waitthenCPU',
                scope='after original independent chain,guide/fullactualreport/independentnumeric audit/PDF;visual review stillmanual'))
        e.write(path,value)
    index=e.BASE/'README.md';text=index.read_text()
    prefix='**当前第29阶段：** '
    if text.startswith(prefix):text=text.split('\n\n',1)[1]
    if status['actual_prediction_complete']:
        leading='35报告条件/11配对的1392记录训练、7056情景预测、1056请求/9408点API回放及报告已完成。17页主报告与3页散射诊断均已全页检查。主联合模型尚无稳定增量，完整Si/Sn性质链仅有条件验证，全部不利结果保留。'
    else:
        leading='35报告条件/11配对的共享库存＋正散射实际训练/预测正在执行；原common20及0/1/2锚点角色保留。'
    goal_note=('完整研究交付已验收，不能解读为新模型精度或创新优势已成立。' if goal_done else
        '系统goal仍active，最终完成由完整验收后人工判断。')
    text=prefix+leading+goal_note+'方向保持ML＋Sandbox预测掺杂性质。[执行入口]('+Path(status['current_entry']).name+')、[目标要求](../../docs/ML_SANDBOX_GOAL_CHECK_PHASE29_20261019.md)。下方为旧阶段状态。\n\n'+text
    index.write_text(text)
    e.write(out/'execution_status_current.json',status)
    print({k:status[k] for k in ('training_process_live','fit_records_available_now','selected_nonstopped_so_far','invalid_training_fits_so_far')},flush=True)


if __name__=='__main__':main()
