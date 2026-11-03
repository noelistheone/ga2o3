"""Match the original prediction goal to final evidence, without changing it.

Run only after the actual report, manual visual review and scientific review.
Missing evidence is an error, never an inferred success. This audit does not
train models, choose a new primary, change scores or complete the system goal.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import phase2_energy as e
from csv_io_v2 import read_result_csv as read
from phase29_report_claims import audit as report_audit

OUT = e.BASE / 'phase29_joint_scattering'
PRIMARY = 'JointProcess_internal'
TITLE = 'ML与Sandbox_共享库存与正散射联合预测研究'
REQUIREMENTS = {
    'R1': '隐式可微的冷热缺陷及同状态输运求解',
    'R2': '实际训练、保存并复用跨条件共享内部关系',
    'R3': '实际化学与声明工艺到跨来源载流子预测',
    'R4': '相同0/1/2物理条件锚点及训练内分层选择',
    'R5': '固定物理、常数校准、经验与直接ML的公平对照',
    'R6': '使用预测状态的迁移率与相容电导性质链',
    'R7': '能量、观测、组成及物理输入的明确审计',
    'R8': '目标导向计算选择及同预算验证',
    'R9': '根据真实成败修正理论并保留不利结果',
    'R10': '可保存调用、无查询表读取的完整复现',
    'R11': '完整理论、结果、PDF及具体下一项预测改进',
}
AUDITS = [
    'data/qualification.json', 'numerical_gates_v2/gates.json',
    'saved_source_gates_v1/gates.json', 'api_gates_v1/gates.json',
    'transport_scope_gates_v1/gates.json',
    'scatter_budget_notation_audit_v1/audit.json',
    'fitted_gradient_audit_v1/audit.json', 'training_audit.json',
    'prediction/audit.json', 'delivery/audit.json',
    'delivery/documentation_audit.json', 'api_replay/audit.json',
    'physical_scope/audit.json', 'information_audit/audit.json',
    'report_v1/numeric_claims_audit.json', 'report_v1/independent_audit.json',
    'rendering_check_v1.json', 'visual_review_v1.json',
    'report_layout_v2/audit.json', 'report_layout_v2/rendering_check_v2.json',
    'report_layout_v2/visual_review_v2.json',
    'transport_attribution_v1/audit.json',
    'transport_attribution_v1/documentation_audit.json',
    'transport_attribution_v1/rendering_check_v1.json',
    'transport_attribution_v1/visual_review_v1.json',
]


def document(path):
    return json.loads(Path(path).read_text())


def verify_hashes(value):
    """Check declared maps; do not refresh them after a file changes."""
    for key in ('input_hashes', 'code_hashes', 'source_hashes',
            'bound_files', 'output_hashes', 'evidence_sha256'):
        for name, digest in value.get(key, {}).items():
            assert isinstance(digest, str) and len(digest) == 64, (key, name)
            assert e.sha(ROOT / name) == digest, (key, name)


def main():
    target = OUT / 'objective_audit_v1.json'
    assert not target.exists(), 'Preserve previous audit before versioned repair'
    # This is a human scientific interpretation of actual results, not a
    # generated threshold which redefines successful research after seeing MAE.
    review_path = OUT / 'scientific_interpretation_v1.json'
    review = document(review_path)
    assert review['fixed_primary'] == PRIMARY and not review['outer_method_reselected']
    assert review['all_adverse_results_retained'] and review['retrospective_development']
    assert review['primary_prediction_conclusion'] and review['full_chain_conclusion']
    assert review['next_testable_predictive_work'] and review['overall_goal_status'] == 'active'
    assert set(review['requirements']) == set(REQUIREMENTS)
    for key, item in review['requirements'].items():
        assert item['status'] in ('complete', 'complete_with_conditional_limits', 'incomplete'), key
        assert item['reason'] and item['scope'] and item['evidence_paths'], key
        for name in item['evidence_paths']:
            assert (ROOT / name).is_file(), (key, name)
    audits = {name: document(OUT / name) for name in AUDITS}
    for name, value in audits.items():
        assert value['passed'], name
        verify_hashes(value)
    for name in ('training_protocol.json', 'prediction_design_protocol.json',
            'pipeline_protocol_before_wait.json', 'post_prediction_protocol_before_wait.json',
            'report_protocol_before_wait.json'):
        verify_hashes(document(OUT / name))
    goal = document(OUT / 'system_goal_direction_receipt_v1.json')['goal']
    assert goal['status'] == 'active'
    assert all(word in goal['objective'] for word in
        ('ML＋Sandbox', '跨来源预测', '共享', '0/1/2', '计算预算', '目标导向'))
    assert not review.get('new_blind_validation', False)
    assert not review.get('rareearth_validation_established', False)
    visual = audits['visual_review_v1.json']
    rendered = audits['rendering_check_v1.json']
    report = OUT / (TITLE + '.md')
    assert visual['all_pages_actually_reviewed']
    assert visual['pages'] == rendered['pages']
    assert visual['pdf_sha256'] == rendered['pdf_sha256'] == e.sha(report.with_suffix('.pdf'))
    assert visual['source_sha256'] == rendered['source_sha256'] == e.sha(report)
    for name, digest in visual['page_image_sha256'].items():
        assert e.sha(ROOT / name) == digest, name
    claims = document(OUT / 'report_v1/numeric_claims.json')
    assert report_audit(report, claims)['passed']
    layout=OUT/'report_layout_v2'
    verify_hashes(document(layout/'protocol_before.json'))
    lr=document(layout/'rendering_check_v2.json')
    lv=document(layout/'visual_review_v2.json')
    preferred=OUT/(TITLE+'_排版修订V2.md')
    assert e.sha(preferred)==e.sha(report)
    assert lv['all_pages_actually_reviewed'] and lv['pages']==lr['pages']==17
    assert lv['pdf_sha256']==lr['pdf_sha256']==e.sha(preferred.with_suffix('.pdf'))
    assert report_audit(preferred,claims)['passed']
    for name,digest in lv['page_image_sha256'].items():assert e.sha(ROOT/name)==digest,name
    diagnostic=OUT/'transport_attribution_v1'
    for name in ('protocol_before.json','docs_protocol_before.json'):
        verify_hashes(document(diagnostic/name))
    da=document(diagnostic/'audit.json');dv=document(diagnostic/'visual_review_v1.json')
    dr=document(diagnostic/'rendering_check_v1.json')
    assert da['actual_saved_states']==49 and da['transport_evaluations']==98
    assert da['new_fits']==0 and da['new_hot_or_cold_solves']==0
    assert da['original_scores_and_serving_weights_unchanged']
    dm=diagnostic/'固定状态散射诊断.md'
    assert dv['all_pages_actually_reviewed'] and dv['pages']==dr['pages']==3
    assert dv['source_sha256']==dr['source_sha256']==e.sha(dm)
    assert dv['pdf_sha256']==dr['pdf_sha256']==e.sha(dm.with_suffix('.pdf'))
    for name,digest in dv['page_image_sha256'].items():assert e.sha(ROOT/name)==digest,name
    assert report_audit(dm,document(diagnostic/'diagnostic_claims.json'))['passed']
    data = audits['data/qualification.json']
    assert (data['reported_conditions'], data['scenario_rows'], data['paired_reported_conditions']) == (35, 49, 11)
    assert data['DOIs'] == 7 and data['related_groups'] == 5
    assert data['original33_values_and99roles_preserved']
    statistics = document(OUT / 'report_v1/study_statistics.json')
    assert statistics['paired_Si'] == 9 and statistics['paired_Sn'] == 2
    assert statistics['common_carrier_conditions'] == 20
    train = document(OUT / 'training_complete.json')
    assert train['fit_records'] == 1392 and train['nonlinear_fit_records'] == 1200
    predictions = read(OUT / 'prediction/predictions.csv')
    assert len(predictions) == 7056 and predictions.variant.nunique() == 48
    summary = read(OUT / 'prediction/summary.csv')
    assert len(summary) == 576 and summary.variant.nunique() == 48
    common = summary[summary.scope.eq('common20')]
    assert len(common) == 144 and common.conditions.eq(20).all()
    prior = e.BASE / 'phase27_charge_feasible/objective_audit.json'
    energy = e.BASE / 'phase2_energy'
    prior_audit = document(prior)
    assert prior_audit['passed']
    old_r6 = next(x for x in prior_audit['requirements'] if x['id'] == 'R6')
    assert old_r6['actual_status'] == 'incomplete'
    acquisition = document(energy / 'acquisition/protocol.json')
    assert acquisition['no_material_measurement_labels_used']
    assert document(energy / 'acquisition/numerical_gates.json')['passed']
    assert not document(energy / 'label_algebra_audit.json')['validated_new_DFT_labels']
    # Adding two conditional Sn rows does not itself establish accurate,
    # transferable transport or uniquely identified microscopic parameters.
    assert review['new_Sn_process_inputs_incomplete']
    assert review['sigma_is_derived_not_third_label']
    assert not review['material_Born_validity_established']
    requirements = [dict(id=key, requirement=text, **review['requirements'][key])
        for key, text in REQUIREMENTS.items()]
    evidence = {str((OUT / name).relative_to(ROOT)): e.sha(OUT / name) for name in AUDITS}
    for item in requirements:
        for name in item['evidence_paths']: evidence[name] = e.sha(ROOT / name)
    for path in (prior, review_path, energy / 'acquisition/protocol.json',
            energy / 'acquisition/summary.csv', OUT / 'report_v1/numeric_claims.json'):
        evidence[str(path.relative_to(ROOT))] = e.sha(path)
    value = dict(passed=True, created_utc=e.now(), original_objective=goal['objective'],
        requirements=requirements, all_requirements_evidenced=all(
            x['status'] != 'incomplete' for x in requirements),
        primary=PRIMARY, new_Sn_pairs_not_automatically_goal_completion=True,
        implementation_is_separate_from_predictive_increment=True,
        primary_prediction_conclusion=review['primary_prediction_conclusion'],
        full_chain_conclusion=review['full_chain_conclusion'],
        next_testable_predictive_work=review['next_testable_predictive_work'],
        predictive_conclusions_are_manual_and_source_bound=True,
        report_claims_rechecked=len(claims['claims']), evidence_sha256=evidence,
        code_sha256=e.sha(Path(__file__)), goal_complete=False,
        system_goal_status_not_changed=True)
    e.write(target, value)
    lines = ['# 第29阶段：完整研究目标逐项验收', '',
        '原目标保持ML＋Sandbox掺杂性质预测。实现、留出增量与材料适用性分别判断；本验收不自动完成系统goal。', '',
        '| 要求 | 证据状态 | 判断及范围 |', '| --- | --- | --- |']
    for item in requirements:
        lines.append('| ' + item['id'] + ' ' + item['requirement'] + ' | ' + item['status'] +
            ' | ' + (item['reason'] + '；' + item['scope']).replace('|', '/') + ' |')
    lines.extend(['', '载流子主终点：' + review['primary_prediction_conclusion'], '',
        '完整性质链：' + review['full_chain_conclusion'], '',
        '下一项可检验的预测工作：' + review['next_testable_predictive_work'], ''])
    (OUT / '完整目标逐项验收.md').write_text('\n'.join(lines))
    print('Objective29 actual eleven requirements audited; system goal unchanged', flush=True)


if __name__ == '__main__':
    main()
