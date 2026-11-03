"""Match the full goal to actual evidence; never change goal status automatically."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from phase27_finish import AUDITS, PROTOCOLS, verify_claim, verify_hash_maps
import phase2_energy as e

OUT = e.BASE / 'phase27_charge_feasible'


def actual_json(name):
    path = ROOT / name
    assert path.is_file(), ('Required actual evidence missing', name)
    return json.loads(path.read_text())


def main():
    assert not (OUT / 'objective_audit.json').exists(), 'Preserve the first complete goal audit'
    audits = {name: json.loads((OUT / name).read_text()) for name in
        (*AUDITS, 'capacity_limit/audit.json', 'independent_input_audit/lowdose_transfer/audit.json',
            'rendering_check.json', 'visual_review.json')}
    for name, value in audits.items():
        assert value['passed'], name
        verify_hash_maps(value, name)
    for name in PROTOCOLS: verify_hash_maps(json.loads((OUT / name).read_text()), name)
    goal = json.loads((OUT / 'goal_reconfirmed_during_training.json').read_text())['system_goal']
    assert goal['status'] == 'active'
    assert all(x in goal['objective'] for x in ('ML＋Sandbox', '跨来源预测', '共享', '0/1/2', '计算预算', '目标导向'))
    direction = json.loads((OUT / 'goal_direction_check.json').read_text())
    assert not direction['goal_complete'] and direction['main_variants'] == 48
    receipt = json.loads((OUT / 'analysis/conclusion_receipt.json').read_text())
    assert receipt['fixed_primary'] == 'FeasibleProcess_internal' and not receipt['outer_method_reselected']
    cache = {}
    for claim in receipt['facts']: verify_claim(claim, cache)
    review = receipt['objective_requirement_conclusions']
    assert set(review) == {f'R{i}' for i in range(1, 12)}
    for key, item in review.items():
        assert item['status'] in ('complete', 'complete_with_conditional_limits', 'incomplete'), key
        assert item['reason'] and item['evidence_paths'] and item['scope'], key
        for path in item['evidence_paths']: assert (ROOT / path).is_file(), (key, path)
    table = read(e.BASE / 'phase25_shared_process/data/conditions33.csv')
    assert len(table) == 33 and table.obs_id.nunique() == 33 and table.family.nunique() == 6
    metrics = read(OUT / 'analysis/all_method_metrics.csv')
    assert len(metrics) == 576 and metrics.variant.nunique() == 48
    for scheme in ('DOI', 'cluster'):
        for budget in (0, 1, 2):
            q = metrics[metrics.scheme.eq(scheme) & metrics.budget.eq(budget) & metrics.scope.eq('common20')]
            assert len(q) == 48 and q.conditions.eq(20).all()
    predictions = read(OUT / 'prediction/combined_predictions.csv')
    assert len(predictions) == 9504 and predictions.variant.nunique() == 48
    comparisons = read(OUT / 'analysis/primary_matched_comparisons.csv')
    assert len(comparisons) == 282 and comparisons.control.nunique() == 47
    assert comparisons.grade.str.contains('not confidenceinterval', regex=False).all()
    energy = actual_json('results/trainable_sandbox_20261017/phase2_energy/label_algebra_audit.json')
    acquisition = actual_json('results/trainable_sandbox_20261017/phase2_energy/acquisition/protocol.json')
    aq_gate = actual_json('results/trainable_sandbox_20261017/phase2_energy/acquisition/numerical_gates.json')
    assert aq_gate['passed'] and acquisition['no_material_measurement_labels_used']
    assert not energy['validated_new_DFT_labels']
    aq = read(e.BASE / 'phase2_energy/acquisition/summary.csv')
    assert len(aq) and set(acquisition['elements']) == {'Si', 'Sn'}
    assert (OUT / 'gate_failure_v1').is_dir() and (OUT / 'predeclare_failure_v1').is_dir()
    assert not json.loads((OUT / 'numerical_gates.json').read_text())['passed']
    assert audits['parent_verification_working.json']['entries'] == 52718
    assert audits['parent_verification_working.json']['changed_parent_files'] == 0
    train = audits['training_audit.json']; replay = audits['delivery/cohort_replay/audit.json']
    assert train['fit_records'] == 912 and train['actual_nonlinear_fits'] == 684
    assert replay['point_rows'] == 9504 and replay['requests'] == 1728
    external = json.loads((OUT / 'external_Azizie/execution_complete.json').read_text())
    assert external['rows'] == 5760 and external['group_requests'] == 720
    info = audits['information_audit/audit.json']
    assert info['named_methods'] == 48 and info['budget_rows'] == 144
    assert info['earlier_goal_acquisition_grade'] == acquisition['validation']
    assert not info['rareearth_training_or_validated_prediction_established']
    docs = audits['delivery/documentation_audit.json']
    assert docs['presets'] == 624 and docs['models'] == 390
    assert docs['reported_carbon_disclosed']
    scope = read(OUT / 'input_scope_annotations/all_main_rows.csv')
    assert len(scope) == len(predictions)
    summary = {
        'R1': ('隐式可微的冷热缺陷/输运求解', 'numerical_gates_v2.json',
            '534项检查含参数/输入/来源链及193旧失败状态；不是材料精度。'),
        'R2': ('实际训练并复用跨条件共享内部关系', 'training_audit.json',
            '六新族912记录/684非线性；共享关系改变内部库存。各折参考回退必须照实区分。'),
        'R3': ('实际化学/声明工艺到跨来源载流子预测', 'prediction/audit.json',
            '33条件/6 DOI/5保守组、回顾性任务；未知工艺按明确情景输入。'),
        'R4': ('0/1/2锚点及分层训练内选参', 'prediction/audit.json',
            '每DOI计锚点；同组含两DOI时总预算0/2/4；共同20查询不变。'),
        'R5': ('固定物理、常数、经验和直接ML的公平比较', 'analysis/audit.json',
            '全部48方法、282主方法对照；同锚点/共同集，六新族相同预算；实际算时及能量先验另报。'),
        'R6': ('完整迁移率及相容电导性质链', 'analysis/audit.json',
            '9对/3 DOI，共同4对/3 DOI；用预测状态而非实测nH；sigma派生，无独立sigma标签。'),
        'R7': ('能量参考、观测及物理输入审计', 'information_audit/audit.json',
            '36用能量先验/12无该先验；已知表面C未入物理库存；不能认证真实Tf或Born适用。'),
        'R8': ('目标导向计算选择及同预算对照', 'information_audit/audit.json',
            '先前遮蔽已发表能量的揭示回放、虚拟前向目标；不是新DFT或材料闭环。'),
        'R9': ('根据实际成败修正理论并保留全部边界', 'capacity_limit/audit.json',
            '阈值设计来自193旧失败状态；饱和激活极限须与直接控制区分，全部失败及旧父输入保留。'),
        'R10': ('可保存、可调用、无查询表读取的完整复现', 'delivery/cohort_replay/audit.json',
            '624预设/390文件；1728请求9504点V2独立回放，V3仅补C元数据提示。'),
        'R11': ('理论、结果、PDF及证据支持的下一项改进', 'rendering_check.json',
            '5页理论及完整48方法结果报告；实际逐页查看，全部数值溯源及下一项预测工作明确。')}
    rows = []
    for key, (requirement, audit, scope_text) in summary.items():
        item = review[key]
        rows.append(dict(id=key, requirement=requirement, actual_status=item['status'],
            actual_reason=item['reason'], scope=scope_text, reviewed_scope=item['scope'],
            primary_evidence=str((OUT / audit).relative_to(ROOT)),
            supporting_evidence=item['evidence_paths']))
    complete = all(r['actual_status'] != 'incomplete' for r in rows)
    evidence = {str((OUT / p).relative_to(ROOT)): e.sha(OUT / p) for p in audits}
    for row in rows:
        for path in row['supporting_evidence']: evidence[path] = e.sha(ROOT / path)
    for path in (e.BASE / 'phase2_energy/acquisition/summary.csv',
        e.BASE / 'phase2_energy/acquisition/protocol.json', OUT / 'analysis/conclusion_receipt.json',
        OUT / 'report_claims.json', OUT / 'goal_direction_check.json'):
        evidence[str(path.relative_to(ROOT))] = e.sha(path)
    value = dict(passed=True, all_goal_requirements_evidenced=complete, created_utc=e.now(),
        objective=goal['objective'], requirements=rows, CSV_conclusion_facts_checked=len(receipt['facts']),
        named_methods=48, primary_unchanged=True, predicted_rows=9504, external_rows=5760,
        predictive_increment_is_separate_from_implementation=True,
        no_universal_material_accuracy_or_publication_claim=True,
        final_freeze_and_manual_goal_status_decision_still_required=True,
        evidence_sha256=evidence, code_sha256=e.sha(Path(__file__)), goal_complete=False)
    e.write(OUT / 'objective_audit.json', value)
    text = ['# 完整目标逐项验收', '', '系统goal保持原文。以下结论逐项对应实际文件；实现、预测增量与材料适用性分别判断，最终冻结及系统完成状态另行核对。', '',
        '| 项目 | 实际状态 | 判断与范围 |', '| --- | --- | --- |']
    for row in rows:
        text.append('| ' + row['id'] + ' ' + row['requirement'] + ' | ' + row['actual_status'] + ' | ' +
            row['actual_reason'].replace('|', '/') + '；' + row['scope'].replace('|', '/') + ' |')
    text.extend(['', '所有项目均有证据：' + str(complete) + '。该判断不自动宣称预测胜出，也不自动改变goal状态。', '',
        '下一项可检验预测工作：' + receipt['next_predictive_work'], ''])
    (OUT / '完整目标逐项验收.md').write_text('\n'.join(text))
    print('Actual full objective audit', len(rows), 'requirements;all evidenced', complete,
        ';manual goal completion decision not automated', flush=True)


if __name__ == '__main__': main()
