"""Evidence-specific conclusions from completed main and supplemental results."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from phase27_report import VARIANTS, PRIMARY, AUDITS
import phase2_energy as e

OUT = e.BASE / 'phase27_charge_feasible'
FACTS = []
CACHE = {}


def value(path, selector, column, integer=False):
    p = OUT / path
    if p not in CACHE: CACHE[p] = read(p)
    frame = CACHE[p]
    for key, wanted in selector.items(): frame = frame[frame[key].eq(wanted)]
    assert len(frame) == 1, (path, selector)
    number = float(frame[column].iloc[0])
    actual = number if np.isfinite(number) else None
    FACTS.append(dict(path=str(p.relative_to(ROOT)), sha256=e.sha(p),
        selector=selector, column=column, value=actual))
    return '未完整覆盖' if actual is None else (str(int(actual)) if integer else f'{actual:.6f}')


def metric(variant, budget, column='DOI_macro_nH_MAE', scheme='cluster'):
    return value('analysis/all_method_metrics.csv',
        dict(scheme=scheme, variant=variant, budget=budget, scope='common20'), column)


def external(variant, budget, tier='learned_process_proxy_12'):
    return value('external_Azizie/summary.csv', dict(context='cluster::without:related_AFRL_Cornell_Wright',
        variant=variant, budget=budget, scope='common_query', input_tier=tier), 'nH_MAE')


def main():
    assert not (OUT / 'analysis/conclusion_receipt.json').exists()
    for name in (*AUDITS, 'capacity_limit/audit.json', 'selection_readback/audit.json',
        'independent_input_audit/lowdose_transfer/audit.json'):
        assert json.loads((OUT / name).read_text())['passed'], name
    assert (OUT / 'pipeline_complete.json').exists()
    blocks = ['### 8.1 实际判断与主方法边界',
        '本轮完成了实际可训练的内部库存关系、相同容量的端点直接读出控制、全部48方法比较及独立保存复用。数值可行性问题得到实质改善，但没有建立新增共享关系的稳定跨来源增量。FixedPhysical也是在训练来源拟合有效Tf与背景的两参数物理参考，不是完全没有材料标签的第一性原理预测。“零锚点”只指新来源没有适应标签。']
    blocks.append('在同一共同20查询、关联组排除、6 DOI等权下：')
    table = ['| 方法 | 0锚点 | 1锚点 | 2锚点 |', '| --- | --- | --- | --- |']
    for variant in (PRIMARY, 'FixedPhysical', 'ShareEffectiveProcess_internal',
        'ActivationProcess', 'MedianActivation', 'UnitDose', 'DirectRidge'):
        table.append('| ' + variant + ' | ' + ' | '.join(metric(variant, b) for b in (0, 1, 2)) + ' |')
    blocks.append('\n'.join(table))
    blocks.append('主方法在该划分与旧参考回退逐来源持平；有锚点时简单激活校准更强。另一方面，零锚点物理参考相对DirectRidge在5组等权误差上的差为' +
        value('analysis/primary_matched_comparisons.csv', dict(scheme='cluster', budget=0, control='DirectRidge'), 'delta_group_macro') +
        ' dex，去一组差值范围为[' +
        value('analysis/primary_matched_comparisons.csv', dict(scheme='cluster', budget=0, control='DirectRidge'), 'leave_one_group_delta_min') + ', ' +
        value('analysis/primary_matched_comparisons.csv', dict(scheme='cluster', budget=0, control='DirectRidge'), 'leave_one_group_delta_max') +
        ']。这支持当前物理参考有用，不能把参考的作用算作新共享函数的贡献，也不是小队列上的显著性结论。')
    blocks.append('DOI排除时，主方法为' + '/'.join(metric(PRIMARY, b, scheme='DOI') for b in (0, 1, 2)) +
        '，旧原型为' + '/'.join(metric('ShareEffectiveProcess_internal', b, scheme='DOI') for b in (0, 1, 2)) +
        ' dex；这些相对原型的改善保留。两锚点时主方法也略低于固定参考' +
        metric('FixedPhysical', 2, scheme='DOI') + '，但仍不及UnitDose的' +
        metric('UnitDose', 2, scheme='DOI') + '；不能挑这个局部比较宣称整体优越。')
    blocks.extend(['### 8.2 理论怎样被这次实验修正',
        '选择记录直接区分“只有参考能评估”与“学习候选能评估但仍输给参考”：'])
    t = ['| 阶段/关联组排除 | 选择参考的折数 | 只有参考有限 | 有限学习候选存在 |', '| --- | --- | --- | --- |']
    for stage in (26, 27):
        selector = dict(stage=stage, scheme='cluster')
        t.append('| ' + str(stage) + ' | ' +
            value('selection_readback/summary.csv', selector, 'selected_reference', True) + ' | ' +
            value('selection_readback/summary.csv', selector, 'reference_only_finite', True) + ' | ' +
            value('selection_readback/summary.csv', selector, 'contexts_with_any_finite_learned_CV', True) + ' |')
    blocks.append('\n'.join(t))
    blocks.append('因此，新映射已处理前一轮可评分性瓶颈；当前仍不能把结果无增量仅归因于CV状态无效。数据只限制这里的低维函数族、输入与优化预算，不证明所有ML或所有工艺特征无用。两起点及统一细化的停止也不证明全局最优。')
    blocks.append(r'''还存在一个条件性的函数形式边界。固定参考剩余电荷QR、两池总掺杂原子W及每原子电荷qC/qI，有

\[
Q_*(f)=N+Df,\qquad N=Q_R+Wq_C,\qquad D=W(q_I-q_C).
\]

在L=fcrit=-N/D>0的活跃分支，f=L+(1-L)s、s=sigmoid(psi)，于是逐步代入得到

\[
Q_*=D(1-L)s=(D+N)s=(Q_R+Wq_I)s.
\]

QR、N、D和Q*均为元电荷数浓度cm^-3，W为原子数浓度，q为每原子的无量纲电荷；当前孤立施主分支qI=1。可见在固定QR/W条件下，复合池的qC从该阈值容量表达式中消去。不过W与QR本身仍依赖热参考及其余库存，不能说全部缺陷能量都消失。冷电荷再分配损失和Hall因子仍在完整求解中；L=0的非活跃分支也不能套用这个消去式。

固定24个探针的自由电子超额与电荷容量比约0.999926–1，支持这些点接近简单正激活极限，不推广为所有材料。这个条件代数说明为什么需要同信息的直接激活/端点控制；它不独自证明预测失败原因。实际选择ParentReference的折并未使用该新份额映射，更不能把回退预测的表现归因于这个公式。''')
    blocks.extend(['### 8.3 完整性质链有权衡，尚未覆盖Sn迁移率',
        '9组完整化学—nH—mu配对全部是Si，共3 DOI；固定共同查询只剩4组Si。Sn有19个载流子条件，但完整mu配对为0。条件mu的52样本任务是另一队列，不能补成这里的完整Sn验证。'])
    table = ['| 关联组排除的主方法/参考性质误差 | 0锚点 | 1锚点 | 2锚点 |', '| --- | --- | --- | --- |']
    for label, variant, column in [('主方法原生mu', PRIMARY, 'native_mu_MAE'),
        ('参考原生mu', 'FixedPhysical', 'native_mu_MAE'), ('主方法CT(predicted nH)', PRIMARY, 'CT_mu_MAE'),
        ('UnitDose的CT mu', 'UnitDose', 'CT_mu_MAE'), ('主方法派生sigma', PRIMARY, 'native_derived_sigma_MAE')]:
        table.append('| ' + label + ' | ' + ' | '.join(metric(variant, b, column) for b in (0, 1, 2)) + ' |')
    blocks.append('\n'.join(table))
    blocks.append('同状态原生mu相对CT头有条件改善，内部来源更新也会改变mu；与此同时nH查询误差可能增加。这些现象不是新增共享关系优势，绝对mu误差仍大，样品数少；不能把sigma的误差抵消算作第三个独立发现。完整模型198个诊断状态中45个有过半电流权重在kl<1，且198个有效Tf均高于名义生长温度。它们限定有效参考与Born/RTA的解释，不是所有留出样品的适用性证明；全部分数均保留。')
    blocks.extend(['### 8.4 Azizie条件转移仍有明显失败',
        '删除整个关联组的B任务使用过程化学代理，10个共同查询。主方法0/1/2锚点误差为' +
        '/'.join(external(PRIMARY, b) for b in (0, 1, 2)) + '；UnitDose零锚点为' +
        external('UnitDose', 0) + '，两锚点ActivationProcess/MedianActivation为' +
        external('ActivationProcess', 2) + '/' + external('MedianActivation', 2) +
        ' dex。A任务只有一个共同高剂量查询，主方法零锚点' +
        external(PRIMARY, 0, 'rounded_same_Hall_2') + ' dex；UnitDose恰好' +
        external('UnitDose', 0, 'rounded_same_Hall_2') +
        '来自打印约数，不能证明真实100%激活。A/B重叠，过程代理不是同Hall样品的精确化学真值，未知H及O3有效氧情景也未因此被识别。'])
    blocks.extend(['### 8.5 低剂量补充外推：有利证据与边界一起保留',
        'Peterson2024校准条件给出代表性Si约3–4e15 cm^-3和nH=3.96e15 cm^-3；它不同于同文TableI A–E，不能移用其mu。只有1个物理条件，使用3个浓度与2个氧情景得到576行，不是576个独立样品。该文曾被项目引用，声明在主外层启动后完成，本测试不是盲验证。生长950°C与化学剂量均在当前Si训练支持之外；未知零H/C不是测得无杂质。[作者原文pp6–7](https://arxiv.org/pdf/2407.17089v1)',
        '下表完整保留48方法的中心剂量3.5e15 cm^-3误差。F/P分别为完整训练/删除同校旧Sn来源；E/G分别为旧有效氧/理想气体进料分压代理。代理不等于测得的表面平衡化学势。全部三个浓度的采样范围见lowdose_transfer/summary.csv，不是连续区间包络或置信区间。'])
    table = ['| 方法 | F/E | F/G | P/E | P/G |', '| --- | --- | --- | --- | --- |']
    for variant in VARIANTS:
        cells = []
        for context in ('cluster::full', 'cluster::without:10.1063/5.0027870'):
            for oxygen in ('old_effective', 'ideal_feed_proxy'):
                cells.append(value('independent_input_audit/lowdose_transfer/summary.csv',
                    dict(context=context, variant=variant, oxygen_scenario=oxygen), 'central_nH_abs_error'))
        table.append('| ' + variant + ' | ' + ' | '.join(cells) + ' |')
    blocks.append('\n'.join(table))
    blocks.append('物理参考约0.041dex、两种氧情景稳定；简单UnitDose约0.054dex也很接近，差别小于输入范围导致的敏感性。因此这是该物理参考的有条件正面外推证据，不能归功于新增ML。DirectRidge在某些情景严重失配也须保留；不能从1个条件推断普遍优越或新增实验室独立性。')
    blocks.extend(['### 8.6 数据、失败与可复现边界',
        '样品D的已知表面富集C未纳入Si/Sn/H物理库存，主队列和所有分数不删。每DOI0/1/2锚点总计0/6/12；正预算另有继承的1个guard，所以全查询是33/26/20，而非把guard当额外锚点。固定common20不变。36方法使用缺陷能量先验、12方法不使用；相同Hall预算不意味着相同先验信息。目标导向能量选点是遮蔽已发表能量的虚拟前向回放，没有新DFT或新材料测量。',
        '首次预测审计84项失败全部是继承的两侧缺失MAE相减；完整原失败和诊断已保存。V2只增加正确缺失比较、非有限掩码和语义检查，50595项通过，预测/分数未改，旧缺失不填0。低剂量审计首次把训练家族从最小参数文件读取而报KeyError，正确元数据位于清单预设；原错误、路径收据设置错误及误触发的缺协议尝试均保存，V2通过5921项，原576行转移未重跑。主流水线的初次失败事件仍保留，未伪装成一次全部成功。',
        '完整9504点独立接口回放支持保存复用，不代替材料预测验证。624预设中的60个不合格项保留并拒绝；V3只补C范围提示，不能冒称另一次全队列回放。当前没有校准材料置信区间，也未验证稀土、Sn完整mu、真实库存、寿命或PDR。'])
    blocks.extend(['### 8.7 下一项预测研发与整体goal状态',
        '该库存份额参数化的继续扩张到此停止。下一步先核验两篇既有Sn来源中化学浓度、nH与mu是否能可靠配对，再判断能否用电荷第一矩与散射第二矩的联合信息训练共享关系；载流子仍是主终点，必须与同标签预算直接ML及已有同nH散射控制比较。第21阶段的qA=1/3控制已经做过，不换名重复宣称创新。只有配对和物理适用性成立才开展新训练；缺数据时不制造标签。',
        '具体输入门槛、方程和实验分支见docs/ML_SANDBOX_PAIRED_TRANSPORT_NEXT_20261019.md。本阶段研究交付不等于整体目标完成；Sn完整链与新增共享增量仍有缺口，goal保持active。'])
    notes = '\n\n'.join(blocks) + '\n'
    assert not [c for c in notes if ord(c) < 32 and c != '\n']
    dest = OUT / 'analysis/predictive_conclusions.md'; dest.write_text(notes)
    prefix = 'results/trainable_sandbox_20261017/phase27_charge_feasible/'
    requirements = {}
    defs = [
        ('R1', 'complete_with_conditional_limits', '冷热电荷、库存、输运及完整隐式链已有推导与独立数值核对。', '固定目录、分支与RTA近似；不是任意材料的微观真值。', ['numerical_gates_v2.json', 'learning_gates.json', 'theory/ML与Sandbox_电荷可行共享学习理论.pdf']),
        ('R2', 'complete_with_conditional_limits', '六族真实共享训练、保存和留出预测已执行；关联组主方法仍回退。', '实现/训练已证明，泛化增量未成立，不推广到所有ML。', ['training_audit.json', 'selection_readback/summary.csv', 'delivery/cohort_replay/audit.json']),
        ('R3', 'complete_with_conditional_limits', '33条件、6 DOI、5组及额外条件已实际预测并记录全部覆盖。', '回顾性Si/Sn及声明工艺情景；不是新实验室盲验证。', ['prediction/audit.json', 'analysis/all_method_metrics.csv', 'external_Azizie/audit.json']),
        ('R4', 'complete', '相同角色、0/1/2每DOI标签、同common20和训练内选择已独立核对。', '正预算另1guard，总锚点0/6/12；查询组成变化单独报告。', ['delivery/role_counts.csv', 'prediction/audit.json']),
        ('R5', 'complete_with_conditional_limits', '全部48方法及282配对比较保留，六新族相同计算上限，旧成本单独继承。', '相同调用上限不等于实际算时；能量先验另计。', ['analysis/primary_matched_comparisons.csv', 'training_protocol.json', 'information_audit/method_information_budgets.csv']),
        ('R6', 'incomplete', '完整mu配对9组均Si，Sn为0；尚不能完成Sn完整性质链的实证验证。', 'Si common4/3DOI，sigma派生；Born/RTA材料适用未认证。', ['delivery/validation_scope.csv', 'analysis/all_method_metrics.csv', 'physical_scope/audit.json']),
        ('R7', 'complete_with_conditional_limits', '能量参考、观测定义、已知C和输入范围已审计与披露。', '输入未知情景、有效Tf和受限组成不能转成真实微观标签。', ['information_audit/audit.json', 'carbon_scope/audit.json', 'input_scope_annotations/audit.json']),
        ('R8', 'complete_with_conditional_limits', '目标导向计算选择已用已有能量遮蔽揭示和独立虚拟输入回放验证。', '没有新DFT或材料测量；只证明所述条件计算选择。', ['information_audit/audit.json']),
        ('R9', 'complete_with_conditional_limits', '从旧无效状态到电荷阈值改进，再到有限CV仍选参考，全部结果和失败保留。', '没有把数值改善等同于精度；下一步使用可验证配对信息。', ['capacity_limit/audit.json', 'selection_readback/audit.json', 'prediction_audit_failure_v1/diagnosis.json']),
        ('R10', 'complete_with_conditional_limits', '624预设、390文件和9504点无表读取的API回放及使用说明齐备。', '运行需已有且SHA一致的第三方参考；资格不等于精度。', ['delivery/documentation_audit.json', 'delivery/cohort_replay/audit.json', 'delivery/README.md']),
        ('R11', 'complete_with_conditional_limits', '本阶段理论、完整结果与实际后续预测门槛写入报告，整体研发仍未完成。', '需最终渲染/逐页查看及冻结；Sn缺口和共享增量继续留在goal内。', ['ML与Sandbox_电荷可行学习与性质预测.pdf', 'rendering_check.json'])]
    for key, status, reason, scope, paths in defs:
        requirements[key] = dict(status=status, reason=reason, scope=scope, evidence_paths=[prefix + p for p in paths])
    extra = ('selection_readback/audit.json', 'capacity_limit/audit.json', 'independent_input_audit/qualification.json',
        'independent_input_audit/lowdose_transfer/audit.json', 'independent_input_audit/lowdose_transfer/audit_protocol_v2.json',
        'independent_input_audit/lowdose_transfer/audit_failure_v1/failure.json',
        'independent_input_audit/lowdose_transfer/audit_protocol_setup_failure_v1/failure.json',
        'prediction_audit_failure_v1/audit.json', 'prediction_audit_failure_v1/diagnosis.json',
        'prediction_audit_resume_protocol_v2.json', 'pipeline_complete.json', 'delivery/documentation_audit.json',
        'physical_scope/audit.json', 'information_audit/audit.json')
    receipt = dict(created_utc=e.now(), notes_sha256=e.sha(dest), fixed_primary=PRIMARY,
        outer_method_reselected=False, facts=FACTS, objective_requirement_conclusions=requirements,
        next_theory='docs/ML_SANDBOX_PAIRED_TRANSPORT_NEXT_20261019.md',
        next_predictive_work='先核验两篇既有Sn来源的同样品化学—nH—mu配对；资格与输运适用性成立后，另冻联合物理学习及同标签强对照，继续以留出nH和完整性质链验收；不扩张本阶段份额参数或制造标签。',
        supplemental_json_sources={prefix + p: e.sha(OUT / p) for p in extra},
        source_code_sha256=e.sha(Path(__file__)), overall_goal_complete=False, goal_complete=False)
    e.write(OUT / 'analysis/conclusion_receipt.json', receipt)
    print('Actual evidence conclusions written', len(FACTS), 'CSVfacts;R6incomplete;goal remainsactive', flush=True)


if __name__ == '__main__': main()
