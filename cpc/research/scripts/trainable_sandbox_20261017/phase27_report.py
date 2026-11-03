"""Full48-method theory/results report; requires actual evidence interpretation."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase27_prediction as current
import phase26_report as previous
import phase2_energy as e

OUT = current.t.OUT
PRIMARY = 'FeasibleProcess_internal'
NEW = tuple(current.VARIANTS)
OLD = tuple(previous.VARIANTS)
VARIANTS = (PRIMARY,) + tuple(v for v in NEW if v != PRIMARY) + OLD
CHAIN = (PRIMARY, 'FeasibleProcess', 'EnvelopeProcess_coordinate', 'EnvelopeProcess',
    'FixedPhysical', 'ActivationProcess', 'MedianActivation', 'UnitDose', 'DirectRidge')
OVERVIEW = (PRIMARY,) + tuple(v for v in NEW if v != PRIMARY) + (
    'FixedPhysical', 'ActivationProcess', 'MedianActivation', 'UnitDose', 'DirectRidge', 'PhysicalAffine')
FAMILY = (PRIMARY, 'EnvelopeProcess_coordinate', 'FixedPhysical', 'ActivationProcess', 'MedianActivation', 'UnitDose')
AUDITS = ('numerical_gates_v2.json', 'readout_gates.json', 'learning_gates.json', 'source_gates.json',
    'saved_model_smoke/audit.json', 'source_smoke/audit.json', 'interface_smoke/audit.json',
    'training_audit.json', 'prediction/audit.json', 'delivery/cohort_replay/audit.json',
    'external_Azizie/audit.json', 'analysis/audit.json', 'physical_scope/audit.json',
    'delivery/final_checks/audit.json', 'information_audit/audit.json',
    'carbon_scope/audit.json', 'input_scope_annotations/audit.json',
    'delivery/documentation_audit.json', 'theory_scope_readback.json', 'theory/visual_review.json',
    'parent_verification_working.json')
THEORY = OUT / 'theory/ML与Sandbox_电荷可行共享学习理论.md'
CLAIMS = []


def number(path, selector, column):
    frame = read(OUT / path)
    for key, value in selector.items(): frame = frame[frame[key].eq(value)]
    assert len(frame) == 1, (path, selector)
    value = float(frame[column].iloc[0]); value = value if np.isfinite(value) else None
    CLAIMS.append(dict(path=str((OUT / path).relative_to(ROOT)), sha256=e.sha(OUT / path),
        selector=selector, column=column, value=value))
    return '未完整覆盖' if value is None else f'{value:.6f}'


def table(scheme='cluster', column='DOI_macro_nH_MAE', variants=VARIANTS, scope='common20'):
    rows = ['| 方法 | 0锚点 | 1锚点 | 2锚点 |', '| --- | --- | --- | --- |']
    for variant in variants:
        values = [number('analysis/all_method_metrics.csv', dict(
            scheme=scheme, variant=variant, budget=b, scope=scope), column) for b in (0, 1, 2)]
        rows.append('| ' + variant + ' | ' + ' | '.join(values) + ' |')
    return '\n'.join(rows)


def external(context, tier, variants=VARIANTS):
    budgets = (0, 1) if tier == 'rounded_same_Hall_2' else (0, 1, 2)
    rows = ['| 方法 | ' + ' | '.join(f'{b}锚点' for b in budgets) + ' |',
        '| ' + ' | '.join(['---'] * (len(budgets) + 1)) + ' |']
    for variant in variants:
        values = [number('external_Azizie/summary.csv', dict(context=context, input_tier=tier,
            variant=variant, budget=b, scope='common_query'), 'nH_MAE') for b in budgets]
        rows.append('| ' + variant + ' | ' + ' | '.join(values) + ' |')
    return '\n'.join(rows)


def figures():
    dest = OUT / 'figures'; dest.mkdir(exist_ok=True)
    d = read(OUT / 'analysis/all_method_metrics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.1), sharey=True)
    values = []
    for scheme in ('DOI', 'cluster'):
        a = np.full((len(OVERVIEW), 3), np.nan)
        for i, v in enumerate(OVERVIEW):
            for b in (0, 1, 2):
                q = d[d.scheme.eq(scheme) & d.scope.eq('common20') & d.variant.eq(v) & d.budget.eq(b)]
                assert len(q) == 1; a[i, b] = q.DOI_macro_nH_MAE.iloc[0]
        values.append(a)
    vmax = max(float(np.nanmax(a)) for a in values)
    cmap = plt.get_cmap('YlGnBu').copy(); cmap.set_bad('#d8d8d8')
    for ax, scheme, a in zip(axes, ['DOI exclusion', 'Related-group exclusion'], values):
        im = ax.imshow(a, vmin=0., vmax=vmax, cmap=cmap, aspect='auto')
        ax.set_title(scheme); ax.set_xticks(range(3), ['0', '1', '2']); ax.set_xlabel('Hall anchors per DOI')
        ax.set_yticks(range(len(OVERVIEW)), OVERVIEW, fontsize=7.3)
        for i in range(len(OVERVIEW)):
            for b in (0, 1, 2):
                ax.text(b, i, 'invalid' if not np.isfinite(a[i, b]) else f'{a[i,b]:.3f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if np.isfinite(a[i,b]) and a[i,b] > .55 * vmax else 'black')
    fig.subplots_adjust(left=.32, right=.9, bottom=.08, top=.95, wspace=.12)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.045, pad=.025)
    cb.set_label('Common20 DOI-macro MAE (dex)')
    fig.savefig(dest / 'fixed_overview.png', dpi=210, bbox_inches='tight'); plt.close(fig)
    fd = read(OUT / 'analysis/all_method_per_family.csv')
    labels = sorted(fd.family.unique()); fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, budget in zip(axes, (0, 2)):
        for variant in FAMILY:
            q = fd[fd.scheme.eq('cluster') & fd.scope.eq('common20') & fd.variant.eq(variant) & fd.budget.eq(budget)].set_index('family')
            ax.plot(range(6), [q.loc[f].nH_MAE for f in labels], marker='o', ms=3, lw=1, label=variant)
        ax.set_title(f'{budget} anchors per DOI'); ax.set_xticks(range(6), [f.split('/')[-1] for f in labels], rotation=38, ha='right', fontsize=7)
        ax.grid(alpha=.2); ax.set_ylabel('Common20 per-DOI MAE (dex)')
    axes[0].legend(fontsize=5.8); fig.tight_layout()
    fig.savefig(dest / 'fixed_family_diagnostics.png', dpi=210, bbox_inches='tight'); plt.close(fig)
    e.write(dest / 'provenance.json', dict(summary_sha256=e.sha(OUT / 'analysis/all_method_metrics.csv'),
        family_sha256=e.sha(OUT / 'analysis/all_method_per_family.csv'), overview_methods=OVERVIEW,
        family_methods=FAMILY, all48_in_tables=True, figure_names_fixed_before_outer=True, no_new_predictions=True))


def design():
    assert len(VARIANTS) == len(set(VARIANTS)) == 48
    assert not (OUT / 'prediction/execution_complete.json').exists(), 'Save report design before outer results'
    value = dict(created_utc=e.now(), primary=PRIMARY, method_order=VARIANTS, complete_main_tables=48,
        overview_methods=OVERVIEW, family_methods=FAMILY, property_chain_methods=CHAIN,
        all_missing_values_retained=True, actual_conclusions_required=True,
        source_code_sha256=e.sha(Path(__file__)), theory_path=str(THEORY.relative_to(ROOT)), theory_sha256=e.sha(THEORY),
        no_result_selection_or_prediction=True, goal_complete=False)
    assert not (OUT / 'report_design_before_results.json').exists()
    e.write(OUT / 'report_design_before_results.json', value)
    print('Declared full48 report design and fixed overview methods', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    if args.declare_only: design(); return
    spec = json.loads((OUT / 'report_design_before_results.json').read_text())
    assert spec['source_code_sha256'] == e.sha(Path(__file__)) and spec['theory_sha256'] == e.sha(THEORY)
    audits = {}
    for name in AUDITS:
        value = json.loads((OUT / name).read_text()); assert value['passed'], name; audits[name] = value
    notes = OUT / 'analysis/predictive_conclusions.md'; receipt = OUT / 'analysis/conclusion_receipt.json'
    conclusion = json.loads(receipt.read_text()); assert conclusion['facts'] and conclusion['notes_sha256'] == e.sha(notes)
    assert conclusion['fixed_primary'] == PRIMARY and not conclusion['outer_method_reselected']
    df = read(OUT / 'analysis/all_method_metrics.csv'); assert len(df) == 576 and set(df.variant) == set(VARIANTS)
    figures(); train = audits['training_audit.json']; analysis = audits['analysis/audit.json']
    replay = audits['delivery/cohort_replay/audit.json']; scope = audits['physical_scope/audit.json']
    info = audits['information_audit/audit.json']; transfer = json.loads((OUT / 'external_Azizie/execution_complete.json').read_text())
    exported = json.loads((OUT / 'delivery/export_complete.json').read_text())
    blocks = ['# ML与Sandbox：电荷可行共享学习与跨来源掺杂性质预测', '''## 1. 本轮主要问题与证据等级

主要假设是：由电荷方程生成输入相关的补偿阈值，再训练共享内部份额，是否能改善实际掺杂性质预测。主方法FeasibleProcess_internal在新外层结果前固定；同参数量、同端点信息的直接回归控制及旧36方法完整保留，共48变体。

仍用33条件、6 DOI、5保守关联组、原有0/1/2锚点角色和common20，旧29条件及common18不改。数据已经被以前研发查看，属于回顾性研究；嵌套训练/读取隔离不消除跨阶段方法选择偏差，不称为新的独立盲来源或实验室验证。

主要监督只有nH，无新mu标签、独立sigma标签、DFT/MD或实体测量。样品D已报道的表面C平均值未进入Si/Sn/H物理库存；它保留在数据和全部分数中，并逐点披露范围。''']
    body = '\n'.join(THEORY.read_text().splitlines()[1:]).replace('\n## ', '\n### ')
    blocks.append('## 2. 完整方法、数学推导与实现范围\n' + body)
    blocks.append(f'''## 3. 实际执行、失败记录与计算预算

实际训练{train['fit_records']}记录，{train['actual_nonlinear_fits']}次新非线性拟合，{train['actual_optimizer_starts']}次优化启动；初始未停止{train['initial_not_stopped']}次，最终选中未停止{train['selected_not_stopped']}条，训练域无效{train['training_invalid']}条。相同两起点/120上限及统一360细化规则用于六新族；停止不是全局最优证明，无效/未停止候选继续保留并在保存预测时拒绝。

新来源阶段{analysis['source_tasks']}任务、{analysis['actual_source_point_evaluations']}次点调用、{analysis['actual_source_vectors']}个向量；点失败{analysis['point_failure_records']}条，无效候选解{analysis['invalid_candidate_solutions']}条，未作有利筛除。新2376行与旧7128行合并{analysis['combined_point_rows']}行，数值{analysis['combined_numeric']}、观测域有效{analysis['combined_valid']}。

首轮raw logTf差分两失败、接口SHA阻止元数据改写及理论草稿单位排版问题均保留。数值门槛采用已验证的两宽度Richardson，未改模型公式/容差；接口另立V2，公开V3只补C范围提示。它们不是材料准确性改善。

actual optimizer starts和点调用分别报告；同调用上限不等于实际耗时完全相等。旧36方法沿用已冻结训练和来源选择，没有重复拟合再挑有利版本。''')
    blocks.append('## 4. 固定共同20查询：全部48方法\n\n单位dex，指log10绝对误差。仅完整覆盖时计算全队列MAE；未覆盖不是零。下表均为6 DOI等权，区别在于训练时按DOI还是整个关联组排除。\n\n### DOI排除\n\n' + table('DOI'))
    blocks.append('### 保守关联组排除\n\n' + table('cluster'))
    blocks.append('![全部新12变体与事先具名控制；完整48方法见上表，灰色为未覆盖。](figures/fixed_overview.png)')
    controls = ('FixedPhysical', 'ShareEffectiveProcess_internal', 'FeasibleProcess', 'EnvelopeProcess_coordinate',
        'EnvelopeProcess', 'ActivationProcess', 'MedianActivation', 'UnitDose', 'DirectRidge', 'RidgeAffine', 'PhysicalAffine')
    rows = ['| 主方法减去控制的5组宏平均MAE | 0锚点 | 1锚点 | 2锚点 |', '| --- | --- | --- | --- |']
    for control in controls:
        rows.append('| ' + control + ' | ' + ' | '.join(number('analysis/primary_matched_comparisons.csv',
            dict(scheme='cluster', budget=b, control=control), 'delta_group_macro') for b in (0, 1, 2)) + ' |')
    blocks.append('### 同一组排除预测：5组等权的配对差\n\n负值表示主方法误差较小。组内先平均DOI，再平均5组；完整48方法的该指标和全部282个主方法对照在CSV中。逐来源改善/持平/恶化、最差来源及去一组敏感性是小队列描述，不是显著性或多重比较校正。\n\n' + '\n'.join(rows))
    blocks.append('![六个预先具名方法的逐来源诊断；0与2锚点使用同一共同查询。](figures/fixed_family_diagnostics.png)')
    blocks.append('全部查询在0/1/2锚点时分别含33/26/20条件，其完整48方法结果在all_method_metrics.csv的all_query行。查询组成改变与固定common20上的锚点增量分开判断。共享选择和来源lambda只能使用相应训练内留出；全部候选非有限时的确定性回退不能称为实证胜出。')
    blocks.append('## 5. 完整性质链\n\n全队列9组mu配对/3 DOI，共同查询4组/3 DOI。下列9个方法在报告代码中预先具名；全部48方法均在完整指标CSV。无新mu训练或独立sigma真值，电导误差可与nH和mu误差相消，应联合解读。\n\n### 原生或经验mu误差\n\n' + table(column='native_mu_MAE', variants=CHAIN))
    blocks.append('### 统一CT(predicted nH)的mu误差\n\n使用预测nH，不是给定实测nH的条件迁移率任务。\n\n' + table(column='CT_mu_MAE', variants=CHAIN))
    blocks.append('### 原生派生sigma误差\n\n' + table(column='native_derived_sigma_MAE', variants=CHAIN))
    context = 'cluster::without:related_AFRL_Cornell_Wright'
    blocks.append('## 6. 到既有Azizie条件的冻结转移\n\n共享权重和来源超参数固定，仅用允许的Hall锚点适应。A为e/k两条印刷约数化学输入，B为12个过程化学校准代理；重叠后仍为12个Hall条件，不是14个独立样品，也不是纯化学输入消融。O3采用已声明的有效氧情景，H未知零，300K为比较约定。\n\n### 删除整个关联组：B任务共同10查询，全部48方法\n\n' + external(context, 'learned_process_proxy_12'))
    blocks.append('### 删除整个关联组：A任务共同高剂量k查询\n\nUnitDose等于印刷约数不证明真实100%激活。\n\n' + external(context, 'rounded_same_Hall_2'))
    for context in ('DOI::full', 'cluster::full'):
        blocks.append('### ' + context + '：B任务固定9方法诊断\n\n完整48方法均在external_Azizie/summary.csv，不能只据full上下文得出跨来源结论。\n\n' + external(context, 'learned_process_proxy_12', CHAIN))
    blocks.append(f"实际转移{transfer['group_requests']}请求/{transfer['rows']}点，数值{transfer['numeric']}、有效{transfer['valid']}；全部拒绝和不利结果保留。a/j相同输入而不同Hall标签的确定性下界见identical_input_bound.csv，不能唯一解释其他误差。")
    blocks.append(f'''## 7. 保存复用、物理适用域和信息成本

实际{exported['presets']}预设/{exported['models']}参数文件，其中{exported['unqualified_presets']}预设未满足资格，保存并拒绝。主V2公共接口独立进程回放{replay['requests']}请求/{replay['point_rows']}点，比较{replay['valid_property_vectors_compared']}个有效三性质向量及{replay['unaccepted_numeric_continuations_compared']}个未接受数值延续，最大对数差{replay['maximum_log_property_difference']:.3e}。这证明复用一致，不是新材料验证。

三个新物理族在两个完整上下文的198槽位中，{scope['numeric_states']}点数值完成，{scope['unqualified_model_points_not_evaluated']}点因模型资格拒绝。多数电流权重位于k乘平均自由程小于1的状态{scope['majority_current_weight_kl_lt1']}个，有效Tf高于名义生长温度{scope['effective_Tf_above_growth']}个；均不删分数，也不把代码域或位点包络当成Born/热历史证书。Envelope不声称中间物理库存。

信息预算{info['budget_rows']}行，{info['physical_energy_methods']}方法使用缺陷能量先验，{info['direct_or_empirical_no_defect_energy_methods']}方法不使用。原始能量条目按Si无H288、Si加H621、Sn无H287计，每类另两个派生中性行不是新DFT；不是最小必需能量、独立标签或DFT任务数。Envelope也使用能量端点，不能伪称与纯直接模型先验相同。

全部9504主行有输入范围注释；其中288行对应同一个已报道C样品的不同模型/划分/预算，不是288个独立C测量。物理层未含该表面富集C，DirectRidge可能将其作为表格特征；原样品和角色保留。V3公开包装器只补提示，8快照数值不变由独立46项范围检查支持，不冒称另一次V3全队列回放。

以前目标导向能量选点采用遮蔽已有能量及虚拟Sandbox性质回放，没有新DFT或材料标签；它的证据等级不能被当前训练升级为真实材料计算闭环。''')
    blocks.append('## 8. 根据完整实际证据形成的判断\n\n' + notes.read_text())
    blocks.append('## 9. 复现与剩余研究范围\n\n训练→训练审计→冻结预测→预测审计→导出→独立API回放→冻结转移→转移审计→全方法分析/物理范围/CLI/信息预算；范围注释、实际结论及报告在结果后完成。所有协议和SHA先核对，CSV用csv_io_v2.read_result_csv读取，不覆盖失败或因观察超时重复启动任务。\n\n公开使用说明见delivery/README.md；默认predict_feasible_properties_v2.py带已知C提示。模型包未复制第三方物理数据库，不能缺少参考时静默换库。局部梯度不含CV或来源重拟合导数；资格/数值通过不提供已校准材料置信区间。\n\n当前只有回顾性Si/Sn任务，稀土、微观速率、寿命和PDR不在已验证范围。论文首创性也不由数学一致性或工具整合自动获得。系统goal继续以真实掺杂性质预测及可复现使用为终点；整体完成与下一研究选择须另外按全部要求和实际证据审计。')
    report = '\n\n'.join(blocks) + '\n'; dest = OUT / 'ML与Sandbox_电荷可行学习与性质预测.md'
    assert not dest.exists(), 'Keep the first completed results report before revision'
    assert not [ord(c) for c in report if ord(c) < 32 and c != '\n']
    dest.write_text(report)
    jsonpaths = (*AUDITS, 'delivery/export_complete.json', 'external_Azizie/execution_complete.json',
        'analysis/conclusion_receipt.json', 'report_design_before_results.json', 'prediction/shared_choices.json')
    e.write(OUT / 'report_claims.json', dict(report_sha256=e.sha(dest), claims=CLAIMS, primary=PRIMARY,
        theory_path=str(THEORY.relative_to(ROOT)), theory_sha256=e.sha(THEORY), source_code_sha256=e.sha(Path(__file__)),
        json_audit_sources={str((OUT / p).relative_to(ROOT)): e.sha(OUT / p) for p in jsonpaths},
        all48_methods_reported=True, outer_method_reselected=False, no_new_fits_or_predictions=True, goal_complete=False))
    print('Actual full48-method report written', len(CLAIMS), 'traceableCSVvalues;goal completion not implied', flush=True)


if __name__ == '__main__': main()
