"""Source-bound scientific judgment of the complete prediction deliverable.

Statuses describe whether required research was performed, never whether a
desired favorable accuracy or uniquely identified material mechanism exists.
"""
from pathlib import Path
import sys,json

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e
from csv_io_v2 import read_result_csv as read

OUT=e.BASE/'phase29_joint_scattering'
PREFIX=str(OUT.relative_to(ROOT))


def evidence(*paths):
    return [PREFIX+'/'+p for p in paths]


def requirement(reason,scope,*paths,status='complete_with_conditional_limits'):
    return dict(status=status,reason=reason,scope=scope,evidence_paths=list(paths))


def main():
    target=OUT/'scientific_interpretation_v1.json'
    assert not target.exists(),'Preserve the original scientific review'
    primary='JointProcess_internal'
    summary=read(OUT/'prediction/summary.csv')
    common=summary[summary.scope.eq('common20')]
    def metrics(variant):
        part=common[common.variant.eq(variant)].sort_values('budget')
        assert len(part)==3 and list(part.budget)==[0,1,2]
        return part[['budget','nH_MAE','mu_MAE','sigma_MAE',
            'conditions','paired_conditions']].to_dict('records')
    result={name:metrics(name) for name in (primary,'ShareProcess205_internal',
        'Reference205_internal','ReferenceTau_internal','JointConstant_internal',
        'MedianActivation_offset','UnitDose_offset','DirectMultiRidge_offset',
        'JointProcess_nHonly_internal')}
    a=result[primary];b=result['ShareProcess205_internal']
    assert all(x['nH_MAE']>y['nH_MAE'] and x['mu_MAE']>y['mu_MAE'] for x,y in zip(a,b))
    requirements={
        'R1':requirement(
            '冷热求解、库存守恒、冷根隐式导数及同状态输运已实现；训练前门槛与真实保存拟合点的完整损失梯度均已独立核验。',
            '平滑有效分支和声明的Si/Sn/H、均匀体、单带RTA约定；正率、根唯一性和数值一致性不证明微观机制真实。',
            *evidence('numerical_gates_v2/gates.json','fitted_gradient_audit_v1/audit.json',
                'scatter_budget_notation_audit_v1/audit.json'),
            'docs/ML_SANDBOX_METHOD_PHASE29_V2_20261019.md'),
        'R2':requirement(
            '跨条件共享库存份额关系及正散射参数已实际训练、保存和重用，不是逐样品自由偏移；固定散射、常数份额及nH-only消融均已评价。',
            '实现和训练要求完成；现有留出队列不支持联合过程模型稳定优于强对照或已发现真实制备路径。',
            *evidence('training_complete.json','training_audit.json','training_protocol.json',
                'prediction/summary.csv'),
            'src/sandbox/joint_inventory_scattering.py'),
        'R3':requirement(
            '完成实际化学掺入量及声明工艺输入到载流子响应的关联组排除、逐来源预测；主方法和共同查询在外层评分前固定。',
            '35报告条件/7DOI/5关联组，主common20；部分工艺和H/C未知，49行是条件内情景而非49独立样品；回顾性研发而非盲测。',
            *evidence('data/qualification.json','data/conditions35.csv',
                'data/scenarios49.csv','prediction/predictions.csv',
                'prediction/per_family.csv','prediction_design_protocol.json')),
        'R4':requirement(
            '0/1/2个物理条件的Hall锚点及guard角色保持；共享选参只用训练来源内部，来源适应仅用该任务允许的Hall锚点。',
            '新增Sn在2锚点没有查询，不补造第三个数值；查询迁移率不用于来源适应；完整训练输入回放不是留出验证。',
            *evidence('data/episodes35.csv','prediction/plan.json',
                'prediction/source_choices.json','prediction/audit.json')),
        'R5':requirement(
            '固定物理、全局/来源常数、同容量常数/剂量/工艺库存、经验激活、仿射、直接多输出ML和训练集输运校准共48变体完整比较。',
            '标签、输入情景、锚点、覆盖集相同；非线性候选统一初始化/迭代规则，实际预算全部记录。闭式Ridge与非线性求解不伪称耗时或物理先验信息量完全相同。',
            *evidence('training_protocol.json','prediction_design_protocol.json',
                'training_audit.json','prediction/execution_complete.json',
                'prediction/summary.csv')),
        'R6':requirement(
            '已从预测状态输出并实际留出评价迁移率和相容电导，扩充为9个Si及2个Sn配对；保留大误差、分量抵消及全部失败，不以实测nH条件回归替代完整链。',
            '本项完成的是性质链实现和有条件验证，不是低误差或跨元素可靠性认证。共同配对仅4Si/3DOI；新增2Sn工艺不全且只有0/1锚点查询；sigma由两个观测方向派生。',
            *evidence('report_v1/study_statistics.json','prediction/summary.csv',
                'prediction/predictions.csv','physical_scope/audit.json',
                'transport_attribution_v1/audit.json')),
        'R7':requirement(
            '带电能量标签、参考、Hall量/自由电子、组成、单位、同样品配对和新增来源资格均有审计；补全背景散射记号并保留原始文档。',
            '未产生新DFT、未实测缺失过程参数、未建立Born/实际散射机制认证。额外3条Sn线索0入队列，不把归一化SIMS或拟合ND当独立化学浓度。',
            'results/trainable_sandbox_20261017/phase2_energy/label_algebra_audit.json',
            *evidence('data/qualification.json','additional_source_audit_v1/qualification.json',
                'scatter_budget_notation_audit_v1/audit.json','physical_scope/audit.json'),
            'docs/ML_SANDBOX_SCATTER_BUDGET_CORRECTION_PHASE29_20261019.md'),
        'R8':requirement(
            '已完成由目标响应灵敏度驱动的计算/标签选择，与能量方差、随机选择的同揭示预算对照和数值核验。',
            '为已发表带电能量的回顾性揭示与虚拟Sandbox性质目标，不是实际新增DFT或新材料闭环；局部高斯/线性理论与能量ML适用性有条件。',
            'results/trainable_sandbox_20261017/phase2_energy/acquisition/protocol.json',
            'results/trainable_sandbox_20261017/phase2_energy/acquisition/summary.csv',
            'results/trainable_sandbox_20261017/phase2_energy/acquisition/numerical_gates.json',
            'results/trainable_sandbox_20261017/phase2_energy/acquisition/选点公式与成立条件.md'),
        'R9':requirement(
            '针对库存坐标、来源适应、过程输入、物理可行性和输运关系多次作版本化改进；最终保留联合训练退化、非停止候选、未覆盖及输入不足。固定状态散射干预定位明确的模型内误差贡献。',
            '后验干预是按给定顺序的数值归因，不是新的验证模型或唯一微观因果。不能把所有失配归于某一种未测杂质或材料结构。',
            *evidence('nonstopped_candidate_observation_v1.json',
                'numerical_gates_v1/gates.json','transport_attribution_v1/audit.json',
                'transport_attribution_v1/summary.csv','information_audit/audit.json')),
        'R10':requirement(
            '保存模型与288研究预设可由无查询表读取的统一接口加载；1056请求/9408点实际独立进程回放及逐值审计完成。',
            '原始第三方KROGER数值表不复制进交付；需已有哈希匹配物理参考。全数据预设不是外推精度证据；拒绝状态不伪造成数值。',
            *evidence('delivery/audit.json','delivery/README.md','api_replay/audit.json',
                'delivery/validation_support.csv'),
            'src/sandbox/joint_property_predictor.py'),
        'R11':requirement(
            '交付完整方法/符号/守恒/推导、全48结果、保存接口、数值回查与PDF；17页主报告和3页诊断实际逐页检查，排版V2仅修公式标题。',
            '理论说明和审计作为真实预测研究的附件，不能单独替代预测贡献；下一项具体、可否定的改进另列，不声称保证发表或精度。',
            *evidence('report_v1/independent_audit.json','visual_review_v1.json',
                'report_layout_v2/audit.json','report_layout_v2/visual_review_v2.json',
                'transport_attribution_v1/visual_review_v1.json'),
            'docs/ML_SANDBOX_METHOD_PHASE29_V2_20261019.md'),
    }
    for item in requirements.values():
        for name in item['evidence_paths']:assert (ROOT/name).is_file(),name
    review=dict(created_utc=e.now(),fixed_primary=primary,outer_method_reselected=False,
        all_adverse_results_retained=True,retrospective_development=True,
        overall_goal_status='active',requirements=requirements,
        statuses_describe_research_deliverable_not_favorable_accuracy=True,
        implementation_complete=True,stable_new_joint_predictive_increment_established=False,
        reliable_cross_element_full_property_prediction_established=False,
        primary_prediction_conclusion=(
            '主JointProcess_internal在固定共同查询的三个锚点预算均未优于固定205的同结构ShareProcess205，'
            '也没有统一优于固定参考、经验少样本适应及直接ML。nH-only联合变体零锚点有较低回顾性误差，'
            '但不恢复迁移率，不能事后替换主方法或宣称全链统一优势。不同预算支持不同强对照，没有单一已验证总冠军。'),
        full_chain_conclusion=(
            '完整链已真实计算并验证，但主要方法迁移率误差大。固定状态恢复205可大幅减小mu误差而略增nH误差，'
            '定位了学习散射通道中的泛化风险；残余差额和数据结构不支持唯一材料成因。'
            '新增Sn只支持固定情景内0/1锚点的有条件结果，不证明真实热历史、稀土排序或九性质器件预测。'),
        source_bound_common20_metrics=result,
        next_testable_predictive_work=(
            '优先检验模型选择目标与完整性质链是否错配：保持原外层关联组、输入情景和Hall锚点预算，'
            '在全新冻结协议中只用训练来源内部的Hall系数与电阻率两观测方向，'
            '对已保存候选作Pareto/基线安全门控选择；主nH损失最小化须同时满足训练内配对电阻率不劣于固定散射参考，'
            '无合格候选则回退并如实计数。μ及sigma仍用同一物理状态，不读查询μ，不把派生sigma作第三标签。'
            '与同标签、同适应预算的固定参考、共享常数、经验和直接ML比较；报告覆盖、回退、逐源与完整误差。'
            '此为待执行的可否定设计，不是本轮结果。进一步的配对Hall锚点需独立增加信息预算并给所有对照。'
            '真正跨材料验证需要同样品独立化学掺入量、完整温度/氧气协议及可靠输运配对；没有则输出输入条件与失配，'
            '不以扩大网络或把未知输入当可调真值代替。'),
        next_experiment_already_executed=False,new_blind_validation=False,
        rareearth_validation_established=False,new_Sn_process_inputs_incomplete=True,
        sigma_is_derived_not_third_label=True,material_Born_validity_established=False,
        auxiliary_theory_is_not_replacement_for_prediction=True,
        goal_complete=False,system_goal_status_not_changed=True)
    review['evidence_sha256']={name:e.sha(ROOT/name) for item in requirements.values()
        for name in item['evidence_paths']}
    review['evidence_sha256'][str((OUT/'prediction/summary.csv').relative_to(ROOT))]=e.sha(OUT/'prediction/summary.csv')
    review['code_sha256']=e.sha(Path(__file__))
    e.write(target,review)
    print('Scientific review: all eleven research requirements evidenced; no stable joint gain claimed',flush=True)


if __name__=='__main__':main()
