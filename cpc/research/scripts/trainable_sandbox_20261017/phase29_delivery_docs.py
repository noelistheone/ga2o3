"""Document actual species/label support and the usable saved prediction API."""
from pathlib import Path
import sys,json

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase29_data as data
import phase2_energy as e

OUT=data.OUT/'delivery'


def main():
    assert not (OUT/'documentation_audit.json').exists(),'preserve previous guide before versioned repair'
    replay=json.loads((data.OUT/'api_replay/audit.json').read_text())
    assert replay['passed'] and replay['requests']==1056
    bundle=OUT/'predictor_bundle';manifest=json.loads((bundle/'manifest.json').read_text())
    conditions=data.load_conditions();cases=data.load_cases();roles=data.load_roles()
    contexts={}
    for spec in manifest['presets'].values():contexts[spec['context']]=spec['training_units']
    support=[]
    for context,units in contexts.items():
        raw=conditions[conditions.related_cluster.isin(units)]
        expanded=cases[cases.related_cluster.isin(units)]
        for element in ('Si','Sn'):
            part=raw[raw.element.eq(element)];scenarios=expanded[expanded.element.eq(element)]
            paired=part[part.mobility_eligible.eq(True)]
            support.append(dict(context=context,element=element,reported_carrier_conditions=len(part),
                carrier_DOIs=part.family.nunique(),carrier_related_groups=part.related_cluster.nunique(),
                reported_mu_pairs=len(paired),paired_DOIs=paired.family.nunique(),
                min_training_chemical_cm3=float(part.incorporated_cm3.min()),
                max_training_chemical_cm3=float(part.incorporated_cm3.max()),
                min_declared_growth_scenario_C=float(scenarios.growth_T_C.min()),
                max_declared_growth_scenario_C=float(scenarios.growth_T_C.max()),
                oxygen_scenarios=sorted(scenarios.scenario_pO2_atm.unique()),
                individual_Sn_new_process_inputs_incomplete=bool(part.family.eq(data.GOGOVA).any()),
                numerical_bounds_are_not_accuracy_bounds=True,independent_sigma_labels=0))
    counts=[]
    for b in (0,1,2):
        part=roles[roles.budget.eq(b)]
        counts.append(dict(budget=b,reported_anchor_conditions=int(part.role.eq('anchor').sum()),
            guard_conditions=int(part.role.eq('guard').sum()),query_conditions=int(part.role.eq('query').sum())))
    e.csvwrite(OUT/'validation_support.csv',support);e.csvwrite(OUT/'role_counts.csv',counts)
    qualified=sum(s['serving_qualified'] for s in manifest['presets'].values())
    text=f"""# 第29阶段：可训练ML＋Sandbox研究预测接口

已导出{len(manifest['presets'])}个预设，其中{qualified}个拟合通过当前服务资格检查，覆盖48个固定方法/消融变体、5个关联组排除上下文及1个全来源上下文。不合格拟合及原始失败保留，对应预设明确拒绝。

接口从实际掺入量和显式工艺输入预测Hall载流子、Hall迁移率及相容电导率。共享权重固定，来源适应仅用0/1/2个允许Hall标签，无查询迁移率输入或调用时重训练。

## 数据和验证范围

35个报告条件、7 DOI、5关联组中有14Si和21Sn条件；11组化学—Hall—迁移率配对为9Si与2Sn。这不是35个已核实互不复用的晶片，电导也不是第三个独立标签。

新增Gogova两条件与旧Baldini按同IKZ组排除。其逐样品工艺及定量C/H不完整，每条件8个固定情景的权重总和为1。49情景行不等于49样品，情景范围不是校准置信区间。Je混合Si/Fe及不可测的Gogova C未被伪造成单掺杂/零标签。

主验证使用固定common20载流子查询。新Sn在0/1锚点有2/1个查询条件，2锚点无nH查询；锚点迁移率单列条件任务。研究属于回顾性研发，没有独立盲测或新元素验证。逐上下文/元素的训练支持见validation_support.csv，角色见role_counts.csv。

## 上下文与调用

- cluster::full::JointProcess_internal：全部训练来源拟合的主研究模型，用于后续条件研究，不是本队列的留出验证。
- cluster::without:关联组::变体名：该组全部来源标签同时从全局训练排除，用于相应来源回放。
- 全部48个变体保留；主方法身份在查看外层结果前固定。固定物理、只学散射、常数/剂量/工艺库存、nH-only、直接ML、常数输运校准和强仿射各有独立身份。

在Ga2O3-Sandbox目录、ga2o3环境中设置PYTHONPATH=src。以下仅为输入情景示例，不包含任何声称的测量或预测结果：

```python
from sandbox.joint_property_predictor import JointPropertyPredictor

predictor = JointPropertyPredictor(
    "results/trainable_sandbox_20261017/phase29_joint_scattering/"
    "delivery/predictor_bundle"
)
conditions = [{{
    "condition_id": "trial",
    "case_id": "trial::scenario0",
    "element": "Sn",
    "incorporated_cm3": 1.0e19,
    "Tobs_K": 300.0,
    "H_known": False, "H_cm3": 0.0,
    "carbon_known": False, "carbon_cm3": 0.0,
    "growth_T_C": 800.0,
    "scenario_pO2_atm": 1.0e-5,
    "oxygen_input_kind": "explicit diagnostic scenario;not measured oxygen",
}}]
result = predictor.predict(
    preset="cluster::full::JointProcess_internal",
    conditions=conditions,
    anchors=[],
)
print(result["rows"])
```

incorporated_cm3须是实际化学掺入量或明确情景，不能自动用名义靶材分数替代。H/C的False+0表示unknown-zero情景，未证明无H/C。物理库存未包含碳缺陷；直接ML的碳特征也不是碳缺陷物理。

growth_T_C为工艺输入，内部有效冻结温度是另一个量。scenario_pO2_atm为正的明确情景，水蒸气、CO2和总压力不能未经转换当作氧分压。当前Tobs_K固定300K，部分文献RT使用300K比较约定。

## 锚点和有限情景设计

anchors只接受condition_id和observed_nH_cm3；0/1/2按不同报告条件计数，不将8情景当8锚点。输入ID由调用者提供，接口没有核实独立晶片身份。锚点的全部情景必须包含在conditions中；正预算要求每条件情景数相同，以复现冻结协议的等权目标。一次请求对应一个来源适应组。

响应正则定义在给定有限输入集合上，可使用未标注的查询输入。改变集合可能改变适应解；它不是与查询集合无关的唯一微观来源参数。未知输入不能按查询误差挑选。

直接双头模型更新Hall头时固定rho头；物理输出偏移保留状态迁移率，经验CT随新Hall浓度重算；内部库存适应改变同一状态的性质。manifest记录各预设的更新规则。

## 输出和证据

每情景返回nH_cm3、muH_cm2Vs、sigma_Scm及对数，满足sigma=e*nH*muH。两项误差反向抵消可能降低电导误差，不能据此单独宣称完整性质链成功。

numeric表示数值可输出，valid还检查相应模型的观测域与整个来源设计；两者不构成Born/RTA、均匀体或热历史认证。资格/输入/适应失败保留拒绝或无效状态。condition_summaries的对数均值及范围是固定情景摘要，不是置信区间。

模型JSON没有实验/训练标签表，bundle不复制第三方KROGER原始输入；manifest绑定实现、数值FD参考和物理输入哈希。1056次独立进程回放覆盖来源排除预测及全来源已拟合输入；回放是实现证据，不是新材料验证。物理与信息范围分别见physical_scope和information_audit。
"""
    (OUT/'README.md').write_text(text)
    e.write(OUT/'documentation_audit.json',dict(passed=True,created_utc=e.now(),presets=len(manifest['presets']),
        serving_qualified=qualified,reported_conditions=35,scenario_rows=49,paired_Si=9,paired_Sn=2,
        guide_sha256=e.sha(OUT/'README.md'),support_sha256=e.sha(OUT/'validation_support.csv'),
        role_counts=counts,no_model_score_or_query_changes=True,goal_complete=False))
    print('Delivery29 guide and actual input/label support written',flush=True)


if __name__=='__main__':main()
