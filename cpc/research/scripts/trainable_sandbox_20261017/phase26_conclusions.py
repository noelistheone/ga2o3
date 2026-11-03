"""Record actual phase26 predictive decisions with exact, checkable evidence."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

OUT = e.BASE / 'phase26_pool_share'
FACTS = []


def fact(path, selector, column):
    frame = read(OUT / path)
    for key, value in selector.items():
        frame = frame[frame[key].eq(value)]
    assert len(frame) == 1, (path, selector)
    x = float(frame[column].iloc[0])
    FACTS.append(dict(path=str((OUT / path).relative_to(ROOT)), sha256=e.sha(OUT / path),
                      selector=selector, column=column, value=x if np.isfinite(x) else None))
    return f'{x:.6f}' if np.isfinite(x) else '未完整覆盖'


def metrics(variant, scheme='cluster', column='DOI_macro_nH_MAE'):
    return ' / '.join(fact('analysis/all_method_metrics.csv',
        dict(scheme=scheme, variant=variant, budget=b, scope='common20'), column) for b in (0, 1, 2))


def transfer(variant, tier='learned_process_proxy_12', budgets=(0, 1, 2)):
    return ' / '.join(fact('external_Azizie/summary.csv', dict(
        context='cluster::without:related_AFRL_Cornell_Wright', variant=variant,
        input_tier=tier, budget=b, scope='common_query'), 'nH_MAE') for b in budgets)


def main():
    dest = OUT / 'analysis/predictive_conclusions.md'
    assert not dest.exists(), 'Keep the first evidence interpretation; revisions need a new file'
    diagnosis = json.loads((OUT / 'inner_domain_diagnosis/audit.json').read_text())
    assert diagnosis['passed'] and diagnosis['cases'] == 193 and diagnosis['completed'] == 193
    primary = 'ShareEffectiveProcess_internal'
    text = f'''### 9.1 主要结论：当前新共享关系没有稳定预测增量

以下三项顺序均为0、1、2个锚点；误差单位dex。保守关联组排除、固定common20、6 DOI等权：主方法为{metrics(primary)}；固定物理为{metrics('FixedPhysical')}；中位数激活校准为{metrics('MedianActivation')}；UnitDose为{metrics('UnitDose')}。零锚点物理参考比部分直接经验控制误差低，但新的共享关系在这一方案没有改变它；1、2锚点内部适应也未超过这些简单控制。

同一预测按5关联组等权，主方法为{metrics(primary, column='group_macro_nH_MAE')}，固定物理为{metrics('FixedPhysical', column='group_macro_nH_MAE')}。不能把DOI和关联组宏平均互换，或只选一种权重宣称优势。DOI单独排除时主方法为{metrics(primary, scheme='DOI')}，同样没有稳定胜出。

主方法在外层分数产生前固定，不改选表现较好的单个变体作为本轮成功。ActivationProcess对照在组排除下为{metrics('ActivationProcess')}：其1锚点较好、2锚点又不如简单中位数，不能据一个预算创造另一项物理创新。完整36方法、两种排除、全部预算及未覆盖行均在报告和CSV中。

### 9.2 回退原因：验证状态越界与有限误差选择应分开

主共享工艺族5个组排除外层全部选择显式参考；其中4个外层的所有有限gamma候选都没有合格的完整内层分数，只有参考有限。余下1个外层中，有限gamma=0有分数但仍不如参考。DOI方案6个外层中3个只有参考有限，1个在有限候选之间参考较好，2个保留有限gamma。两个full上下文也只剩参考有限。因此不能把回退概括为所有工艺特征无信息或所有模型强度都被普通正则化收缩掉。

这些统计来自selection_finiteness.json，null分数对应无穷/不合格验证，不能当成零误差。GrowthReference的13个共享上下文有12个全候选不合格；三个ShareGrowth族相应为5、8、11个。其None选择可能只是全部不合格时的确定性选择规则，不构成参考模型的实证胜出。保存接口继续按训练资格拒绝这些模型，主表不计算部分覆盖MAE。

9个外层参考回退上下文及2个full上下文，153个对应预测与旧InventoryProcess_internal逐点核对，最大差约2.90e-13 dex，来源政策也一致。这些输出是已知参考的精确复现，不是新ML贡献。

### 9.3 新的失败定位：绝对份额越过电荷补偿边界

在预先固定的193个失效内层状态上，独立复算全部成功，最大保存回放差约6.60e-12 dex。193个自由电子浓度都低于既定下限，其中190个不满足电子占优。相同学习温度/背景仅恢复参考库存后193个全部有效，相同训练单元的继承参考也全部有效。它们不是新的物理实验、独立样本或外层准确性；也不能把数学上的电子不占优延续解释成真实p型材料预测。

对每个状态，在固定目标费米能级处独立构造电荷方程，验证Q随份额f仿射；在正电荷对比及冷根严格递减的条件下，fcrit将自由电子低于/高于下限精确分开。193项均满足条件，均与原求解状态一致。fcrit在该诊断中的范围约0.364至0.712；它不是通用掺杂激活下限，而取决于实际参考库存、背景和观测约定。自由电子下限不等于Hall下限，仍须完整输运观测门槛。

这项定位支持下一预测改进的机制假设，却没有修复本轮任何预测，也不唯一排除其他工艺/能量/缺陷目录失配。所有旧结果和主方法保持原样。

### 9.4 性质链、转移与适用域没有被额外包装

主方法在共同4组mu配对/3 DOI上的原生mu误差为{metrics(primary, column='native_mu_MAE')}，原生派生sigma误差为{metrics(primary, column='native_derived_sigma_MAE')}。同一模型输出三个数不等于三次独立材料验证；没有新mu监督，没有独立sigma标签。低维nH学习未证明相应真实散射库存被识别。

删除整个AFRL/Cornell/Wright关联组后，Azizie过程代理B任务固定10查询主方法为{transfer(primary)}，而中位数为{transfer('MedianActivation')}，ActivationProcess为{transfer('ActivationProcess')}，UnitDose为{transfer('UnitDose')}。主共享模型在该上下文回退，允许的来源适应也不改变主误差。新方法没有改善这里的跨条件转移。

A任务共同高剂量k点主方法0、1锚点为{transfer(primary, 'rounded_same_Hall_2', (0, 1))}；UnitDose零锚点恰好等于印刷约数，不是100%激活的真实测量确认。A/B合计仍是12个Hall条件，不是14个独立样本；化学输入层级、O3情景和缺失H等约定保留。它是已知数据的冻结条件转移，不是盲测或独立实验室验证。

全来源物理检查264个合格模型状态数值完成，另198个未达模型资格的槽位拒绝；55个状态多数电流权重位于k乘平均自由程小于1，198个有效温度高于名义生长温度。它们都保留，既不删分数，也不证明Born近似或热历史正确。复杂库存给出的迁移率仍受模型有效性限制。

### 9.5 下一步仍做预测：用物理可行份额训练一个共享关系

下一项只检验一个具体改动：先由输入和当前电荷模型计算补偿阈值，再学习阈值以上可用孤立份额，不让跨输入线性logit轻易落入无效补偿分支。推导、每个符号、梯度链和反例条件见docs/ML_SANDBOX_CHARGE_FEASIBLE_NEXT_20261018.md。当前尚未实现/训练，不将解析阈值或193项回放当成精度增量。

先通过固定状态、原子守恒、阈值与全链导数门槛，再在新冻结版本中用同数据/分组/锚点/预算重训。强直接激活、相同可达包络的直接读出、固定物理、常数与输出校准都要保留，区分约束减少越界、函数类/先验变化和实际物理预测增量。不得放宽Hall门槛、使用查询标签调边界、事后改主模型或接入未验证能量。

本轮没有稳定新共享增量，因此预声明的rho机制匹配归因没有启动。若下轮仍无增益，就停止此处参数化扩张，查找独立配对信息与具体物理参考误差，不无限扩大网络。本轮交付可复现训练、预测、保存接口及完整成败证据；整体goal仍active，新稀土、寿命、PDR及微观动力学预测尚未验证。
'''
    dest.write_text(text)
    supplemental = ['inner_domain_diagnosis/audit.json',
        'inner_domain_diagnosis/protocol_before_replay.json',
        'inner_domain_diagnosis/selection_finiteness.json', 'fallback_equivalence/audit.json']
    e.write(OUT / 'analysis/conclusion_receipt.json', dict(created_utc=e.now(),
        notes_sha256=e.sha(dest), facts=FACTS,
        supplemental_json_sources={str((OUT / p).relative_to(ROOT)): e.sha(OUT / p) for p in supplemental},
        fixed_primary=primary, outer_method_reselected=False, favorable_gain_presumed=False,
        next_theory='docs/ML_SANDBOX_CHARGE_FEASIBLE_NEXT_20261018.md',
        next_predictive_work='Implement and gate a charge-feasible shared excess-share map, then use a new matched prediction protocol; do not alter phase26 or promise accuracy gain.',
        goal_complete=False, code_sha256=e.sha(Path(__file__))))
    print('Saved actual phase26 conclusions', len(FACTS), 'CSV facts; primary unchanged, goal active', flush=True)


if __name__ == '__main__':
    main()
