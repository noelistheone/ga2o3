"""Traceable theory and full predictive results; no new fits or outer selection."""
from pathlib import Path
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

OUT = e.BASE / 'phase25_shared_process'
CLAIMS = []
CACHE = {}
VARIANTS = ('FixedPhysical', 'InventoryConstant', 'InventoryDose', 'InventoryProcess',
    'InventoryConstant_internal', 'InventoryDose_internal', 'InventoryProcess_internal',
    'OutputConstant', 'OutputDose', 'OutputProcess', 'DirectDose', 'DirectProcess',
    'DirectRidge', 'UnitDose', 'MedianActivation', 'DoseLine', 'DoseLineAffine', 'RidgeAffine', 'PhysicalAffine')


def value(path, selector, column, fmt='.6f'):
    if path not in CACHE:
        CACHE[path] = read(path)
    d = CACHE[path]
    for k, v in selector.items():
        d = d[d[k].eq(v)]
    assert len(d) == 1, (path, selector, len(d))
    x = float(d[column].iloc[0])
    CLAIMS.append(dict(path=str(path.relative_to(ROOT)), sha256=e.sha(path), selector=selector,
                       column=column, value=x if np.isfinite(x) else None))
    return format(x, fmt) if np.isfinite(x) else '未完整覆盖'


def summary_table(scheme, scope='common20', column='nH_MAE'):
    rows = ['| 方法 | 0 Hall锚点 | 1 Hall锚点 | 2 Hall锚点 |', '| --- | --- | --- | --- |']
    for v in VARIANTS:
        rows.append('| ' + v + ' | ' + ' | '.join(value(OUT/'prediction/summary.csv',
            dict(scheme=scheme, variant=v, scope=scope, budget=b), column) for b in (0, 1, 2)) + ' |')
    return '\n'.join(rows)


def figures():
    s = read(OUT/'prediction/summary.csv')
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 8), sharey=True)
    finite = s.loc[s.scope.eq('common20'), 'nH_MAE'].dropna()
    vmax = max(float(finite.max()), .05)
    for ax, scheme in zip(axes, ('DOI', 'cluster')):
        z = np.array([[s[(s.scope.eq('common20')) & s.scheme.eq(scheme) & s.variant.eq(v) & s.budget.eq(b)].nH_MAE.iloc[0]
                       for b in (0, 1, 2)] for v in VARIANTS])
        im = ax.imshow(z, cmap='YlGnBu', vmin=0, vmax=vmax, aspect='auto')
        for i in range(len(VARIANTS)):
            for j in range(3):
                ax.text(j, i, f'{z[i,j]:.3f}' if np.isfinite(z[i,j]) else 'invalid', ha='center', va='center',
                        color='white' if z[i,j] > .55*vmax else 'black', fontsize=7)
        ax.set_xticks(range(3), ['0', '1', '2'])
        ax.set_yticks(range(len(VARIANTS)), VARIANTS, fontsize=7)
        ax.set(xlabel='Hall anchors per DOI', title='Leave-one-DOI-out' if scheme=='DOI' else 'Leave-related-group-out')
    fig.subplots_adjust(left=.28, right=.82, wspace=.12, top=.95, bottom=.07)
    colorbar_axes = fig.add_axes([.85, .20, .025, .60])
    fig.colorbar(im, cax=colorbar_axes, label='Common-twenty DOI-macro MAE (dex)')
    for ext in ('png', 'pdf'):
        fig.savefig(folder/f'common20_all_methods.{ext}', dpi=180)
    plt.close(fig)
    p = read(OUT/'prediction/per_family.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    methods = ('FixedPhysical', 'InventoryProcess_internal', 'OutputProcess', 'MedianActivation', 'DoseLineAffine', 'DirectRidge')
    families = sorted(p.family.unique())
    for ax, budget in zip(axes, (0, 2)):
        for v in methods:
            d=p[p.scheme.eq('cluster') & p.scope.eq('common20') & p.variant.eq(v) & p.budget.eq(budget)].set_index('family')
            ax.plot(range(len(families)), [d.loc[f].nH_MAE if f in d.index else np.nan for f in families], 'o-', label=v, ms=3)
        ax.set_xticks(range(len(families)), [f.split('/')[-1] for f in families], rotation=35, ha='right', fontsize=7)
        ax.set(ylabel='Common-query MAE (dex)', title=f'Related-group exclusion, {budget} anchors per DOI')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(folder/f'family_errors.{ext}', dpi=180)
    plt.close(fig)
    e.write(folder/'provenance.json', dict(summary_sha256=e.sha(OUT/'prediction/summary.csv'),
        family_sha256=e.sha(OUT/'prediction/per_family.csv'), all19_primary_variants_in_heatmap=True,
        family_plot='six explicitly named diagnostics;all variants retained in CSV and heatmap', no_new_fits=True))


def main():
    for p in ('prediction/audit.json', 'delivery/cohort_replay_v2/audit.json', 'external_Azizie_v2/audit.json',
              'physical_scope/audit.json', 'method_policy/audit.json', 'analysis/audit.json',
              'analysis/candidate_diagnosis/audit.json', 'Leedy_label_sensitivity_v2/audit.json', 'delivery/final_checks/audit.json'):
        assert json.loads((OUT/p).read_text())['passed'], p
    figures()
    audit = json.loads((OUT/'analysis/audit.json').read_text())
    replay = json.loads((OUT/'delivery/cohort_replay_v2/audit.json').read_text())
    physical = json.loads((OUT/'physical_scope/audit.json').read_text())
    exported = json.loads((OUT/'delivery/export_complete.json').read_text())
    external = json.loads((OUT/'external_Azizie_v2/execution_complete.json').read_text())
    paired = read(OUT/'prediction/paired_chain_summary.csv')
    blocks = ['# ML与Sandbox：共享库存学习与跨来源掺杂性质预测', r'''## 1. 本轮回答的预测问题

目标是实际预测掺杂性质：给定Si/Sn元素、实际掺入化学浓度C及可追溯条件，预测300K比较约定下的Hall载流子浓度nH，并输出迁移率muH和派生电导sigma。新方法在训练后复用同一个共享函数，先改变缺陷库存，再重新求解电荷和输运。理论说明服务于这个预测任务，不能用机制解释或数值门槛代替留出误差。

本轮将Leedy2018四个近均匀化学/Hall条件加入独立新版：33条件、6 DOI、5个保守关联组。原29条件、原角色及旧共同18查询不改；新共同2查询组成固定common20。数据是已公开、已被本研究查看的条件关联，不是新物理样品、盲测或已证明独立实验室队列。

主任务训练只用nH，不用迁移率或变温标签。全部19变体在外层分数产生前固定，包含内部学习、输出学习、直接ML、单参数和双参数来源强控制。没有新增HSE、DFT、MD或实体测量。''']
    blocks.append(r'''## 2. 数据输入与比较协议

[Leedy2018原文](https://corescholar.libraries.wright.edu/physics/1482/)DOI10.1063/1.5047214的0.01/0.025/0.1/1wt条件采用印刷SIMS代表浓度，而不是由靶材比例换算；第5个0.05wt存在显著纵向非均匀，独立保留，不混入本轮均匀输入主任务。0.025wt图与正文的Hall值有冲突，原值及另一版本均保留；保留记录不等于已经完成另一个版本的重训验证。没有原始晶片ID，故只能说公开条件对应。

名义生长550C和接触退火470C/60s不等于完整冷却历史。300K是本研究比较约定，不冒称原文精确仪器温度。分子O2分压按0.05×1.3Pa/101325换算；其他未知压力仍为显式1e-5atm有效情景，不能把O3或等离子体腔压直接当平衡O2。H/碳缺失采用未知零情景，不能解释为实测无杂质。

两套留出分别按DOI和保守关联组删除完整训练单元。Leedy与McCandless基于机构关联合为一组，不宣称同一生长实验室；也未证明其余各组完全独立。每个DOI使用相同的0/1/2个Hall锚点；含两DOI的关联组总标签为0/2/4，不是整个关联组只有0/1/2个标签。关联组留出不将不同DOI强制共享来源修正。

共享超参数在外层训练单元内，以零锚点内层MAE选择；来源惩罚在相同内层角色的查询MAE选择。外层查询标签不进入模型或来源适应。主表同时保留全部查询33/26/20与固定common20，预算间比较优先用后者。每个DOI先取查询均值，再等权宏平均；另报5组宏平均和最差DOI，不能把33条件当33个独立来源。''')
    blocks.append('所有方法的化学/Hall队列、查询和锚点标签预算相同，非线性拟合的预设启动与细化上限相同；这不等于实际运行时间或历史物理信息成本相同。物理分支额外使用冻结KROGER缺陷目录、DOS及散射先验，直接ML无这些能量信息。本轮没有新增DFT，但不能据此称物理数据库是零成本或各方法监督信息完全相同。逐拟合和来源阶段实际调用/时间均保存。DirectRidge的已报告分子氧标志在本队列仅Leedy为1，语义为输入可用性，部署按oxygen_input_kind恢复，不依赖新样品DOI。该标志与来源关联，不能据它学习出的系数作氧气因果解释。')
    blocks.append(r'''## 3. ML如何进入真实物理状态

令x=(log10[C/(1cm^-3)]−18)/2，t=(Tg/摄氏度−700)/300，s为Sn指示量。三种特征phi分别为(1)、(1,x)、(1,x,t,s)。共享函数为

\[
h_\theta(C,u)=6\tanh(\phi^T\theta/6).
\]

theta为训练系数；u为这些显式输入；h是无量纲log10库存赔率。常数/剂量/工艺模型连同有效Tf与背景浓度分别为3/4/6个参数，固定物理为2个。h的±6范围及theta的±12界限是数值设计，不是第一性原理定理。

参考高温状态先由缺陷能量、电中性、总外来原子预算求出母类库存c_i^0。i为相同原子组成的缺陷母类，k_i是其中外来原子数。将纯掺杂且额外Ga亏损的复合物池记为Cplx，两个孤立替位池记为Iso，定义U=sum_Cplx k_i c_i^0，V=sum_Iso c_i^0，W=U+V。选择原子份额f使

\[
J(f)=f\ln\frac{f}{V/W}+(1-f)\ln\frac{1-f}{U/W}-(\ln10)hf
\]

最小。求导得J'=ln[f/(1−f)]−ln(V/U)−ln(10)h、J''=1/f+1/(1−f)>0，因此唯一内部极小点为

\[
f=\operatorname{logistic}\{\ln(V/U)+(\ln10)h\}.
\]

复合物池乘W(1−f)/U，孤立池乘Wf/V，各池内部比例不变。新两池外来原子总数为W(1−f)+Wf=W；其他母类不改。两池无O，故O计数也不变；Ga亏损一般改变，这是一种开放Ga的有效端点，不能称闭合Ga/O反应网络或真实动力学耗散证明。

本轮孤立施主采用固定完全离化分支（localized_fraction=0），电荷+1。虽沿用参数字典中的0.0366eV字段，它在此分支不提供中性占据贡献，不能声称本轮学习了浅施主离化能。学习的是有效库存分配，不是新缺陷能量、微观反应速率或因果生长机制。有效Tf也不能读成测得的冻结温度。''')
    blocks.append(r'''## 4. 电荷根、隐式梯度与相容性质

冻结库存后，对启用的常规电荷态，pi_iq正比于g_iq exp[−(A_iq+qEF)/(kB T)]。q为整数电荷，g为简并权重，A/EF为eV，kB为eV/K；各母类概率和为1。对分子与分母共同求导得到

\[
\frac{\partial\bar q_i}{\partial E_F}=-\frac{\operatorname{Var}_i(q)}{k_BT},\qquad
Q(E_F)=\sum_i c_i\bar q_i+N_{bg}+\sum_a p_a-n=0.
\]

实现保留价带孔与旧自陷孔通道；自由电子使用C2数值FD积分。在固定完全离化孤立施主的当前极限，该部分方差为0。其余方差非负、电子压缩率正、孔导数负，给出当前有限载流子域内Q_EF<0，即至多一个根；存在性仍须实际括区间及残差检查。不能将此推导推广到所有未启用的混合施主统计。

对连续参数w隐式求导：dEF/dw=−Q_w/Q_EF。高温多变量根用du/dw=−F_u^-1 F_w。完整g可为log10 nH、log10 muH或log10 sigma；因h本身随剂量变化，必须补上

\[
\frac{dg}{d\log_{10}C}=\left.\frac{\partial g}{\partial\log_{10}C}\right|_h+
\frac{\partial g}{\partial h}\frac{\theta_{dose}}2\operatorname{sech}^2(\phi^T\theta/6).
\]

第一项包含高温参考库存的剂量导数；第二项只补新共享场，另有生长温度导数theta_T sech²/300。已通过实际解的独立扰动检查；不是对CV选择或来源重拟合映射的导数。

同一状态给出Nq2=Nbg+sum_iq q² c_i pi_iq，以其和自由n计算单标量质量、离化散射Born/RTA输运矩，再加入205cm2/(V s)固定额外迁移率先验。205不是本轮算出的声子散射。Hall读出nH=n/rH，muH=rH mu_drift，因此

\[
\sigma=e\,n_H\mu_H=e\,n\mu_{drift}.
\]

输出残差/输出来源校准改变nH而保留原物理mu，其sigma仍派生，但nH不再属于原缺陷状态；作为经验强控制报告。所有方法另报共同CT(predicted nH)迁移率，区分载流子与输运公式贡献。单电子有效性代码只检查n>价带p等必要条件及检测下限，不是实际材料单带、全部孔可忽略或Born近似的证明。''')
    blocks.append(r'''## 5. 训练、少锚点适应与数值证据

训练损失为来源/关联组等权nH对数均方误差，加gamma倍相对相同Tf/背景、h=0参考的响应改变量均方；gamma取0/0.1/1/无穷，最后一种精确退回参考。输出关系和直接ML采用对应参数化，强Ridge有训练内标准化及独立L2网格。权重、域与正则坐标均显式给出，不把数值惩罚当物理定理。

来源内部更新可加一个库存场z；输出控制加一个对数偏移；仿射强控制另有剂量斜率，共2参数但Hall标签预算相同。损失为平均锚点残差平方，加lambda倍有限已知输入集合Gamma上的响应变化平方。Gamma仅含输入，重复元组去重，lambda取0/0.01/0.1/1/10/无穷。这是给定输入集的条件适应，不是连续域安全证明、校准后验或全球最优证书。

161状态/导数检查、119训练/划分检查、131保存模型/来源检查、21仿射检查通过。117点隔离来源烟测和36目标函数核对合计279项；另83接口烟测含27点。这些小门槛与后续全队列回放不同，不能累加成材料实验样本。

初始BASE名称错误发生在协议/训练前，原代码与日志保留。初版读取规则错误阻止了KROGER物理NPZ，旧失败也保留；修复只允许精确路径且SHA验证的物理参考，未开放查询标签。冻结后未改变训练/外层模型公式。''')
    blocks.append(f"实际完成{audit['training_fit_records']}条拟合记录，其中950条数值非线性拟合，{audit['training_optimizer_starts']}次实际优化启动。{audit['initial_not_stopped']}个初始未停止记录按统一规则细化，最终选中未停止{audit['selected_not_stopped']}。训练审计16305项通过；这不认证全局最优或物理真值。来源阶段{audit['source_tasks']}个任务、{audit['actual_source_point_evaluations']}次点调用（包含经验点调用）、{audit['actual_source_vectors']}个候选向量；{audit['point_failure_records']}条点失败及{audit['invalid_candidate_solutions']}个无效候选完整保留。主输出{audit['main_rows']}行，数值{audit['main_numeric']}、有效{audit['main_valid']}。")
    blocks.append('## 6. 主预测结果：固定共同20查询\n\n误差均为log10单位dex，10的该数值次方是乘性误差尺度；宏MAE不是逐样本倍数均值。19变体全部报告；未完整覆盖则不计算选择性MAE。\n\n### DOI留出\n\n'+summary_table('DOI'))
    blocks.append('### 保守关联组留出\n\n'+summary_table('cluster'))
    blocks.append('![完整19变体共同查询误差；颜色统一，不以预算改变查询。](figures/common20_all_methods.png)')
    blocks.append('### 保守关联组等权指标\n\n下表仍是相同20查询，将同关联组内DOI先平均，再等权平均5组；与上表6 DOI等权的差异来自汇总权重，不是另一套预测。\n\n'+summary_table('cluster', column='nH_cluster_MAE'))
    blocks.append('### 全部查询结果\n\n预算0/1/2分别33/26/20查询。查询组成变化，不能只看该表随预算的升降判断锚点效果。\n\n'+summary_table('cluster', scope='all_query'))
    blocks.append('![按DOI展开的诊断；六方法固定展示，其余方法均保留于完整表与CSV。](figures/family_errors.png)')
    rows=['| 内部方法／相同形状输出对照 | 0锚点差值 | 1锚点差值 | 2锚点差值 |','| --- | --- | --- | --- |']
    for internal in ('InventoryConstant','InventoryDose','InventoryProcess','InventoryConstant_internal','InventoryDose_internal','InventoryProcess_internal'):
        control='Output'+internal.split('_')[0].replace('Inventory','')
        rows.append('| '+internal+'/'+control+' | '+' | '.join(value(OUT/'analysis/matched_comparisons.csv',
            dict(scheme='cluster',budget=b,internal=internal,control=control),'delta_group_macro') for b in (0,1,2))+' |')
    blocks.append('## 7. 内部学习的贡献必须如何判断\n\n下表为内部模型减去同形状输出模型的5组等权误差差值，负值才是内部更小。匹配的是共享关系形状和标签预算；_internal使用一个内部来源场，对照为输出来源偏移，含义明确区分。与固定物理、激活比例、直接Ridge及仿射强控制的全部逐来源差值、改善来源数和删一关联组敏感性见analysis/matched_comparisons.csv；这些是回顾性描述，不是校正多重比较的显著性检验。\n\n'+'\n'.join(rows))
    rows=['| 训练内选方法策略／关联组留出 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for policy in ('BestOverall','BestInternal'):
        rows.append('| '+policy+' | '+' | '.join(value(OUT/'method_policy/summary.csv',
            dict(scheme='cluster',policy=policy,budget=b,scope='common20'),'nH_MAE') for b in (0,1,2))+' |')
    blocks.append('操作策略BestOverall从19变体、BestInternal从3内部来源变体选择，依据训练内分数与预先固定平手规则，未看外层误差挑模型。Best是策略名称，不保证外层最优；本轮选族也未建立稳健增益。query_labels_used=False字段指外层/新应用查询；训练内验证查询用于选择。78决策/26别名仅重用既有预测，不是新增独立实验证据。预算变化可能同时换方法，不将策略的误差变化归因于单一来源更新。\n\n'+'\n'.join(rows))
    rows=['| 来源／解除惩罚的内部适应 | 1锚点MAE | 2锚点MAE | 说明 |','| --- | --- | --- | --- |']
    for f in sorted(read(OUT/'data/conditions33.csv').family.unique()):
        rows.append('| '+f+' | '+' | '.join(value(OUT/'analysis/candidate_diagnosis/per_family_candidates.csv',
            dict(scheme='cluster',variant='InventoryProcess_internal',family=f,budget=b,lambda_text=0.),'common_MAE') for b in (1,2))+' | 已有候选、事后诊断 |')
    blocks.append('### 已有全惩罚候选的探索性诊断\n\n外层结果完成后，仅重新读取原来保存的全lambda候选，未再拟合或求新状态。432来源候选行/72汇总显示：lambda=0可修复部分Leedy误差，却会严重损害其他来源；锚点小导数与查询大导数可同时存在。下表使用外层标签描述，不能据它反选惩罚、替换主结果或宣称oracle已验证。\n\n'+'\n'.join(rows))
    rows=['| 方法／Leedy固定共同2查询 | 原标签1锚点 | 正文1锚点 | 原标签2锚点 | 正文2锚点 |','| --- | --- | --- | --- | --- |']
    for v in VARIANTS:
        vals=[value(OUT/'Leedy_label_sensitivity_v2/summary.csv',dict(scheme='cluster',variant=v,budget=b,scope='common_query'),col)
              for b in (1,2) for col in ('primary_nH_MAE','alternative_nH_MAE')]
        rows.append('| '+v+' | '+' | '.join(vals)+' |')
    blocks.append('### 图与正文冲突的固定参数敏感性\n\n保持完整来源排除的共享权重及超参数，用正文0.025wt nH=3.25e19、mu=20替换允许锚点/评分中的该条标签。114组/456点全数值有效，536项审计通过，主数据不改。它没有执行以另一标签版本重训全33条件的敏感性，不能扩大解释。固定共同查询的标签自身不变；误差不变有时也可能来自绝对误差平台，不等于没有读取锚点。\n\n'+'\n'.join(rows))
    mu_counts=paired[paired.scheme.eq('cluster')&paired.variant.eq('FixedPhysical')&paired.scope.eq('common20')&paired.budget.eq(0)].iloc[0]
    blocks.append(f"## 8. 完整性质链：迁移率与派生电导\n\n全部33条件有9组可用mu配对、3 DOI；固定common20中实际为{int(mu_counts.paired_conditions)}组、{int(mu_counts.paired_DOIs)} DOI。下面所有预算保持该配对子集。没有新mu训练，也无独立sigma标签。sigma误差可能因nH/mu分量相消，不能当第三次独立成功。\n\n### 原生物理或经验迁移率MAE\n\n"+summary_table('cluster',column='predicted_log_mu_MAE'))
    blocks.append('### 统一CT(predicted nH)迁移率MAE\n\n同样的配对标签和预测nH，统一使用旧CT经验式；不能把此表称为利用实测nH的条件迁移率任务。\n\n'+summary_table('cluster',column='predicted_log_mu_CT_MAE'))
    rows=['| 方法的原生派生sigma MAE | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for v in VARIANTS:
        rows.append('| '+v+' | '+' | '.join(value(OUT/'prediction/paired_chain_summary.csv',
            dict(scheme='cluster',variant=v,scope='common20',budget=b),'physical_or_native_derived_sigma_MAE') for b in (0,1,2))+' |')
    blocks.append('### 同配对的派生电导误差\n\n'+'\n'.join(rows))
    blocks.append('## 9. 保存模型到Azizie过程条件的冻结转移\n\n共享模型与来源超参数固定，A任务为e/k两片近似同Hall化学读数（约2e18/3e19cm^-3），B任务为12片名义源温/速度经第24阶段central50::PairSelected校准得到化学代理。A/B中e/k重叠，实际仍为12个Hall条件，不是14个独立样品，也不是只改变化学输入的纯消融。B固定d/c为锚点，common10；A固定e为唯一锚点，common1为k。\n\n三个上下文为full DOI、full关联组，以及删除整个AFRL/Cornell/Wright关联组。后者避免已知关联来源训练，但不能改称独立实验室盲测。O3、缺失H与不完整历史等情景不变。a/j报告输入相同而Hall相差约1.610dex，因此确定性输入模型对B全部12/共同10的MAE分别至少约0.134/0.161dex；这是条件下界，不足以单独解释所有失败。')
    for context in ('DOI::full','cluster::full','cluster::without:related_AFRL_Cornell_Wright'):
        rows=['| 方法／12片过程化学代理 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
        for v in VARIANTS:
            rows.append('| '+v+' | '+' | '.join(value(OUT/'external_Azizie_v2/summary.csv',
                dict(context=context,variant=v,input_tier='learned_process_proxy_12',budget=b,scope='common_query'),'nH_MAE') for b in (0,1,2))+' |')
        blocks.append('### '+context+'：B任务共同10片\n\n'+'\n'.join(rows))
    rows=['| 方法／排除关联组的近似实测化学 | 0锚点 | 1锚点 | 说明 |','| --- | --- | --- | --- |']
    for v in VARIANTS:
        rows.append('| '+v+' | '+' | '.join(value(OUT/'external_Azizie_v2/summary.csv',
            dict(context='cluster::without:related_AFRL_Cornell_Wright',variant=v,input_tier='rounded_same_Hall_2',budget=b,scope='common_query'),'nH_MAE') for b in (0,1))+' | 单一k查询 |')
    blocks.append('### A任务：共同高剂量k条件\n\nUnitDose若刚好零误差，是印刷约数输入与标签相等，不能证明真实100%电激活。全19方法及另外两上下文的A结果都保留于CSV。\n\n'+'\n'.join(rows))
    blocks.append(f"实际转移{external['group_requests']}组请求、{external['rows']}行，数值{external['numeric']}、有效{external['valid']}；其余无效与外推警告也保留。不将这批旧公开条件称为未经查看的新盲集。")
    blocks.append('正式V1 API回放的四个工作进程在最终写JSON时失败，复现到Output预设same_physical_state_for_output的NumPy布尔类型；原代码、全部任务与日志保留。V2仅在同一冻结预测器外递归将NumPy标量/数组转换为Python原生JSON类型，非有限值仍拒绝；未改变共享参数、来源求解、主结果或物理公式。V2完整重做回放后才执行新条件转移、标签敏感性和最终CLI检查。不能把V1失效隐藏成一次成功运行。')
    blocks.append(f"## 10. 数值与物理适用域核对\n\n正式保存模型247预设、{exported['selected_model_files']}个最小参数文件，模型包不含查询标签，也未复制第三方KROGER数据。独立进程实际回放{replay['requests']}组/{replay['point_rows']}点，最大三性质对数差{replay['maximum_log_property_difference']:.3e}。保存模型可复用不等于新材料验证。\n\n另对完整训练上下文7种物理基础方法×33条件×两方案的462状态做无过滤诊断，这是已知输入状态核对，不是留出精度。{physical['numeric_states']}状态数值完成，独立512点输运重构最大差{physical['maximum_independent_mu_reconstruction_error']:.3e}；{physical['majority_current_weight_kl_lt1']}状态超过半数电流权重落在kℓ<1；{physical['shared_site_claim_exceeds_one']}状态总共享位点计数超过1；{physical['effective_Tf_above_growth']}状态有效Tf高于名义生长温度。\n\n\\(k\\ell=2E\\tau/\\hbar\\)只是散射近似的诊断量，非Born有效性的严格判据。原始目录中同母类各电荷态的位点计数并不完全一致，本诊断取逐列最大计数包络计算容量比，不能解释成唯一真实共同位点占据；也不构成完整排斥格点模型。以上诊断不删除任何主预测或改变分数；Tf超出生长温度也不能在缺完整热历史时唯一归因。")
    blocks.append(r'''## 11. 文献定位与理论边界

[KROGER2025](https://doi.org/10.1039/D4CP04817B)已经提供缺陷统计及完整/部分平衡框架；[py-sc-fermi论文](https://joss.theoj.org/papers/10.21105/joss.04962)与[当前文档](https://py-sc-fermi.readthedocs.io/en/latest/)也覆盖自洽费米能级与冻结浓度约束，当前文档能力不能全部追溯为2023版功能。因而单纯把缺陷数据库、热力学求解和冻结库存串起来不是本轮独创。

[AD-NEGF](https://arxiv.org/abs/2202.05098v2)已将隐式可微物理层用于输运参数学习与掺杂优化；[npj2023材料生成与物理模拟工作](https://www.nature.com/articles/s41524-023-01080-x)也说明ML与模拟闭环已有先例。这些对照不等于相同Ga2O3条件下的复现实验，不能据此声称本方法优于它们，也不能把通用“ML+物理”作为未经检验的首次创新。

本轮可检验的具体方法对象是：化学原子预算内的条件库存函数、同一电荷/FD输运状态的可微读出、训练内选择的少锚点适应，以及完整跨来源对照。是否形成有用增量必须由上述全部结果判断；不能把每一次负面结果改写成“创新已证实”。文献阅读深度、缓存与来源哈希见literature_refresh/primary_sources.json和docs/ML_SANDBOX_CONTEXT_PRIOR_ART_20261018.md。

完整数学说明见docs/ML_SANDBOX_CONTEXT_THEORY_20261018.md；数据、预算、导数、读出与失败边界均与实际代码对应。正库存、守恒、IFT成立、优化器停止与sigma恒等式只是必要的实现条件，彼此均不能替代跨来源预测准确率。''')
    notes=OUT/'analysis/predictive_conclusions.md'
    assert notes.exists(), 'Read actual results and write evidence-specific conclusions before exporting'
    blocks.append(notes.read_text())
    blocks.append('## 13. 复现入口与剩余目标\n\n脚本顺序：phase25_data、gates/learning_gates、training、training_audit、prediction、prediction_audit、export、cohort_replay、external、external_audit，再analysis、method_policy、physical_scope。现有结果应先校验SHA，不能直接覆盖冻结文件。所有结果CSV通过csv_io_v2.read_result_csv读取，保留字样样品ID不被缺失值解析吞掉。\n\n公共predict_context_properties_v2.py接受保存模型预设、10个显式输入字段及最多2个Hall锚点；不接收查询目标表。零锚点物理模型可导出共享参数/输入IFT，不提供CV或来源重拟合导数。本轮主要为300K、Si/Sn；未经验证的稀土、新温度与器件光电性能不能用此队列分数背书。\n\n系统goal仍active：持续优化ML+Sandbox对掺杂性质的预测，使用实际留出增量决定下一项可检验改进，不以本报告完成替代目标，也不预设一定优于强基线或一定发表。')
    report=OUT/'ML与Sandbox_共享库存学习与跨来源性质预测.md'
    text='\n\n'.join(blocks)+'\n'
    assert not any(ord(c)<32 and c!='\n' for c in text)
    report.write_text(text)
    e.write(OUT/'report_claims.json',dict(report_sha256=e.sha(report),claims=CLAIMS,
        source_code_sha256=e.sha(Path(__file__)), no_new_predictions_or_fits=True,
        json_audit_sources={str((OUT/p).relative_to(ROOT)):e.sha(OUT/p) for p in ('analysis/audit.json',
          'delivery/cohort_replay_v2/audit.json','physical_scope/audit.json','delivery/export_complete.json','external_Azizie_v2/execution_complete.json')}))
    print('Wrote phase25 report',len(CLAIMS),'traceable CSV values',flush=True)


if __name__=='__main__':
    main()
