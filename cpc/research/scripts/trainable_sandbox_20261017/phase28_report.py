"""Traceable source/transfer/core report; does not retrain or select a winner."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

OUT=e.BASE/'phase28_paired_transport'
TRANSFER=OUT/'frozen_transport_v1'
REPORT=OUT/'report_v1'
THEORY=ROOT/'docs/ML_SANDBOX_JOINT_SCATTERING_THEORY_PHASE28_20261019.md'
TITLE='ML与Sandbox_Sn配对观测与联合学习接口研究'
PRIMARY_CONTEXT='cluster::without:10.1007/s10853-015-9693-6'
PRIMARY='FeasibleProcess_internal'
OXYGEN=('old_effective','higher_effective_activity')
CLAIMS=[];HASHES={}


def digest(path):
    key=str(path)
    if key not in HASHES:HASHES[key]=e.sha(path)
    return HASHES[key]


def number(path,selector,column):
    frame=read(path)
    for key,value in selector.items():frame=frame[frame[key].eq(value)]
    assert len(frame)==1,(selector,column)
    value=float(frame[column].iloc[0])
    value=value if np.isfinite(value) else None
    CLAIMS.append(dict(kind='csv',path=str(path.relative_to(ROOT)),sha256=digest(path),selector=selector,column=column,value=value))
    return '未覆盖' if value is None else f'{value:.6f}'


def fact(path,keys):
    data=json.loads(path.read_text())
    for key in keys:data=data[key]
    CLAIMS.append(dict(kind='json',path=str(path.relative_to(ROOT)),sha256=digest(path),keys=keys,value=data))
    return str(data)


def metric(variant,oxygen,column,context=PRIMARY_CONTEXT):
    return number(TRANSFER/'summary.csv',dict(context=context,variant=variant,oxygen_scenario=oxygen,
        scope='four_T_diagnostic_average'),column)


def all_methods(context,variants):
    rows=['| 方法 | nH旧氧 | nH高活度 | μ旧氧 | μ高活度 |','| --- | --- | --- | --- | --- |']
    order=[PRIMARY]+[v for v in variants if v!=PRIMARY]
    for variant in order:
        values=[metric(variant,o,c,context) for c in ('nH_MAE_dex','muH_MAE_dex') for o in OXYGEN]
        rows.append('| '+variant+' | '+' | '.join(values)+' |')
    return '\n'.join(rows)


def figures(frame,conditions):
    folder=OUT/'figures_v1';folder.mkdir(exist_ok=True)
    selected=[PRIMARY,'ActivationProcess','MedianActivation','UnitDose','DirectRidge']
    labels={PRIMARY:'Selected primary (reference fallback)','ActivationProcess':'Activation process',
        'MedianActivation':'Median activation','UnitDose':'Unit dose','DirectRidge':'Direct Ridge'}
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    for i,variant in enumerate(selected):
        group=frame[frame.context.eq(PRIMARY_CONTEXT)&frame.scope.eq('four_T_diagnostic_average')&frame.variant.eq(variant)]
        a=group[group.oxygen_scenario.eq(OXYGEN[0])].iloc[0];b=group[group.oxygen_scenario.eq(OXYGEN[1])].iloc[0]
        color=f'C{i}'
        ax.plot([a.nH_MAE_dex,b.nH_MAE_dex],[a.muH_MAE_dex,b.muH_MAE_dex],color=color,alpha=.7,lw=1)
        ax.scatter(a.nH_MAE_dex,a.muH_MAE_dex,color=color,marker='o',s=42,label=labels[variant])
        ax.scatter(b.nH_MAE_dex,b.muH_MAE_dex,facecolors='none',edgecolors=color,marker='s',s=65)
    ax.set(xlabel='nH MAE (dex)',ylabel='Native / assigned muH MAE (dex)',
        title='Two Sn conditions; old IKZ cohort excluded\nCircles: old effective activity; squares: higher assumed activity')
    ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best');fig.tight_layout()
    fig.savefig(folder/'carrier_mobility_tradeoff.png',dpi=160);fig.savefig(folder/'carrier_mobility_tradeoff.pdf');plt.close(fig)
    q=conditions[conditions.identity_applicable.eq(True)].copy()
    fig,ax=plt.subplots(figsize=(7.2,3.4));positions=np.arange(len(q))
    for i,row in enumerate(q.itertuples()):
        color='C0' if bool(row.rounding_only_identity_possible) else 'C3'
        ax.errorbar(row.identity_q,i,xerr=[[row.identity_q-row.identity_q_lo],[row.identity_q_hi-row.identity_q]],
            fmt='o',color=color,capsize=4)
    ax.axvline(1,color='black',ls='--',lw=1)
    ax.set(yticks=positions,yticklabels=['UID','Sn nominal 0.05 mol%','Sn nominal 0.1 mol%',
        'Sn nominal 1.5 mol%','Sn nominal 2 mol%'],xlabel='e * reported nH * reported muH * reported rho',
        title='Je 2023: display-rounding intervals, not experimental confidence')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.tight_layout()
    fig.savefig(folder/'hall_identity_rounding.png',dpi=160);fig.savefig(folder/'hall_identity_rounding.pdf');plt.close(fig)


def main():
    assert json.loads((TRANSFER/'audit.json').read_text())['passed']
    assert json.loads((OUT/'source_qualification_v1/audit.json').read_text())['passed']
    assert json.loads((OUT/'joint_scattering_core_v1/gates.json').read_text())['passed']
    theory=THEORY.read_text()
    assert not any(ord(c)<32 and c not in '\n' for c in theory)
    assert theory.count(r'\(')==theory.count(r'\)')>0
    assert theory.count(r'\[')==theory.count(r'\]')>0
    REPORT.mkdir(exist_ok=True)
    summary=read(TRANSFER/'summary.csv');conditions=read(OUT/'source_qualification_v1/condition_audit.csv')
    variants=sorted(summary.variant.unique());assert len(variants)==48
    figures(summary,conditions)
    source=OUT/'source_qualification_v1/summary.json'
    transfer=TRANSFER/'execution_complete.json'
    gates=OUT/'joint_scattering_core_v1/gates.json'
    text=f"""# ML＋Sandbox：Sn配对观测、冻结性质预测与联合学习接口

## 1. 本轮目的与实际完成范围

研究方向保持为可训练ML＋Sandbox预测掺杂性质，载流子浓度为第一终点。本轮完成新增同条件观测核查、现有冻结方法的完整性质链情景检验，以及下一项联合学习接口的实现和导数核验。整体goal仍active。

原33条件、6DOI、5关联组、0/1/2锚点角色和common20均保持冻结。新来源另存版本；没有把新增表格直接并入旧训练集。新增物理实验、DFT、MD和GPU作业均为0；GPU1授权保留。

共审计{fact(source,['source_conditions'])}个条件，确认{fact(source,['reported_condition_triples'])}个表列同条件三量配对，其中{fact(source,['intentionally_Sn_doped_triples'])}个为有意Sn掺杂，另1个为UID。配对是论文内条件对应，不是已取得原始晶片ID，也不是材料模型适用性认证。

冻结预测共{fact(transfer,['requests'])}个请求、{fact(transfer,['point_rows'])}行，{fact(transfer,['numeric'])}行数值且观测域有效。全部48方法保留。12个模型—上下文组合因旧保存拟合不合格而拒绝，共192行情景输出缺失；未重新拟合或补成有效预测。有效与否仅指实现的数值/观测门槛。

新联合接口通过{fact(gates,['checks'])}项核验，保存{fact(gates,['full_numerical_states'])}个数值状态，最大缩放差异{fact(gates,['maximum_scaled_difference'])}。这些不是新训练或材料准确度结果。

## 2. 原有Sn来源为何仍不能直接拼接迁移率

Baldini原文Fig5给出化学浓度—Hall浓度点；迁移率未画出，仅给范围和最高值，缺乏将其与具体SIMS点相连的样品证据。Mauze原文Fig4(a,b)缺少统一样品编号，另两组高迁移率是约1000nm厚样品，不能移入200–300nm队列。其变温Hall拟合施主浓度也不是独立SIMS输入。

旧图中的高Sn低Hall末点为普通三角形标记，本轮未将其改为删失或非导电标签。原8/11个载流子条件及原52组条件迁移率的资格说明保持原样。

## 3. 两个新公开来源：配对与适用性分别核查

Gogova等2015作者上传的9页原始论文，通过网页PDF文本和作者全文页面核对实验段、电学段及Table1；本地直下的403和旧PSU连接超时均保留。本地没有该PDF数值哈希，不能把网页读取写成本地全文归档。

Table1明确配对A、B的化学浓度、室温Hall浓度与迁移率；C两项Hall值无法测量，无数值上限，不能填0。样品A为Sn8e18、nH6e17、μ10；B为Sn3.6e18、nH7.1e17、μ13.7，浓度单位cm⁻³、迁移率单位cm²/(V·s)。

该来源在IKZ进行TMGa/TESn/H2O生长，报道了C-H相关Raman信号但无定量C/H；单样品温度、压力和厚度对应缺失。保守地将其与旧Baldini归为关联体系。表中化学量标为SIMS，方法还描述部分样品使用浓度—前驱体流量标定；逐行直接测量与标定路径未进一步明确。

Je等2023的SIMS与Hall表有相同条件名称，5组表列配对保留，另保留只有SIMS的商业参考和未表征的3mol%样品。SIMS为200–500nm平均，Hall试片约1mm厚，Si/Fe浓度显著，实际Hall温度未明确。这些不是当前单一Si/Sn输入接口已支持的完整组成。

对Je的5行，{fact(source,['identity_not_explained_by_display_rounding'])}行偏差不能仅由末位显示精度解释；受限均匀Si/Sn正电预算亦有{fact(source,['restricted_uniform_charge_budget_exceeded'])}行超界。它们不是同一个错误诊断，也没有唯一归因。本轮没有擅自修正任何表值。

![Je原表的定义恒等式检查；横线只表示显示舍入区间。](figures_v1/hall_identity_rounding.png)

## 4. 两个Sn配对条件的冻结预测协议

主要上下文排除旧IKZ/Baldini来源，另报告全来源参数的对照。两个上下文均不含新DOI标签。预测用原模型参数和0个Hall锚点，运行时禁止读取实验/训练结果表。配对标签只用于事后评分。

每个条件采用原文列出的775/800/825/850摄氏度四种情景，并分别使用原有效氧输入1e-5与较高有效氧活度1e-2。后者不是测得的H2O氧化还原条件或pO2，也不是已证明包含真值的区间。没有从温度范围、总压力或表值反推某一实际条件。

报告的每个氧设置误差，是4种温度和2个条件的等权平均；独立观测条件始终只有2个，不能把16行情景当作16个样品。源文RT使用300K比较约定；H/C采用接口的unknown-zero情景，明确不代表实验测得不存在。

### 4.1 载流子与迁移率：必须同时报告

主要方法在排除IKZ时实际采用ParentReference回退，与固定物理数值相同。原有效氧情景下nH/μ误差为{metric(PRIMARY,OXYGEN[0],'nH_MAE_dex')}/{metric(PRIMARY,OXYGEN[0],'muH_MAE_dex')}dex；较高活度情景为{metric(PRIMARY,OXYGEN[1],'nH_MAE_dex')}/{metric(PRIMARY,OXYGEN[1],'muH_MAE_dex')}dex。不能选择后一个情景并宣称实测外推已达到该精度，也不能把回退结果归功于新共享关系。

相同载流子预测用CT读出时，μ误差分别为{metric(PRIMARY,OXYGEN[0],'mu_CT_MAE_dex')}与{metric(PRIMARY,OXYGEN[1],'mu_CT_MAE_dex')}dex。原生输运在该比较中更好，但仍有明显绝对偏差，且没有Born/RTA材料适用性认证。

![两组Sn条件的载流子—迁移率情景比较；连接线表示氧输入变化，不是置信带。](figures_v1/carrier_mobility_tradeoff.png)

### 4.2 派生电导不能替代两个分量

DirectRidge在较高氧情景下nH误差{metric('DirectRidge',OXYGEN[1],'nH_MAE_dex')}dex、μ误差{metric('DirectRidge',OXYGEN[1],'muH_MAE_dex')}dex，而派生σ误差仅{metric('DirectRidge',OXYGEN[1],'sigma_MAE_dex')}dex。这体现相反对数误差的抵消；不证明两个构成量准确，也不增加独立验证标签。

## 5. 下一项联合学习的理论与已实现接口

下文完整给出定义、假设、正散射导数、库存链式连接和配对监督的相关性。其作用是为下一项实际预测训练提供可核验方程。

"""
    text+='\n'.join(theory.splitlines()[2:])+'\n\n'
    text+="""## 6. 与旧联合学习的区别及下一项决定性比较

第16阶段已经训练过nH/μ联合损失，并有常数μ输出头对照；不能将联合监督本身重新称为未做过的创新。其固定205通道和温度/施主统计对象，与本轮新增的显式正散射参数及电荷可行库存接口不同。新接口的参数、状态、输入和来源导数已经通过测试，但尚未训练或获得新增精度。

下一步另冻数据与协议：原33条件及角色保持可回查，新Gogova两组配对另列；Je的混合组成与表内冲突暂不作为当前单一掺杂完整输入真值。所有缺失工艺采用显式情景，情景重复不增加物理样本或标签数；整个IKZ关联体系同时排除。

决定性对照包括：固定库存与205、固定库存加可学习正散射、常数库存加正散射、剂量/工艺共享库存加正散射，以及同标签预算的直接多输出ML、CT加常数位置校准。配对二元损失替代重复的载流子项；损失/正则化和来源适应只在训练数据内选择。

nH仍是主要终点，完整性质链另报。新来源只有2个数值条件，2锚点会用尽其载流子标签；不得制造额外留出点。μ在已知Hall锚点上的条件预测与从掺杂输入的完整预测分开。

只有条件共享库存相对散射常数、常数库存和直接ML有可重复的源外增量，才支持相应方法贡献。若只能校准整体散射尺度，就如实解释；若nH恶化，也保留。当前整体goal未完成。

## 7. 全48方法：主要关联来源排除情景

所有数字均为dex，列分别是两个氧设置下的nH和原生/指定迁移率误差。四温度平均是协议定义的敏感性平均，不是实测工艺边缘化或置信区间。“未覆盖”保留旧模型拒绝。

"""
    text+=all_methods(PRIMARY_CONTEXT,variants)+'\n\n'
    text+='## 8. 全48方法：全来源参数对照\n\n此上下文使用包含旧IKZ队列的参数；不能作为新实验室独立验证。\n\n'
    text+=all_methods('cluster::full',variants)+'\n\n'
    text+="""## 9. 复现与证据索引

source_observations_v1.json保存全部原始表值、配对层级、缺失量及来源；source_qualification_v1保存舍入与受限库存检查。frozen_transport_v1保存运行前协议、96请求、完整输出、1536逐行情景、960指标行和38152项评分/隔离核查。其16个输入情景不进入新参数拟合。

joint_scattering_core_v1保存运行前代码哈希、318项核验和15个数值状态；新源代码分别为learnable_constant_tau.py、joint_inventory_scattering.py和paired_hall_observations.py。report_v1保存表值回查，figures_v1提供可独立导出的科学图。

正文/附录、旧队列、已冻结代码及所有不利结果均未改写。原始KROGER数值不复制进公开交付。一个未冻结理论草稿的TeX转义错误已在渲染前修复，原草稿保留；模型、分数和上述数值门槛未变。

公开来源：[Gogova原论文DOI](https://doi.org/10.1039/C5CE01106J)、[作者上传原论文](https://www.researchgate.net/publication/280631468_Homo-_and_heteroepitaxial_growth_of_Sn-doped_b-Ga2O3_layers_by_MOVPE)、[Je原论文](https://journal.kci.go.kr/jkcgct/archive/articleView?artiId=ART002955272)、[NIST Hall测量定义](https://www.nist.gov/pml/nanoscale-device-characterization-division/popular-links/hall-effect/hall-effect)、[Varley原论文DOI](https://doi.org/10.1063/1.3499306)。
"""
    md=OUT/(TITLE+'.md');md.write_text(text)
    e.write(REPORT/'numeric_claims.json',dict(created_utc=e.now(),report_sha256=e.sha(md),
        theory_path=str(THEORY.relative_to(ROOT)),theory_sha256=e.sha(THEORY),claims=CLAIMS))
    verified=0
    for claim in CLAIMS:
        path=ROOT/claim['path'];assert e.sha(path)==claim['sha256']
        if claim['kind']=='csv':
            frame=read(path)
            for key,value in claim['selector'].items():frame=frame[frame[key].eq(value)]
            assert len(frame)==1
            value=float(frame[claim['column']].iloc[0]);value=value if np.isfinite(value) else None
        else:
            value=json.loads(path.read_text())
            for key in claim['keys']:value=value[key]
        assert value==claim['value'],claim
        verified+=1
    e.write(REPORT/'numeric_claims_audit.json',dict(passed=True,claims_checked=verified,
        theory_controls_absent=True,theory_inline_math_balanced=True,theory_display_math_balanced=True,
        no_new_model_training_or_selection=True,report_sha256=e.sha(md),goal_complete=False))
    print('Wrote phase28 report;',verified,'numeric claims checked;model training remains pending',flush=True)


if __name__=='__main__':main()
