"""Actual joint-prediction report; all dynamic values carry source/span evidence."""
from pathlib import Path
import sys,json,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from phase29_report_claims import Evidence,sanitize_units,audit
import phase29_data as data
import phase2_energy as e

OUT=data.OUT;REPORT=OUT/'report_v1';TITLE='ML与Sandbox_共享库存与正散射联合预测研究'
PRIMARY='JointProcess_internal'
MAIN=[PRIMARY,'Reference205_offset','ReferenceTau_internal','JointConstant_internal',
    'DirectMultiRidge_offset','MedianActivation_offset','UnitDose_offset']
LABELS={PRIMARY:'Joint process + internal source','Reference205_offset':'Reference inventory +205',
    'ReferenceTau_internal':'Reference inventory + learned tau0','JointConstant_internal':'Constant share + learned tau0',
    'DirectMultiRidge_offset':'Direct two-head Ridge','MedianActivation_offset':'Median activation +CT',
    'UnitDose_offset':'Unit dose +CT'}
THEORY=[
    ROOT/'docs/ML_SANDBOX_METHOD_PHASE29_V2_20261019.md',
    ROOT/'docs/ML_SANDBOX_JOINT_TRAINING_MATH_PHASE29_20261019.md',
    ROOT/'docs/ML_SANDBOX_SCENARIO_INFORMATION_PHASE29_20261019.md',
    ROOT/'docs/ML_SANDBOX_TRANSPORT_SCOPE_PHASE29_20261019.md',
    ROOT/'docs/ML_SANDBOX_SCATTER_BUDGET_CORRECTION_PHASE29_20261019.md',
    ROOT/'docs/ML_SANDBOX_JOINT_SCATTERING_THEORY_PHASE28_20261019.md',
]


def write_selected_models():
    plans=json.loads((OUT/'prediction/plan.json').read_text())['plans']
    choices=json.loads((OUT/'prediction/source_choices.json').read_text())
    source={(x['plan_id'],int(x['budget'])):x['selected']['lam'] for x in choices};rows=[]
    for p in plans:
        fit=json.loads((ROOT/p['fit_path']).read_text());method=p['spec']['method'];pars=fit['parameters']
        physical=method.startswith(('Joint','Share','Reference'))
        learned=method.startswith('Joint') or method=='ReferenceTau'
        mu0=(205*10**float(pars[-1]) if learned and pars is not None else 205. if physical else None)
        row=dict(variant=p['variant'],context='full' if p['outer_held'] is None else p['outer_held'],
            fit_key=p['fit_key'],method=method,gamma=fit['config'].get('gamma'),
            alpha=fit['config'].get('alpha',fit['config'].get('ridge_alpha')),
            parameters=pars,training_valid=fit['training_valid'],selected_success=fit['selected_success'],
            effective_extra_mobility_cm2Vs=mu0,transport_head=p['spec']['head'],
            paired_training_conditions=fit['paired_reported_conditions'])
        for b in (0,1,2):
            lam=source[(p['id'],b)]
            row['lambda'+str(b)+'_text']='no_update' if lam is None else str(lam)
        rows.append(row)
    e.csvwrite(REPORT/'selected_models.csv',rows)
    started=json.loads((OUT/'training_started.json').read_text())
    complete=json.loads((OUT/'training_complete.json').read_text())
    seconds=(datetime.datetime.fromisoformat(complete['created_utc'])-
        datetime.datetime.fromisoformat(started['created_utc'])).total_seconds()
    conditions=data.load_conditions();cases=data.load_cases();roles=data.load_roles()
    counts=[]
    for b in (0,1,2):
        r=roles[roles.budget.eq(b)]
        counts.append(dict(budget=b,anchors=int(r.role.eq('anchor').sum()),queries=int(r.role.eq('query').sum()),guards=int(r.role.eq('guard').sum())))
    statistics=dict(created_utc=e.now(),actual_training_wall_seconds=seconds,
        reported_conditions=len(conditions),scenario_rows=len(cases),
        paired_reported_conditions=int(conditions.mobility_eligible.eq(True).sum()),
        paired_scenario_rows=int(cases.mobility_eligible.eq(True).sum()),
        Si_conditions=int(conditions.element.eq('Si').sum()),Sn_conditions=int(conditions.element.eq('Sn').sum()),
        paired_Si=int((conditions.element.eq('Si')&conditions.mobility_eligible.eq(True)).sum()),
        paired_Sn=int((conditions.element.eq('Sn')&conditions.mobility_eligible.eq(True)).sum()),
        common_carrier_conditions=cases.loc[cases.primary_common20.eq(True),'sample_id'].nunique(),
        common_paired_conditions=cases.loc[cases.primary_common20.eq(True)&cases.mobility_eligible.eq(True),'sample_id'].nunique(),
        role_counts=counts,
        selected_method_rows=len(rows),raw_fit_parameters_not_unique_microphysics=True)
    e.write(REPORT/'study_statistics.json',statistics)
    return rows


def descriptive_comparisons():
    table=read(OUT/'prediction/summary.csv');families=read(OUT/'prediction/per_family.csv')
    controls=['ReferenceTau_internal','JointConstant_internal','ShareProcess205_internal',
        'DirectMultiRidge_offset','MedianActivation_offset','UnitDose_offset']
    records=[];verdicts=[]
    for control in controls:
        differences=[]
        for budget in (0,1,2):
            def line(variant):
                q=table[table.variant.eq(variant)&table.scope.eq('common20')&table.budget.eq(budget)]
                assert len(q)==1;return q.iloc[0]
            a,b=line(PRIMARY),line(control)
            good=bool(np.isfinite(a.nH_MAE) and np.isfinite(b.nH_MAE))
            delta=float(a.nH_MAE-b.nH_MAE) if good else None;differences.append(delta)
            signs=[]
            if good:
                p=families[families.scope.eq('common20')&families.budget.eq(budget)]
                x=p[p.variant.eq(PRIMARY)].set_index('family').nH_MAE
                y=p[p.variant.eq(control)].set_index('family').nH_MAE
                assert set(x.index)==set(y.index)
                signs=(x-y).to_numpy(float)
            records.append(dict(control=control,budget=budget,full_coverage=good,delta_nH_MAE=delta,
                improved_DOIs=int((signs<-1e-10).sum()) if good else None,
                worse_DOIs=int((signs>1e-10).sum()) if good else None,
                numerically_equal_DOIs=int((abs(signs)<=1e-10).sum()) if good else None,
                source_DOIs=len(signs) if good else None,direction_tolerance_is_not_significance=1e-10))
        if any(v is None for v in differences):
            message='至少一个预算缺少完整查询资格，不能作三预算全覆盖优势声明。'
        elif all(v<-1e-10 for v in differences):
            message='在固定回顾性队列的三个预算均为数值较低；独立验证与显著性未由此建立。'
        elif all(abs(v)<=1e-10 for v in differences):
            message='三个预算的共同查询误差数值一致，没有建立该比较中的增量。'
        else:
            message='未出现三个预算一致的误差下降，须逐预算和来源解释，不支持统一优于该对照。'
        verdicts.append(dict(control=control,deltas=differences,verdict=message))
    e.csvwrite(REPORT/'descriptive_comparisons.csv',records)
    e.write(REPORT/'descriptive_interpretation.json',dict(controls=verdicts,all_controls_preserved=True,
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (
            OUT/'prediction/summary.csv',OUT/'prediction/per_family.csv')},
        numerical_direction_tolerance=1e-10,no_pvalue_or_novelty_claim=True))


def figures(summary,predictions):
    folder=OUT/'figures_v1';folder.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=170);fig.savefig(folder/(name+'.pdf'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for variant in MAIN:
        q=summary[summary.variant.eq(variant)&summary.scope.eq('common20')].sort_values('budget')
        for ax,column in zip(axes,('nH_MAE','nH_group_macro_MAE')):
            ax.plot(q.budget,q[column],marker='o',ms=4,lw=1.2,label=LABELS[variant])
            ax.set(xlabel='Hall anchors per DOI',ylabel='nH MAE (dex)',xticks=[0,1,2]);ax.grid(alpha=.2)
    axes[0].set_title('Fixed common20; DOI macro')
    axes[1].set_title('Same predictions; related-group macro')
    axes[1].legend(fontsize=6.5,loc='best')
    save(fig,'carrier_common20')
    q=predictions[predictions.variant.eq(PRIMARY)&predictions.budget.eq(0)&
        predictions.role.eq('query')&predictions.mobility_eligible.eq(True)&predictions.numeric.eq(True)].copy()
    q['dn']=q.predicted-q.observed;q['dm']=q.predicted_log_mu-q.observed_log_mu
    fig,ax=plt.subplots(figsize=(6.2,4.7))
    for new,color,label in ((False,'C0','Old Si pairs'),(True,'C3','New Sn input scenarios')):
        a=q[q.new_Sn.eq(new)]
        good=a[a.valid.eq(True)];bad=a[~a.valid.eq(True)]
        ax.scatter(good.dn,good.dm,c=color,label=label,s=25,alpha=.5)
        if not bad.empty:ax.scatter(bad.dn,bad.dm,c='black',marker='x',s=38,label=label+' invalid')
        for sid,sub in good.groupby('sample_id'):
            ax.scatter(sub.dn.mean(),sub.dm.mean(),c=color,s=55,marker='D',edgecolors='black',lw=.5)
    ax.axhline(0,color='black',lw=.7);ax.axvline(0,color='black',lw=.7)
    if not q.empty:
        lower=min(q.dn.min(),-q.dm.max())-.1;upper=max(q.dn.max(),-q.dm.min())+.1
        ax.plot([lower,upper],[-lower,-upper],ls='--',c='grey',label='Derived sigma error =0')
    ax.set(xlabel='Signed log10 nH error (dex)',ylabel='Signed log10 muH error (dex)',
        title='Primary,zero anchors:component errors and cancellation')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'paired_component_errors')
    scope=read(OUT/'physical_scope/states.csv')
    plans=json.loads((OUT/'prediction/plan.json').read_text())['plans']
    key=next(p['fit_key'] for p in plans if p['outer_held'] is None and p['variant']==PRIMARY)
    current=scope[scope.scope.eq('full_training_inputs')&scope.fit_key.eq(key)&scope.numeric_state.eq(True)].copy()
    cases=data.load_cases().set_index('case_id');fig,ax=plt.subplots(figsize=(6.4,4.))
    for element,color in (('Si','C0'),('Sn','C3')):
        rows=current[current.case_id.map(cases.element).eq(element)]
        ax.scatter(rows.case_id.map(cases.incorporated_cm3),rows.fraction_sigma_kl_lt1,
            c=color,alpha=.5,s=24,label=element+' input scenarios')
    ax.set(xscale='log',ylim=(-.03,1.03),xlabel='Incorporated concentration (cm$^{-3}$)',
        ylabel='Current weight with k*l<1',title='Full-data primary state diagnostic;not held-out accuracy')
    ax.axhline(.5,c='grey',ls='--',lw=.8);ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'transport_scope')
    e.write(REPORT/'figure_evidence.json',dict(
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (
            OUT/'prediction/summary.csv',OUT/'prediction/predictions.csv',OUT/'physical_scope/states.csv',
            OUT/'prediction/plan.json',OUT/'data/scenarios49.csv')},
        output_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in folder.glob('*')},
        no_confidence_intervals_or_new_independent_samples=True,all_variants_retained_in_tables=True))


def metric(ev,variant,budget,column,scope='common20'):
    return ev.csv(OUT/'prediction/summary.csv',dict(variant=variant,budget=budget,scope=scope),column)


def table(ev,variants,column,scope):
    lines=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for v in variants:lines.append('| '+v+' | '+' | '.join(metric(ev,v,b,column,scope) for b in (0,1,2))+' |')
    return '\n'.join(lines)


def main():
    assert not (REPORT/'protocol_before_report.json').exists(),'preserve previous report before a versioned repair'
    assert json.loads((OUT/'post_prediction_independent_checks_complete.json').read_text())['passed']
    for path in ('prediction/audit.json','training_audit.json','api_replay/audit.json','physical_scope/audit.json',
            'information_audit/audit.json','scatter_budget_notation_audit_v1/audit.json'):
        assert json.loads((OUT/path).read_text())['passed'],path
    files=[OUT/p for p in ('training_complete.json','training_audit.json','prediction/summary.csv',
        'prediction/per_family.csv','prediction/predictions.csv','prediction/plan.json','prediction/source_choices.json',
        'prediction/execution_complete.json','api_replay/audit.json','delivery/audit.json',
        'physical_scope/audit.json','information_audit/audit.json','information_audit/fits.json')]
    code=[Path(__file__),Path(__file__).with_name('phase29_report_claims.py')]
    REPORT.mkdir(exist_ok=True)
    e.write(REPORT/'protocol_before_report.json',dict(created_utc=e.now(),
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in [*files,*THEORY]},
        fixed_primary=PRIMARY,all48_variants_included=True,no_new_training_or_selection=True,goal_complete=False))
    selected=write_selected_models()
    descriptive_comparisons()
    summary=read(OUT/'prediction/summary.csv');predictions=read(OUT/'prediction/predictions.csv')
    variants=[PRIMARY]+[v for v in sorted(summary.variant.unique()) if v!=PRIMARY]
    assert len(variants)==48
    figures(summary,predictions)
    ev=Evidence()
    text=compose(ev,variants,selected)
    text,unit_counts=sanitize_units(text)
    text,spans=ev.render(text)
    path=OUT/(TITLE+'.md');path.write_text(text)
    claims=dict(created_utc=e.now(),claims=ev.claims,spans=spans,report_sha256=e.sha(path),
        theory_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in THEORY},unit_substitutions=unit_counts,
        main_method_fixed_before_outer_results=True,no_new_fits_or_test_selection=True)
    e.write(REPORT/'numeric_claims.json',claims)
    checked=audit(path,claims);e.write(REPORT/'numeric_claims_audit.json',checked)
    print('Report29',checked['passed'],'claims',checked['claims_checked'],flush=True)
    if not checked['passed']:raise RuntimeError('Preserve numeric/span/math report failure')


def compose(ev,variants,selected):
    statistics=REPORT/'study_statistics.json';training=OUT/'training_complete.json'
    prediction=OUT/'prediction/execution_complete.json';api=OUT/'api_replay/audit.json'
    physical=OUT/'physical_scope/audit.json';information=OUT/'information_audit/audit.json'
    qualification=OUT/'data/qualification.json';gate=OUT/'numerical_gates_v2/gates.json'
    text=f"""# ML＋Sandbox：共享库存、正散射与掺杂性质联合预测

## 1. 本轮研究回答什么

研究目标是从掺杂元素、实际化学掺入量及可获得工艺条件出发，通过可训练共享关系与物理Sandbox预测性质。载流子浓度nH为第一终点，迁移率和相容电导率组成后续性质链。本报告是一个实际训练、来源留出和接口回放检查点；不以理论、训练损失或软件可运行单独代替预测贡献。

实际完成{ev.fact(training,['fit_records'])}个拟合记录，其中{ev.fact(training,['nonlinear_fit_records'])}个非线性记录，实际启动优化器{ev.fact(training,['actual_optimizer_starts'])}次。拟合记录不是材料实验数；多次初始化和训练子集也不是新增样品。

全部48个固定变体输出{ev.fact(prediction,['outer_scenario_rows'])}行情景预测。独立进程API回放{ev.fact(api,['requests'])}次请求，最大对数读出差{ev.fact(api,['maximum_absolute_log_difference'],'.3e')}。回放验证实现一致，不是新增材料准确度。

当前研究为反复使用既有公开队列的回顾性研发。原稿的宽泛九性质、器件/PDR和稀土排序结论不能由本报告恢复。正文主方法JointProcess_internal在查看外层结果前已固定，完整对照和失败均保留。

## 2. 数据契约：条件、情景与独立性

数据版本有{ev.fact(statistics,['reported_conditions'])}个报告条件、{ev.fact(qualification,['DOIs'])} DOI和{ev.fact(qualification,['related_groups'])}个保守关联组。其中Si条件{ev.fact(statistics,['Si_conditions'])}个、Sn条件{ev.fact(statistics,['Sn_conditions'])}个；配对迁移率标签{ev.fact(statistics,['paired_reported_conditions'])}组，分为{ev.fact(statistics,['paired_Si'])} Si与{ev.fact(statistics,['paired_Sn'])} Sn。报告条件不是已核实互不复用的晶片。

新Gogova两组Sn表列配对与旧Baldini按同IKZ体系一起排除，不能宣称新的独立实验室。Je的Si/Fe混合组成、深度/体积差异和表内冲突保留在审计中，没有强行成为本单掺杂模型的标签。Gogova C的Hall不可测不是数值0，也没有被伪造为第三个留出样品。

原33条件、原角色和common20保持可回查。主载流子共同查询为{ev.fact(statistics,['common_carrier_conditions'])}个条件；共同查询中的迁移率配对只有{ev.fact(statistics,['common_paired_conditions'])}组，来自既有Si队列。新增Sn在2锚点下没有剩余nH查询；其0/1锚点结果单列，不能放入固定共同查询制造优势。

总情景行数为{ev.fact(statistics,['scenario_rows'])}，其中配对情景行{ev.fact(statistics,['paired_scenario_rows'])}。新Sn每条件的四个温度与两个有效氧设置构成固定8情景，权重总和为1。它们不是测得的输入概率或已证明包含真实工艺的区间。H/C的unknown-zero也不证明样品没有H/C；水蒸气、CO2和总压力没有被直接替换为氧分压。

| 每DOI锚点预算 | 全队列锚点条件 | 查询条件 | guard条件 |
| --- | --- | --- | --- |
"""
    for b in (0,1,2):
        text+='| '+str(b)+' | '+' | '.join(ev.fact(statistics,['role_counts',b,k]) for k in ('anchors','queries','guards'))+' |\n'
    text+="""

正预算下继承的guard不作为额外锚点，也不参加查询评分；其输入仍属于声明的有限设计。完整配对数据只有两个原始观测方向，派生sigma不是第三个标签。

## 3. 主模型怎样保留ML＋Sandbox耦合

共享过程函数用常数、化学剂量、生长温度和Si/Sn指示变量生成内部库存份额；它进入原子预算与电荷补偿阈值。高温参考及冷电中性根确定同一状态的自由电子和总电荷平方预算。另一个全局正参数调整常数弛豫时间通道；Hall因子、nH、muH和sigma一起变化。

主过程模型有7个全局参数：有效温度、正背景、4个库存系数及1个正散射尺度。常数/剂量库存、只学散射及固定205对照分别缩减容量。没有逐样品自由散射参数，也没有新增神经网络来替代低维识别问题。微观形成能、真实冻结历史或散射机制并未因此被唯一反演。

本轮孤立施主固定完全离化；库存变化是端点闭合，不是已验证的反应动力学或四母类空间PDE。高温参考、Ga开放端点和未建模C等假设均保留。KROGER先验能量信息与直接ML的信息量分别说明，不能只因数据行数相同就宣称先验完全相同。

特别更正：实际总散射预算始终是正背景加显式缺陷电荷平方，而非只含缺陷求和。代码和导数没有遗漏该背景项；历史不完整记号保留，V2方法与更正说明给出精确定义。不能将此文字更正当成新增物理改进。

## 4. 训练、选择与公平对照

固定实验包括参考库存＋205、参考库存＋可学习正散射、常数/剂量/工艺库存＋固定或学习散射、同容量经验激活、直接双原始量Ridge，以及单位剂量/中位数激活/剂量直线。另保留nH-only、训练集常数输运校准和强仿射来源控制。直接模型与物理模型均只用本队列允许的载流子和配对标签；未调用旧52组条件迁移率的额外监督。

整个关联组的两类标签同时排除。共享超参数只用训练来源内部再次留组的零锚点nH误差选择。共享参数随后固定，来源lambda按训练内查询nH选择；查询迁移率不参加适应。相同0/1/2预算下，参数变化、函数头变化和来源适应分别记录。

每个非线性候选使用同样两初值及120次上限；任一未停止时按协议用原两初值和有限最佳点各360次统一细化。选择最低有限损失，若其仍未停止则保留而不服务，不因另一次较高损失已停止而改选。候选网格、实际启动/求解调用和时间均存档；参考模型不存在有效alpha正则时不重复无效网格。闭式Ridge的计算量与非线性求解不同，不能把候选数等同于实际算时。

训练按关联组、组内来源、来源内条件等权；主nH评价按DOI等权，另外报告关联组等权。多个情景先在同一条件内平均，不能重复增加条件权重。主MAE是绝对对数误差，显示6位供重现，不表示实验精度有6位。

数学上10的MAE次方对应声明权重下绝对倍数误差的几何平均，不是每个点的误差上限或置信区间。实验噪声协方差缺失，训练度量不解释为测得协方差。完整损失和导数见附录。

## 5. 固定共同查询上的载流子预测

以下均为同一common20、DOI宏平均nH MAE，单位dex。“未覆盖”表示完整查询资格不满足，保留失败，不用较容易的子集代替。

"""
    text+=table(ev,MAIN,'nH_MAE','common20')+'\n\n'
    text+='![载流子主终点；左为DOI宏平均，右为同一输出的关联组宏平均，缺口保留未覆盖。](figures_v1/carrier_common20.png)\n\n'
    text+='### 5.1 共享过程关系相对小容量对照的增量\n\n'
    text+='差值定义为主方法MAE减对照MAE；负值才表示该预算下误差较小。它不是显著性检验，也没有把3个重叠预算当作独立重复实验。\n\n'
    text+='| 对照 | 0锚点差值 | 1锚点差值 | 2锚点差值 |\n| --- | --- | --- | --- |\n'
    for control in ('ReferenceTau_internal','JointConstant_internal','ShareProcess205_internal',
            'DirectMultiRidge_offset','MedianActivation_offset','UnitDose_offset'):
        values=[]
        for b in (0,1,2):
            a=dict(variant=PRIMARY,budget=b,scope='common20');c=dict(variant=control,budget=b,scope='common20')
            values.append(ev.difference(OUT/'prediction/summary.csv',a,c,'nH_MAE'))
        text+='| '+control+' | '+' | '.join(values)+' |\n'
    text+='\n逐项描述性判读如下；1e-10只用于去除浮点方向噪声，不是实际意义或统计显著性阈值。\n\n'
    for i,control in enumerate(('ReferenceTau_internal','JointConstant_internal','ShareProcess205_internal',
            'DirectMultiRidge_offset','MedianActivation_offset','UnitDose_offset')):
        text+=control+'：'+ev.fact(REPORT/'descriptive_interpretation.json',['controls',i,'verdict'])+'\n\n'
    text+="""

只优于固定205不能单独证明过程特征有用：仍须比较只学散射和常数库存。只改善mu但使主要nH退化，也不能作为完整成功。共享参数训练、来源适应与输出常数校准的收益分别判读；全部48方法表附后。

### 5.2 逐DOI结果

逐来源表继续使用同一共同查询；它显示平均值是否由少数来源驱动，不提供新的独立样品或P值。

| 预算与DOI | 主方法 | 参考＋散射 | 中位数激活 |
| --- | --- | --- | --- |
"""
    per=read(OUT/'prediction/per_family.csv')
    families=sorted(per[per.scope.eq('common20')].family.unique())
    for b in (0,1,2):
        for family in families:
            values=[ev.csv(OUT/'prediction/per_family.csv',dict(variant=v,budget=b,scope='common20',family=family),'nH_MAE')
                for v in (PRIMARY,'ReferenceTau_internal','MedianActivation_offset')]
            text+='| '+str(b)+' / '+family+' | '+' | '.join(values)+' |\n'
    text+="""

## 6. 完整性质链：mu与派生sigma

下面的固定共同查询配对来自Si，使用模型预测的nH，不能称作“给定实测nH的条件迁移率”。新增Sn另列。mu和sigma都用DOI/条件/情景层级汇总，单位dex。

"""
    text+=table(ev,MAIN,'mu_MAE','common20')+'\n\n'
    text+=table(ev,MAIN,'sigma_MAE','common20')+'\n\n'
    text+='![主方法零锚点的分量有符号误差。小点是声明情景，菱形为有效条件均值，黑叉保留实现无效数值；虚线表示派生sigma误差抵消，情景不当作独立样品。](figures_v1/paired_component_errors.png)\n\n'
    text+="""
小的sigma误差可能来自nH与mu的反向偏差，不能替代两者的准确度。所有Hall锚点上的mu输出在Hall_anchor_conditional中另报；它们不混入从化学输入出发的完整查询。

### 6.1 新Sn条件的0/1锚点检验

这里的工艺/氧输入仍不完整，只能评价固定情景集合下的条件预测。2锚点时没有新Sn查询，不填补第三个点。以下列分别为nH、mu、派生sigma MAE。

"""
    for b in (0,1):
        text+='### '+str(b)+'个Hall锚点\n\n| 方法 | nH | muH | sigma |\n| --- | --- | --- | --- |\n'
        for v in MAIN:
            text+='| '+v+' | '+' | '.join(metric(ev,v,b,column,'newSn_query') for column in ('nH_MAE','mu_MAE','sigma_MAE'))+' |\n'
        text+='\n'
    text+=f"""## 7. 物理适用性与信息解释

实际输出/选中全来源模型中记录的可重建输运状态为{ev.fact(physical,['numeric_transport_states'])}行，独立输运矩最大读出差{ev.fact(physical,['independent_readout_maximum_error_dex'],'.3e')}dex。行数含不同方法、预算和情景的重复读出，不是独立晶片。唯一数值输运状态中，有{ev.fact(physical,['unique_numerical_transport_states_majority_kl_lt1'])}个状态的过半电导权重位于k*l小于1区域。

这提示当前半经典模型的适用风险，不是用该指标给材料做新的验证或删点。即使k*l大于1，也不证明Born、单带、独立点电荷和均匀体均成立。全来源已拟合输入回查不作为留出精度；有效Tf高于所列生长情景也不自动证明实际峰值历史违反物理。没有按诊断重新过滤主要评分。

![主全来源拟合状态的k*l诊断；颜色表示元素，点是输入情景，不是留出误差或置信区间。](figures_v1/transport_scope.png)

    信息审计共回查{ev.fact(information,['selected_unique_nonlinear_fits'])}个选中非线性拟合，不合格拟合仅记录资格、不伪造Jacobian。情景扩展的替代损失Jacobian与条件平均响应、情景变化约束分别记录；旧observation_jacobian_rank字段不能被解释为实测信息秩。正则或假设情景提高秩，不证明真实微观参数唯一。精确Gram分解和适用条件见附录。

## 8. 训练失败、计算预算与可调用交付

所选最优中未停止记录{ev.fact(training,['selected_nonstopped'])}个，实现训练状态无效记录{ev.fact(training,['invalid_training_fits'])}个。全部原始初始化、细化、边界、失败和最低有限损失保留；拒绝服务是事前资格规则，不是事后挑有利结果。

实际训练壁钟时间{ev.fact(statistics,['actual_training_wall_seconds'],'.1f')}秒，求解工作进程4个。训练审计记录实际物理点调用{ev.fact(OUT/'training_audit.json',['actual_physical_point_calls'])}次，闭式线性求解{ev.fact(training,['new_linear_solves'])}次；这些计算预算与材料样本数无关。本轮未做新DFT、MD或实体制备测量，GPU1授权保留而未使用GPU。

导出{ev.fact(OUT/'delivery/audit.json',['presets'])}个研究预设；其中不合格预设{ev.fact(OUT/'delivery/audit.json',['unqualified_presets'])}个保留拒绝。独立API回放请求拒绝{ev.fact(api,['request_refusals'])}次，按其与原记录的资格/失败是否一致核验，未补成有效数字。

全来源预设与关联组排除预设分开，调用只读取保存权重、允许Hall锚点及物理输入，不读查询/训练标签表。源码和模型绑定哈希；不复制第三方KROGER原始数值。API支持Si/Sn300K和明确输入情景，未验证稀土、新元素或绝对寿命/PDR。详细调用及按元素/上下文的支持范围见delivery。

## 9. 如何据此判断贡献

判断顺序是：先检查覆盖和失败，再看固定common20载流子主终点，再看小容量/常数校准/直接ML控制，最后看完整性质链及物理适用性。一个预算或来源的下降不能替代跨来源一致证据；软件、守恒、正速率或可微本身也不证明新增预测贡献。

本队列规模与独立性有限，所有来源已被多轮研发查看。本文未预设优于基线、未把不利结果移出比较，也不从同一队列生成独立盲测或“审稿保证”。下一项模型改进应对应本轮实证瓶颈：若仅全局散射校准有效，不能包装成过程特征创新；若主要问题是未报告组成/边界或超出输运范围，应补相容输入或新的可检验闭合，而不是凭空增加参数。

下文保留全部方法的固定查询指标及完整数学说明。总体goal状态仍须由完整要求审计判定，本报告不自动将其标成完成。

## 10. 全48方法：共同载流子查询

"""
    text+=table(ev,variants,'nH_MAE','common20')+'\n\n'
    text+='## 11. 全48方法：共同配对迁移率查询\n\n所有配对共用模型预测nH；这里的固定查询没有新增Sn，不能误称跨元素全预算迁移率验证。\n\n'
    text+=table(ev,variants,'mu_MAE','common20')+'\n\n'
    text+='## 12. 全48方法：共同配对派生电导\n\nsigma不是第三个独立标签；较低误差可能含分量抵消。\n\n'
    text+=table(ev,variants,'sigma_MAE','common20')+'\n\n'
    text+='## 附录A. 完整前向模型、符号、守恒与隐式导数\n\n'
    text+='\n'.join(THEORY[0].read_text().splitlines()[2:])+'\n\n'
    text+='## 附录B. 配对损失、条件权重与完整训练梯度\n\n'
    loss=THEORY[1].read_text();loss='## 2.'+loss.split('## 2.',1)[1]
    loss=loss.split('## 5.',1)[0];text+=loss+'\n\n'
    text+='## 附录C. 正散射尺度的完整偏导\n\n这里M2均指含正背景的总预算；下述固定状态偏导不改变库存或背景。\n\n'
    derivative=THEORY[5].read_text().split('## 3.',1)[1].split('## 4.',1)[0]
    derivative=derivative[derivative.index('在固定'):]
    text+=derivative+'\n\n'
    text+='## 附录D. 情景扩展与信息秩的精确分解\n\n'
    text+='\n'.join(THEORY[2].read_text().splitlines()[2:])+'\n\n'
    text+='## 附录E. 输运适用风险及阈值积分\n\n'
    text+='\n'.join(THEORY[3].read_text().splitlines()[2:])+'\n\n'
    text+='## 附录F. 背景散射记号更正与证据边界\n\n'
    text+='\n'.join(THEORY[4].read_text().splitlines()[2:])+'\n\n'
    text+="""
## 13. 复现与逐数证据索引

数据、角色与情景见data；模型、全部初始化/细化见fits和training_protocol；来源选择、全部输出及逐来源指标见prediction。独立审计、无标签API回放和物理/信息审计分别保留原始请求、结果与绑定代码。

report_v1/numeric_claims.json将动态指标绑定到文件哈希、CSV筛选/列或JSON键、显示值及正文字符区间；numeric_claims_audit.json逐项独立回查。数学常数和模型定义不是测量数据，理论文件另绑定原始哈希。figures_v1提供独立PNG/PDF科学图，figure_evidence.json记录输入和输出哈希。

当前合成/数值门槛、实际材料留出、全来源已拟合输入回查、接口一致性和物理近似各有不同证据层级，不相互替代。历史错误、非停止拟合、无效状态、不可测条件和未覆盖结果保留。没有改写npj稿件或重新宣称旧宽泛九性质精度。
"""
    return text


if __name__=='__main__':main()
