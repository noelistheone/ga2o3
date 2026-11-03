"""Trace chemical learning, coupled property prediction and the next shared-inventory hypothesis."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv as read
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';PRED=OUT/'property_prediction';OLD=BASE/'phase23_process_inputs/source_prediction'
CLAIMS=[];CACHE={};METHODS=('ArrheniusMassBalance','MonotoneFlux','RegimeFlux','DirectProcessSpline')


def value(path,selector,column,fmt='.6f'):
    if path not in CACHE:CACHE[path]=read(path)
    d=CACHE[path]
    for k,v in selector.items():d=d[d[k].eq(v)]
    assert len(d)==1,(path,selector,len(d));x=float(d[column].iloc[0]);finite=np.isfinite(x)
    CLAIMS.append(dict(path=str(path.relative_to(ROOT)),sha256=e.sha(path),selector=selector,column=column,value=x if finite else None))
    return format(x,fmt) if finite else '未完整覆盖'


def property_value(context,method,budget,cal='MonotoneFlux',scenario='central50',prop='nH'):
    if cal=='Printed':return value(OLD/'summary.csv',dict(context=context,method=method,budget=budget,scope='common_query',tier='process_calibrated_12',scenario='regime_rates'),prop+'_MAE')
    return value(PRED/'summary.csv',dict(context=context,method=method,budget=budget,scope='common_query',chemical_model=scenario+'::'+cal),prop+'_MAE')


def property_table(context,methods,prop='nH'):
    rows=['| 性质方法 | 0 Hall锚点 | 1 Hall锚点 | 2 Hall锚点 |','| --- | --- | --- | --- |']
    for m in methods:rows.append('| '+m+' | '+' | '.join(property_value(context,m,b,prop=prop) for b in (0,1,2))+' |')
    return '\n'.join(rows)


def auxiliary_tables():
    bundle=OUT/'delivery/chemical_bundle';manifest=json.loads((bundle/'manifest.json').read_text());rows=[]
    for name,path in manifest['models'].items():
        r=json.loads((bundle/path).read_text());ns=1 if r['method']=='ArrheniusMassBalance' else 5;slopes=np.array(r['coefficients'][1:1+ns])*np.log(10.)
        rate=(r['coefficients'][-1] if r['method'] in ('RegimeFlux','DirectProcessSpline') else 0.)+(0. if r['method']=='DirectProcessSpline' else -1.)
        rows.append(dict(chemical_model=name,scenario=r['chemical_scenario'],method=r['method'],parameters=len(r['coefficients']),selected_lambda=r['lambda_'],
            effective_E_min_eV=float(slopes.min()),effective_E_max_eV=float(slopes.max()),logC_rate_elasticity=rate))
    e.csvwrite(OUT/'chemical_learning/model_diagnostics.csv',rows)
    states=read(OUT/'inventory_scope/states.csv');rows=[]
    for r in states[states.source_shift.eq(0)].itertuples():
        q=states[states.element.eq(r.element)&states.incorporated_cm3.eq(r.incorporated_cm3)&states.source_shift.eq(6)].iloc[0]
        rows.append(dict(element=r.element,incorporated_cm3=r.incorporated_cm3,vacancy_over_C=r.modeled_reference_vacancy_equivalent_cm3/r.incorporated_cm3,
            vacancy_dose_elasticity=r.reference_vacancy_dose_elasticity,native_negative_fraction=r.native_negative_fraction,complex_negative_fraction=r.vacancy_complex_negative_fraction,
            sameGa_negative_fraction=r.sameGa_foreign_negative_fraction,lognH_zero=r.log10_nH,lognH_shift6=q.log10_nH))
    e.csvwrite(OUT/'inventory_scope/summary.csv',rows)
    s=read(PRED/'summary.csv');rows=[]
    for key,g in s.groupby(['context','method','calibration_method','budget','scope']):
        for prop in ('nH','mu','sigma'):
            complete=bool((g.numeric==g.queries).all());values=g[prop+'_MAE'].to_numpy(float)
            rows.append(dict(context=key[0],method=key[1],calibration_method=key[2],budget=key[3],scope=key[4],property=prop,
                all_scenarios_complete=complete,minimum=float(values.min()) if complete else None,maximum=float(values.max()) if complete else None))
    e.csvwrite(PRED/'audit/reading_sensitivity.csv',rows)


def figures():
    folder=OUT/'figures';folder.mkdir(parents=True,exist_ok=True);pred=read(OUT/'chemical_learning/predictions.csv');layers=read(OUT/'extraction/layer_readings.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4));colors=['#2864a0','#ce8700','#248c55','#8a49a3']
    for ax,stack in zip(axes,['m','n']):
        a=layers[layers.scenario.eq('central50')&layers.stack_id.eq(stack)].sort_values('Tsource_nominal_C')
        ax.errorbar(a.Tsource_nominal_C,a.log10_C_chemical,yerr=np.vstack([a.log10_C_chemical-a.log10_reading_lower,a.log10_reading_upper-a.log10_C_chemical]),fmt='o',color='black',ms=4,label='SIMS interior reading')
        for method,color in zip(METHODS,colors):
            q=pred[pred.method.eq(method)&pred.scenario.eq('central50')&pred.held_stack.eq(stack)].sort_values('Tsource_nominal_C')
            ax.plot(q.Tsource_nominal_C,q.predicted_log10_C,'o-',color=color,ms=3,lw=1.2,label=method)
        ax.set(xlabel='Nominal source setting (C)',ylabel=r'$\log_{10} C_{\rm chemical}\,[{\rm cm}^{-3}]$',title=f'Held-out stack {stack}');ax.grid(alpha=.2)
    axes[0].legend(fontsize=6.5);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'chemical_transfer.{ext}',dpi=180)
    plt.close(fig)
    s=read(PRED/'summary.csv');old=read(OLD/'summary.csv');names=['OriginalHall','FieldAnchorSafe','InventoryEnsemble','UnitDose','MedianActivation','DirectRidge']
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for ax,prop in zip(axes,['nH','mu']):
        for j,(isnew,b,color) in enumerate([(False,0,'#b1b1b1'),(True,0,'#2864a0'),(False,2,'#e6bd6e'),(True,2,'#b46600')]):
            vals=[]
            for method in names:
                q=s[s.context.eq('full5')&s.method.eq(method)&s.chemical_model.eq('central50::MonotoneFlux')&s.budget.eq(b)&s.scope.eq('common_query')] if isnew else old[
                    old.context.eq('full5')&old.method.eq(method)&old.tier.eq('process_calibrated_12')&old.scenario.eq('regime_rates')&old.budget.eq(b)&old.scope.eq('common_query')]
                vals.append(float(q[prop+'_MAE'].iloc[0]))
            ax.bar(np.arange(len(names))+(j-1.5)*.18,vals,width=.17,color=color,label=('Learned chemistry' if isnew else 'Printed proxy')+f', {b} anchors')
        ax.set_xticks(np.arange(len(names)),names,rotation=35,ha='right',fontsize=7);ax.set(ylabel='Common-ten MAE (dex)',title='Carrier density' if prop=='nH' else 'Hall mobility');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=6.5);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'property_input_comparison.{ext}',dpi=180)
    plt.close(fig)
    e.write(folder/'provenance.json',dict(chemical_predictions_sha256=e.sha(OUT/'chemical_learning/predictions.csv'),property_summary_sha256=e.sha(PRED/'summary.csv'),
        prior_summary_sha256=e.sha(OLD/'summary.csv'),chemical_model_shown='central50::MonotoneFlux',all_four_family_results_in_tables_and_CSV=True))


def main():
    audit=json.loads((PRED/'audit/contract_audit.json').read_text());chem=json.loads((OUT/'chemical_learning/audit.json').read_text());grad=json.loads((OUT/'process_gradients/gates.json').read_text())
    paired=json.loads((OUT/'paired_property_prediction/audit.json').read_text());pairchem=json.loads((OUT/'paired_validation/audit.json').read_text())
    assert audit['passed'] and chem['passed'] and grad['passed'] and paired['passed'] and pairchem['passed'];auxiliary_tables();figures()
    p=json.loads((PRED/'protocol.json').read_text());full=sorted(c['method'] for c in p['models'] if c['context']=='full5');excluded=sorted(c['method'] for c in p['models'] if c['context']=='exclude_related_McCandless')
    blocks=['# ML与Sandbox：化学过程学习、可微耦合与完整性质预测',r'''## 1. 方向、完成范围与判断方式

目标仍是训练并验证 ML＋Sandbox 对掺杂性质的预测。理论和化学监督用于改进真正的载流子、迁移率和电导读出，不以接口或解释完成作为整体goal完成。系统goal继续active。

本轮实际训练共享的过程到化学输入关系：名义源炉温度/生长速度 → 化学浓度代理 → 已保存物理或经验预测器。首先4类学习器、3固定读取窗，共12个保存参数；1188次小规模凸拟合、36选择上下文、144整堆栈留出化学预测。随后发现整堆栈任务同时删除唯一速度，补充独立冻结的已知两条件配对层验证、简单三参数对照及3个化学选择部署预设。两组性质请求共32940，全部保留。下游物理共享权重及来源先验全部冻结；没有新Hall训练、独立sigma标签或实体/DFT/MD实验。

必须分别回答：化学预测是否有收益；该收益是否超过两参数基线；相同化学信息给所有方法后，性质预测是否改善；改进能否跨相关来源排除上下文保持。本阶段仅有一个化学DOI/两片堆栈，不能宣称跨实验室过程泛化或新的微观掺入机制。''',
r'''## 2. 新化学监督来自哪里，图文冲突如何处理

[Azizie2023原作者PDF](https://djena.engineering.cornell.edu/papers-new/2023/azizie2023silicon.pdf)第7页Fig3有m/n两片SIMS堆栈，各6个名义源温层。源温来自目标示意图；尚未与仪器日志逐片配对。化学读数来自独立SIMS曲线，与Hall标签分开。一个层的多个像素不是独立样品，12层仍为2片/1DOI。

从原PDF提取1649×1427原图，按固定坐标、金色Si线、每列中位位置读取。主窗为已声明层内部中央50%，另25%/75%为敏感性，不按Hall误差挑窗。C为等间距深度列浓度的算术均值，再取log；线宽和坐标±1像素包络不当作实验置信区间。36个窗的像素覆盖均100%。

按图中层顺序，低名义浓度层的曲线内部值高于示意图目标，和正文“低于目标”叙述方向相反。原图、目标、正文全部保留，不推定作者代码错误，也不借Hall决定改哪一个。下面是主窗的完整12条件。''']
    rows=['| 堆栈/名义源温 | 示意目标cm^-3 | SIMS内部读数cm^-3 | 图读数/目标 |','| --- | --- | --- | --- |']
    for stack,T in [('m',T) for T in [1282.3,1237.1,1194.7,1152.9,1116.6,1080.4]]+[('n',T) for T in [1285.4,1238.1,1193.7,1150.3,1112.5,1074.9]]:
        sel=dict(stack_id=stack,Tsource_nominal_C=T,scenario='central50');f=OUT/'extraction/layer_readings.csv'
        rows.append('| '+f'{stack}/{T:.1f}C'+' | '+value(f,sel,'nominal_target_cm3','.3e')+' | '+value(f,sel,'C_chemical_cm3','.3e')+' | '+value(f,sel,'observed_to_target_ratio','.3f')+' |')
    blocks.append('\n'.join(rows));blocks.append(r'''Fig2印刷5.3eV通量线由全部SIMS和额外XRR共同拟合，包含被留出化学数据。因此只作既有参考，不在化学留出检验中假称同信息独立训练基线；本轮学习器没有使用它的系数或能量先验。''')
    blocks.append(r'''## 3. 共享关系、约束与真正的留出协议

设y=log10[C/(1cm^-3)]，q=1/(kB·1500K)−1/[kB(Ts+273.15)]，r=log10[v/(1μm/h)]。Ts是摄氏源温，kB=8.617333262145e-5eV/K。此处q为倒温度校准坐标，不是缺陷电荷。两参数ArrheniusMassBalance为

\[
y=a+bq-r.
\]

MonotoneFlux把bq改为5段积分基函数之和，固定结点为源温1100/1150/1200/1250C；每段斜率b_j≥0，保证固定速度下C随源温不减。RegimeFlux再加d(r−训练均值r)，允许有效掺入通量随速率/相关工艺改变。DirectProcessSpline采用相同样条与速率输入，放开斜率及速率系数、去掉-r先验。所有方法都在log浓度学习，保证C>0。

约束斜率上限15/ln10，对应15eV只是预先声明的数值搜索界，不是第一性原理物理定理。目标为化学训练均方误差加相邻斜率差及可选速率系数的L2项；9个lambda从1e-5到1e3。超参数按MAE选，平手偏向更强正则化。

外层整片堆栈留出2折，每折6层；内层只在另外6层中留一层选参，不随机拆像素。训练只有一个速率时，中心化速率列为0，系数明确固定0，未从数据识别新速率效应。物理模型继承-r约定，直接模型继承零速率效应，这一归纳先验差异须明确。最终12层拟合的lambda仅由化学堆栈CV选择；无Hall查询进入。

化学表中的误差是当前图读数条件下的交叉堆栈开发结果，不是新化学测量精度或独立实验室证据。''')
    rows=['| 化学方法 | 留出m MAE | 留出n MAE | 两堆栈均值 |','| --- | --- | --- | --- |']
    for m in METHODS:rows.append('| '+m+' | '+' | '.join(value(OUT/'chemical_learning/summary.csv',dict(method=m,scenario='central50'),c) for c in ['stack_m_MAE','stack_n_MAE','nested_LOSO_MAE'])+' |')
    blocks.append('\n'.join(rows));blocks.append(r'''主窗单调/工艺模型约0.054dex优于自由回归约0.125dex，但两参数约0.0538dex已经相当。主窗复杂模型选择lambda=1000，额外工艺系数约2.41e-6；该任务不能识别未知速度效应，不能据此判定真实工艺无价值。主窗有限SIMS范围的有效斜率约4.83465eV，也不能替换作者由全XRR/SIMS所得5.3eV或称新发现。

![整堆栈留出化学预测；像素包络不是实验置信区间。](figures/chemical_transfer.png)''')
    blocks.append(r'''### 验证目标复核：已知生长条件与未见生长条件必须分开

整堆栈留出同时去掉唯一生长速度，新增速率列在训练中为0；RegimeFlux与MonotoneFlux外层相同是结构必然。共同lambda按该任务选择，不能用其full12系数缩零作为独立工艺贡献判断。

实际Hall过程预测的0.86/1.3um/h两条件已在化学训练中出现，因此另外冻结“留出相同名义目标层级的m/n层对”任务。外层6对、每次2测试层，内层只用另5对选择族与lambda；训练始终保留两速率。加入RateAffine三参数对照，不允许把工艺相关项收益归给非线性样条。2346次额外凸拟合、216外层记录，含独立的训练内选族策略；原全部模型与26352预测不变。

两种任务的训练条件与问题不同，不把下列两列当同一难度。所有结果仍只来自同DOI/两片堆栈，生长速度又与Ga2O通量/生长区间绑定，不能认定是速度单独的因果效应。''')
    rows=['| 化学方法/策略 | 未见条件：堆栈留出 | 已知两条件：层对留出 | full12训练内选择CV |','| --- | --- | --- | --- |']
    for family in [*METHODS,'RateAffine','SelectedPolicy']:
        previous=value(OUT/'chemical_learning/summary.csv',dict(method=family,scenario='central50'),'nested_LOSO_MAE') if family in METHODS else '未执行'
        sel=dict(family=family,scenario='central50');rows.append('| '+family+' | '+previous+' | '+value(OUT/'paired_validation/summary.csv',sel,'nested_pair_MAE')+' | '+value(OUT/'paired_validation/summary.csv',sel,'full_selection_CV_MAE')+' |')
    blocks.append('\n'.join(rows));blocks.append(r'''已知两条件下，三参数约0.01666dex优于两参数0.02687；嵌套选族策略外层约0.01837，不能用最终选择CV0.01559代替该验证误差。额外条件项在此任务有信号，但更复杂样条没有独有优势。三固定读取窗的最终选择分别为RegimeFlux/RateAffine/RateAffine，完整敏感性保留。

RateAffine通过7×3系结矩阵写入兼容的5段系数，5个斜率严格相同，独立学习参数仍是3。新增process_parameter_gradients.py在原IFT链后乘系结矩阵；6个独立扰动案例通过，最大缩放差1.45e-9，避免把7个存储系数误当7个独立学习参数。程序固定采用eV作为能量单位，正则化作用于b/(1eV)和速率系数的数值；换单位必须同步变换正则化，不是坐标不变物理先验。''')
    blocks.append(r'''## 4. 接入Sandbox与真正连通的解析梯度

ProcessPropertyPredictor从名义源温与速度调用保存化学模型，再以相同C调用冻结的物理或经验性质模型，得到nH、muH及sigma=e*nH*muH。它只通过C连接过程层，没有在本轮学习工艺对缺陷分配的额外直接作用，也没有新联合Hall训练。

新增独立process_property_gradients.py连接了零Hall锚点下的完整一阶链。令c=log10 C，g=(log10 nH,log10 muH,log10 sigma)，theta为连续化学学习系数，x=(Ts,log10v)，则

\[
\frac{\partial g}{\partial x}=\frac{\partial g}{\partial c}\frac{\partial c}{\partial x},\qquad
\frac{\partial g}{\partial\theta}=\frac{\partial g}{\partial c}\frac{\partial c}{\partial\theta}.
\]

物理因子使用实际热库存、电荷中性IFT及输运输入雅可比；并非把有限差分冒充反传。3种化学模型×2个物理/输出预设×3个工艺点共18例，37项独立扰动检查通过，最大缩放差3.13e-9。化学层凸KKT、梯度、选参及预测另有14786检查，最大化学导数差1.72e-10。

非零锚点涉及来源重拟合，有限集合中位数还有决策不光滑性；本梯度接口明确拒绝这些分支。结点处源温导数采用右侧值；未对CV/拟合映射求导。完整0/1/2性质预测仍执行原来源适应，但不能将其预测成功说成完整适应后导数已实现。''')
    rows=['| 主化学关系下的OriginalHall | dlognH/dlogC | dlogmuH/dlogC | dlogsigma/dlogC |','| --- | --- | --- | --- |']
    for T in (1115.,1187.,1319.):
        sel=dict(chemical_model='central50::MonotoneFlux',method='OriginalHall',source_T_C=T)
        rows.append('| '+str(T)+'C | '+' | '.join(value(OUT/'process_gradients/sensitivities.csv',sel,c) for c in ['lognH_dlogC','logmu_dlogC','logsigma_dlogC'])+' |')
    blocks.append('\n'.join(rows));blocks.append(r'''高浓度点的nH对C响应仅约0.10，而muH响应约-0.62，故输入变准不自动保证当前性质模型变准。数值可导、概率归一化、守恒和RTA/Born模型内部一致性都不能单独证明真实材料准确率。''')
    blocks.append('## 5. 完整性质协议与覆盖\n\n4化学类×3读取窗×61个旧上下文方法×3锚点预算×12片，共26352请求；实际数值'+str(audit['numeric'])+'，拒绝'+str(audit['refusals'])+'，全部保留。4392行完整分层汇总。full5为34方法，排除相关McCandless为27，未伪造后者7个能量包。旧物理权重、来源先验和数值预算上限不变；不同方法实际调用数与能量信息成本不声称相等。\n\n0/1/2指Hall锚点，每种方法都得到同样的12层新来源化学监督。锚点固定d/c，所有预算共同查询为排除两片后的10片，与第23阶段完全相同。a/j相同报告输入仍同预测；不能通过新化学函数区别它们或消除其误差下界。实际气氛为O3，模型仍用未校准的旧有效O2情景；525C/300K输入及缺失历史限定全部保留。\n\n以下预先固定展示主读取窗MonotoneFlux全性质方法，未按Hall结果选最佳化学类；其余所有化学类及读取窗均在后表/CSV报告。')
    blocks.append('### 完整旧5来源：共同10片nH MAE（dex）\n\n'+property_table('full5',full))
    blocks.append('### 排除相关来源：共同10片nH MAE（dex）\n\n'+property_table('exclude_related_McCandless',excluded))
    rows=['| 性质方法／化学输入 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for m in ['OriginalHall','FieldAnchorSafe','InventoryEnsemble','UnitDose','DirectRidge']:
        for cal in ['Printed',*METHODS]:rows.append('| '+m+'/'+cal+' | '+' | '.join(property_value('full5',m,b,cal=cal) for b in (0,1,2))+' |')
    blocks.append('## 6. 全四类化学学习器与原印刷代理：性质增量\n\n下表固定同一10片、完整旧来源上下文，不能用化学CV误差替代这些nH结果，也不按最有利化学模型选择主结论。\n\n'+'\n'.join(rows))
    blocks.append(r'''原物理模型主窗MAE从印刷代理的0.617816降为0.591625，改善有限；库存集合两锚点从0.545530降至0.499210，但简单MedianActivation此时为0.341795。FieldAnchorSafe两锚点0.357150也未低于该简单对照。化学输入改善不等于共享缺陷学习已经解决。

原始化学验证任务较差的DirectProcessSpline，在某些下游方法下也会略好，进一步说明化学CV误差和性质误差不能互相替代。不能用已看过的Hall结果反选化学模型。''')
    selected=['OriginalHall','FieldAnchorSafe','InventoryEnsemble','SharedReadout','UnitDose','MedianActivation','DirectRidge']
    blocks.append('## 7. 同次迁移率和派生电导\n\n共同10片muH MAE（dex），相同主化学模型。\n\n### full5\n\n'+property_table('full5',selected,prop='mu'))
    blocks.append('### 排除相关来源\n\n'+property_table('exclude_related_McCandless',selected,prop='mu'))
    blocks.append(r'''sigma逐请求由同次nH与muH派生，没有新增独立sigma实验标签。其误差可能因两个分量相消而下降，不能当第三个独立验证成功。图中展示旧印刷代理与固定主化学学习器的0/2锚点结果，全部其他预算和方法仍保留。

![化学输入学习后的载流子与迁移率预测；相同共同查询与Hall预算。](figures/property_input_comparison.png)''')
    blocks.append('### 已知两条件、只用化学CV选择后的6588次补充性质预测\n\n仍为相同34/27性质方法、d/c锚点及共同10片。额外实际数值'+str(paired['numeric'])+'、拒绝'+str(paired['refusals'])+'；1098分层汇总。原26352结果不改写。下表为主读取窗所选RateAffine，不能将这项条件选择称为独立跨来源过程泛化。')
    for context in ('full5','exclude_related_McCandless'):
        for prop in ('nH','mu'):
            rows=['| 性质方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
            for method in selected:
                cells=[value(OUT/'paired_property_prediction/summary.csv',dict(context=context,method=method,scenario='central50',budget=b,scope='common_query'),prop+'_MAE') for b in (0,1,2)]
                rows.append('| '+method+' | '+' | '.join(cells)+' |')
            blocks.append('### '+context+' / '+prop+' MAE（dex）\n\n'+'\n'.join(rows))
    blocks.append(r'''配对选择的full5主窗下，OriginalHall仍为0.593217dex，库存集合两锚点0.507289，而MedianActivation两锚点0.347842。化学关系的条件预测改善没有修复当前性质缺口；不能以后者结果反选化学族。

部分低剂量Hall锚点高于这套过程浓度估计，也提醒我们：堆栈内部化学标定不等于每片Hall薄膜的实际掺入量或背景库存。这里只保留输入情景与预测，未用Hall反推化学量当新真值。''')
    rows=['| 性质方法／化学类 | 0锚点三窗范围 | 1锚点三窗范围 | 2锚点三窗范围 |','| --- | --- | --- | --- |']
    for m in ['OriginalHall','InventoryEnsemble','UnitDose','DirectRidge']:
        for cal in METHODS:
            cols=[]
            for b in (0,1,2):
                sel=dict(context='full5',method=m,calibration_method=cal,budget=b,scope='common_query',property='nH')
                cols.append(value(PRED/'audit/reading_sensitivity.csv',sel,'minimum')+'–'+value(PRED/'audit/reading_sensitivity.csv',sel,'maximum'))
            rows.append('| '+m+'/'+cal+' | '+' | '.join(cols)+' |')
    blocks.append('## 8. 全部固定读取窗的敏感性\n\n范围是25%/50%/75%三种读取规则下的条件MAE，不是置信区间。任何窗有拒绝即不计算完整范围，不按完整样本子集刷分。\n\n'+'\n'.join(rows))
    blocks.append(r'''## 9. 预测缺口的内部状态审计与下一项真实学习

36个冻结Si/Sn剂量/来源场状态、2178母类行的审计保留24个电子Hall有效状态及12个负向极端延续；后者不当材料预测。起初怀疑未被原学习坐标覆盖的同Ga结构池主导补偿，但实际负电荷占比约1e-8–1e-6，不支持该猜测；不能据此提出虚假的新机制。183与169同(Si:+1,Ga:-1)，184为(Si:+1,Ga:-2)，并非两个Si的复合物。

当前负电荷主要为原生Ga空位与额外Ga亏损的掺杂复合物。高剂量下参考正空位等效库存接近C/3，对C的log弹性接近1。对主导的纯Ga替换/空位母类，i为母类，c_i为其浓度，k_i为外来原子数，nu_i为正Ga亏损等效计数（不是所有结构空位总数），q_i为以元电荷计的电荷态；它与前面的校准坐标q不同。忽略其他次要电荷且q_i≈k_i−3nu_i时，

\[
C=\sum_i k_i c_i,\quad N_V=\sum_i\nu_i c_i,\quad
\sum_iq_ic_i\simeq C-3N_V,\quad n\simeq C+N_{\rm bg}-3N_V.
\]

n是自由电子浓度，Nbg为有效正背景，Hall浓度还需nH=n/rH。该近似与模型补偿平台一致，只解释当前状态，不证明真实样品按此机制演化。来源正方向场能提高nH，故也不能说该模型任何参数都无法达到高载流子浓度。''')
    rows=['| 元素／输入C | 参考Nv/C | Nv对C的log弹性 | 正向源场6的lognH |','| --- | --- | --- | --- |']
    for element in ('Si','Sn'):
        for C in (1e17,1e19,3e20):
            sel=dict(element=element,incorporated_cm3=C);f=OUT/'inventory_scope/summary.csv'
            rows.append('| '+element+'/'+format(C,'.0e')+' | '+' | '.join(value(f,sel,c) for c in ['vacancy_over_C','vacancy_dose_elasticity','lognH_shift6'])+' |')
    blocks.append('\n'.join(rows));blocks.append(r'''下一轮应依据真实高掺杂化学/Hall条件检验共享内部关系，而非继续只加过程校准或来源先验。现有Leedy2018的4个均匀条件可作为新33行任务的候选，先核对化学配对、标签冲突、实际氧压和关联来源；其非均匀第五条件与全部旧结果保留。新33行数据契约及重新训练尚未实施，不能写成已验证增量。

候选h_theta(C,u)进入守恒库存分配，须与原常数场、同容量物理输出和直接ML在相同新增数据/分组/预算下比较。若h依赖C，新增链式项必须进入物理输入梯度。更改Ga库存约束前也须先写出原子/边界账，不能因可截断就叫第一性原理。具体方案见docs/ML_SANDBOX_HIGH_DOSE_SHARED_LEARNING_NEXT_20261018.md。

本轮实际能力是已训练过程层、保存部署模型、限定范围内可核验的端到端梯度及完整性质检验；不能因这些能力成立就宣称额外ML自由度、跨实验室泛化或材料创新已经成立。第23阶段粗化学e/k测试不变，过程代理不能重定义那两个独立化学输入来改善分数。''')
    blocks.append('## 10. 可复现交付与严谨性边界\n\n原性质协议独立核验'+str(audit['checks'])+'项，配对选择性质核验'+str(paired['checks'])+'项；原化学14786、配对化学'+str(pairchem['checks'])+'、零锚点梯度37项和6个系结案例，父清单43717项未变。它们验证执行与模型数学条件，不替代材料验证。独立进程禁止读取化学训练表、Hall目标表、fit和查询参考，仅读取保存参数、当前工艺输入及允许的Hall锚点。\n\nchemical_process.py、process_property_predictor.py、process_property_gradients.py和process_parameter_gradients.py提供真实可微链；delivery的两个化学包分别保存原12预设与配对选择3预设。CLI及完整批量请求对照另存。全部表格值有路径、筛选和SHA；完整优化/候选/外层/拒绝记录保留。首次化学审计脚本语法失败已归档，修正只影响审计表达式，不改数值模型。初稿未充分区分验证任务的解释也保留在report_before_pair_design_review；修订后不将不同条件的验证混成一个结论。\n\n原29/5、全部旧强对照、冻结结果和失败不改写。当前阶段CPU计算，GPU1授权保留。整体goal仍active；尚未证明稳定跨来源内部学习增量，不能把过程输入学习等同于已完成掺杂性质方法研发。')
    report=OUT/'ML与Sandbox_化学过程学习与完整性质预测.md';report.write_text('\n\n'.join(blocks)+'\n')
    e.write(OUT/'report_claims.json',dict(report_sha256=e.sha(report),numeric_table_claims=len(CLAIMS),claims=CLAIMS));print('phase24 report',len(CLAIMS),'traceable values',flush=True)


if __name__=='__main__':main()
