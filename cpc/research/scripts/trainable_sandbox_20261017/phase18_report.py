"""Report trained log-odds predictors, failures, full-chain scope and next action."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv as read
import phase18_audit as audit
import phase2_energy as energy
study=audit.study;OUT=study.OUT;BASE=energy.BASE;CLAIMS=[]
METHODS=list(study.STRUCTURES)+list(audit.source.VARIANTS)


def path_for(method):return OUT/'validation/primary_summary.csv' if method in study.STRUCTURES else audit.source.OUT/'validation/primary_summary.csv'


def label(method):
    prefix='内部来源更新＋' if method.startswith('SourceLogit_') else ''
    m=method.removeprefix('SourceLogit_')
    return prefix+(m.replace('Logit_inventory_','库存／').replace('Logit_output_','输出／').replace('Logit_add_','直接add／')
        .replace('Logit_product_','直接product／').replace('original_RT','原RT').replace('binding_joint','E(C)').replace('joint','联合'))


def value(path,selector,column='MAE'):
    d=read(path)
    for k,v in selector.items():d=d[d[k].eq(v)]
    assert len(d)==1,(path,selector)
    v=float(d[column].iloc[0]);CLAIMS.append(dict(path=str(path.relative_to(ROOT)),selectors=selector,column=column,value=v))
    return v


def table(methods,path=None,scope='all_query',kind=None):
    rows=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for m in methods:
        scores=[]
        for b in (0,1,2):
            selector=dict(method=m,budget=b,scope=scope)
            if kind:selector['kind']=kind
            scores.append(value(path if path is not None else path_for(m),selector))
        rows.append('| '+label(m)+' | '+' | '.join(f'{s:.6f}' for s in scores)+' |')
    return '\n'.join(rows)+'\n'


def figures():
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    n=pd.concat([read(OUT/'validation/primary_summary.csv'),read(audit.source.OUT/'validation/primary_summary.csv')],ignore_index=True)
    old=read(BASE/'phase15_coherent_hall/validation/primary_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    methods=[('Logit_inventory_original_RT','Inventory / output adaptation'),('SourceLogit_Logit_inventory_original_RT','Inventory / log-odds adaptation'),
        ('Logit_add_joint','Direct joint / output adaptation'),('SourceLogit_Logit_add_joint','Direct joint / log-odds adaptation')]
    for ax,scope in zip(axes,('all_query','common18')):
        for m,l in methods:
            d=n[n.method.eq(m)&n.scope.eq(scope)].sort_values('budget');ax.plot(d.budget,d.MAE,'o-',label=l,ms=4)
        d=old[old.method.eq('original_RT')&old.scope.eq(scope)].sort_values('budget');ax.plot(d.budget,d.MAE,'s--',color='black',label='Retained original model',ms=4)
        ax.set(xticks=[0,1,2],xlabel='RT specimen anchors per source',ylabel='Source-equal nH MAE (dex)',title='29 / 23 / 18 queries' if scope=='all_query' else 'Same common18 specimens')
        ax.grid(alpha=.2);ax.legend(fontsize=6.5)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'carrier_and_adaptation.{ext}',dpi=170)
    plt.close(fig)
    diag=read(OUT/'relation_training_diagnostic.csv');fig,axes=plt.subplots(1,2,figsize=(10.8,3.7))
    for ax,basis in zip(axes,('original_RT','binding_joint')):
        d=diag[diag.basis.eq(basis)&diag.old_relation.eq('dose_minus')].sort_values('train_families')
        ax.bar(range(len(d)),d.difference,color=['tab:blue' if x<0 else 'tab:orange' for x in d.difference])
        ax.axhline(0,color='black',lw=.7);ax.set(title=basis,xlabel='15 fixed training-source contexts',ylabel='New minus old training loss (dex squared)')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'training_diagnostic.{ext}',dpi=170)
    plt.close(fig)
    scope=read(OUT/'property_evaluation/transport_scope.csv');mu=read(OUT/'property_evaluation/summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.9))
    for m in study.STRUCTURES:
        x=n[n.method.eq(m)&n.scope.eq('all_query')&n.budget.eq(0)].MAE.iloc[0]
        y=mu[mu.method.eq(m)&mu.scope.eq('all_query')&mu.kind.eq('RT')&mu.budget.eq(0)].MAE.iloc[0]
        axes[0].scatter(x,y,s=25,label=m.replace('Logit_','').replace('_binding_joint',' E(C)').replace('_original_RT',' RT'))
    axes[0].set(xlabel='nH MAE,29 specimens /5 DOI (dex)',ylabel='muH MAE,5 specimens /2 DOI (dex)',title='Different cohorts; retain the trade-off');axes[0].grid(alpha=.2)
    axes[0].legend(loc='upper right',fontsize=6.2)
    axes[1].hist(scope.fraction_sigma_kell_lt1,bins=np.linspace(0,1,11),color='tab:orange',edgecolor='white')
    axes[1].axvline(.5,color='black',ls='--');axes[1].set(xlabel='Conductivity-weight fraction at k*l < 1',ylabel='Number of prediction states',title='472 examined states; no filtering')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'property_scope.{ext}',dpi=170)
    plt.close(fig)


def main():
    files=['numerical_gates.json','validation/contract_audit.json','source_adaptation_v2/validation/contract_audit.json',
        'research_runtime_v2/verification.json','property_evaluation/contract_audit.json','relation_diagnostic.json','temperature_anchor_protocol_audit.json','predictor_scope_gates.json','source_selected_update_audit.json']
    checks={p:json.loads((OUT/p).read_text()) for p in files};assert all(v['passed'] for v in checks.values())
    a=checks['validation/contract_audit.json'];s=checks['source_adaptation_v2/validation/contract_audit.json'];r=checks['research_runtime_v2/verification.json'];q=checks['property_evaluation/contract_audit.json'];g=checks['numerical_gates.json']
    figures();mu=OUT/'property_evaluation/summary.csv';nt=OUT/'property_evaluation/RT_specimen_budget_nT_summary.csv'
    key=['Logit_inventory_original_RT','SourceLogit_Logit_inventory_original_RT','Logit_inventory_binding_joint','SourceLogit_Logit_inventory_binding_joint',
        'Logit_add_joint','SourceLogit_Logit_add_joint']
    text=r'''# ML与Sandbox：共享分配关系、少样本适应与泛化检验

## 1. 方向和这一轮实际完成的工作

目标仍是“掺杂元素/实际掺入量/可追溯工艺→训练的内部共享关系→物理Sandbox→掺杂性质预测”。载流子是当前主终点，mu与派生sigma继续验证。理论分析用于设计、核验并修正实际预测模型，不以解释性报告完成整个goal。

本轮实现log-odds库存学习、同容量输出、两种直接激活对照及内部来源适应；训练后导出可调用预测器并独立重算。原29片Si/Sn、5DOI、0/1/2个室温nH样品锚点、29/23/18查询和共同18不变，保留全部旧强结果。历史队列已经多次查看，本轮是回顾性研发，不是独立盲测。没有新增物理样品、DFT或MD，本低维训练使用CPU，GPU1授权保留。

主要结果不支持新库存函数整体胜出。库存原RT零锚点0.460056dex，高于上一轮内选库存0.353275；相同联合标签的直接add/product约0.26254/0.26253，仍须保留。内部来源更新只在部分预算改善，不能称已解决泛化。新关系的训练损失多半更低却留出误差更高，说明继续增加表达自由度不是有证据的默认下一步。

## 2. 输入情景与共享学习器

实际掺入量C以cm^-3计，H为已报氢；未知H继续旧零情景，C杂质没有编造形成能。所有来源氧分压缺失，统一1e-5atm是假设。记录Tcap不是完整实际热史；固定温度坐标6，Tf在600K与Tcap之间，Mauze900°C仅是区间上端。完整实际峰值/冷却轨迹已知数仍为0。

令x=log10(C/10^18cm^-3)，学习两个共享无量纲参数a、b：

\[
 g=a+bx,\qquad S=\ln(1+e^g),\qquad
 \alpha=1-e^{-S}=\frac{1}{1+e^{-g}}.
\]

S是非负分配坐标，alpha是掺杂复合体中X原子转入既有孤立X_Ga母类的比例。所有来源复用相同a、b；实测查询nH不作为输入。alpha在数学上处于(0,1)，浮点极限可舍入端点。a=b=0给alpha=.5，不能称为无分配参考。

母类库存c以cm^-3计，D_L选择含额外Ga缺失的纯Si/Sn复合体，u_i是该母类X原子数，r为两个参考孤立接收母类的归一化权重。复用阶段17算子：

\[
 c'=R_\alpha c,\qquad R_\alpha=I-\alpha D_L+\alpha r u^T,
 \qquad A R_\alpha=A.
\]

A每列为母类中的Si/Sn/H原子数。非负矩阵R保证库存非负；A D_L=e_XuT、A r=e_X保证外来原子逐项守恒。Ga/O是开放生长端点交换，不是封闭原子反应证明。H混合复合体不参加该最小转移。共享alpha是条件预测关系，不是测得反应率、DFT形成能或唯一原子几何。

## 3. 完整梯度和冷态物理求解

原热方程、接收比例r及其导数、固定库存电荷根和Hall矩全部保留。新封装只组合冻结constant-complex模型h(w,S;v)，w是背景/绑定参数，v是物理输入：

\[
 S_a=\alpha,\quad S_b=x\alpha,\quad
 S_{\log_{10}C}=b\alpha,\qquad
 h_\theta=h_{(w,S)}\frac{\partial(w,S)}{\partial\theta}.
\]

剂量输入额外加h_S b alpha；mu、sigma也需要同样链式求导，不能只更新nH。观测温度对输运的显式导数保留。

冷态总电荷Q=sum_i c'_i〈q〉_i+p+Nd−n。正温度下Q_EF=p_EF−n_EF−sum_i c'_i chi_i/(kBT)<0，实际端点括区异号还需检查。普通母类chi为电荷方差；固定局域/永久电离子群施主chi=ell f(1−f)，不能替换为总体混合概率的方差。因而在这些条件下根至多一个，IFT给dEF/dtheta=−Q_theta/Q_EF。n是自由电子，p为模型空穴项，Nd为正背景，EF为费米能级，kB为玻尔兹曼常数。微小Q_EF仍可病态。

同一状态用实际Nq²、自由电子屏蔽及Hall矩读出nH=n/rH、muH=rH mu_d与sigma=e nH muH。e是正元电荷，单位分别cm^-3、cm²/Vs和S/cm。固定205额外迁移率是经验先验，不是新声子计算；单标量能带、Born/RTA与自由电子屏蔽的适用域不由恒等式自动验证。

## 4. 固定8个基础方法和少样本协议

原RT只监督nRT，E(C)基础监督nRT+.1nT（训练T≤275K），按片内读数、DOI内样品、训练DOI等权。两个基础各含：内部logit库存；log10 nH加c0+c1 x的仿射输出；直接add为Nd+(C+H)sigmoid(g)；直接product为(C+H+Nd)sigmoid(g)。后两者背景所在括号不同，都不调用缺陷目录；RT-product准确复用阶段17旧拟合，没有增加重训预算。直接联合模型没有显式温度特征，因此其室温改善不代表解释了真实变温激活。

a,b∈[−12,12]，输出c0,c1∈[−6,6]dex，背景log10 Nd∈[14,18]，E0与屏蔽范围沿用旧域。原RT库存/输出3参数，E(C)5参数，直接模型3参数；参数数相同不等于函数范围或运行成本相同。两个预设起点、120求值上限和1e-7停止阈值；所有失败起点按统一规则复核，不能按外层好坏处理。

基础方法以原7个lambda在内层整DOI选择一个来源输出偏移。另对两库存、两add、两product共6方法固定检验内部/函数log-odds更新g_s=g+ln(10)z，z∈[−6,6]：

\[
 \min_z\left\{\sum_{j\in\mathcal A}
 [\log_{10}n_{H,j}(z)-y_j]^2+\lambda z^2\right\}.
\]

A集合仅含授权的室温样品锚点，y_j是其观测log10 nH。lambda→infinity固定z=0；有限lambda以49节点、所有采样局部谷的有界细化求解，保留端点和零，不宣称连续全局最优。lambda只看训练来源内留出MAE，同分优先更强收缩。z是log10 odds，输出偏移是dex，使用相同lambda网格不代表相同函数先验。库存内部更新重新求电荷与mu，直接模型更新其激活比例；本轮无mu锚点或新mu训练。
'''
    text+=f'\n实际新增{a["new_fits"]}拟合/{a["initial_new_starts"]}起点、{a["inherited_references"]}复用参考，共{a["records"]}记录，全部达到预设停止，无状态失败或查询缺失，最大损失重建差{a["maximum_objective_replay_error"]:.3g}。基础{a["inner_scores"]}与来源适应{s["inner_scores"]}条内层分数独立复核。来源{s["episodes"]}情景×7共2100条正则化解，含1800有限惩罚与300零更新；{s["source_prediction_callbacks"]}次锚点前向回调，0失败探针、0查询状态失败。17个情景解触z边界，外层选中的解触边界为0；这些有效坐标不是微观真值。\n'
    text+='\n## 5. 室温载流子：全部14方法\n\nDOI等权log10 MAE，单位dex。每预算查询组成不同，必须同时看共同18。\n\n'+table(METHODS)
    text+='\n### 共同18：相同主任务查询\n\n'+table(key,scope='common18')
    old=BASE/'phase15_coherent_hall/validation/primary_summary.csv';older=['original_RT','HallII_original_RT','binding_screen_joint','direct_ridge_RT','direct_ridge_joint']
    text+='\n### 保留的强旧对照\n\n| 方法 | 0锚点 | 1锚点 | 2锚点 |\n| --- | --- | --- | --- |\n'
    for m in older:
        vals=[value(old,dict(method=m,budget=b,scope='all_query')) for b in (0,1,2)]
        text+='| '+m+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |\n'
    text+='\n旧方法的热范围/Hall近似不完全相同，表中不将旧RT-only与新joint当作相同标签；本轮两直接joint和库存joint才是同标签比较。阶段12常数库存内部1锚点.229552、常数输出2锚点.250297，以及阶段17全部31方法也继续保留，没有被本轮稳定超过。\n'
    value(BASE/'phase12_stock_adaptation/physical_scope_v2/summary.csv',dict(model='constant_stock',budget=1,scope='all_query'))
    value(BASE/'phase12_stock_adaptation/physical_scope_v2/summary.csv',dict(model='constant_offset',budget=2,scope='all_query'))
    text+='\n![相同样品锚点预算与共同18的完整比较](figures/carrier_and_adaptation.png)\n'
    text+='\n内部来源更新使原RT库存1锚点.507597→.450825，但2锚点.392878→.431290变差；联合库存也没有稳定提升。直接joint内部更新1锚点约.268525、2锚点.301519，优于其输出更新但仍未超过强旧对照。不能用一个预算或一个来源的改善宣称整个链条成立。逐来源表和所有逐片输出完整保存。\n'
    text+='\n进一步核对选择结果：原RT库存的1锚点5个外层来源全部选择lambda=infinity，z全部为0；两种直接RT也相同。全6变体的30个1锚点来源情景有27个选择无限收缩。因此原RT库存1锚点优于输出校准，实际是避免了不利更新，不能称为学习了有效的内部适应。在共同18上，它0/1锚点均为0.436673，2锚点才降至0.431290，幅度小且仍不及旧强对照。全查询0→1的下降还含查询组成变化。source_selected_update_audit.json保留每个坐标与选择。\n'
    text+='\n## 6. 两种变温锚点协议，分开汇报\n\n主任务预算为每来源0/1/2个室温样品锚点。旧补充变温函数则对每条曲线使用0/1/2个温度标签：先RT，再最近150K点，另有RT附近25K与第二锚点附近10K的guard。它们的信息预算不同；旧表的1/2列不能与来源内部更新的1/2列直接比较。原主任务和旧冻结数值没有改变，新增逐行重建审计及协议说明。\n'
    text+='\n### 主任务样品预算下的自身温度nH\n\n以下统一使用主任务室温样品锚点，没有额外150K标签。共同样品且40–125K的固定查询共有37读数，分别片内/来源等权；它不是旧7片共同80读数任务。全部14方法的全查询和共同集结果在RT_specimen_budget_nT_summary.csv。\n\n'+table(key,nt,'common_specimens_T40_125')
    text+='\n### 旧曲线温度锚点任务，仅作独立补充\n\n下表是8基础方法的旧协议共同80读数，每条曲线自带温度锚点。该任务没有用于内部来源lambda选择。\n\n'+table(list(study.STRUCTURES),OUT/'validation/temperature_summary.csv','common40_125')
    text+='\n## 7. 完整性质链与输运边界\n\nmu始终在自身T预测，不输入实测查询nH。RT仍仅5配对/2DOI，变温mu54读数/3旧片/1DOI；sigma仅RT nH×mu派生，没有独立标签。以下0锚点列属于不同队列；1/2锚点全部结果及共同2片RT、共同1片24读数变温mu均保存。\n\n| 方法 | RT mu MAE | 变温mu MAE |\n| --- | --- | --- |\n'
    for m in METHODS:
        vals=[value(mu,dict(method=m,budget=0,scope='all_query',kind=k)) for k in ('RT','curve')]
        text+='| '+label(m)+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |\n'
    text+='\n### 同一RT配对上的完整链，0锚点\n\n| 方法 | nH MAE | mu MAE | 派生sigma MAE |\n| --- | --- | --- | --- |\n'
    for m in key:
        vals=[value(OUT/'property_evaluation/full_chain_summary.csv',dict(method=m,budget=0),c) for c in ('nH_MAE','mu_MAE','sigma_MAE')]
        text+='| '+label(m)+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |\n'
    text+='\n配对子集的nH不是29片主误差；sigma两项偏差可抵消，不能当作第三个独立成功端点。内部source更新会改变物理mu，输出偏移不改物理mu；直接模型会按校准后的预测nH重算CT。低温CT超出其300–500K经验校准，明确标为外推。\n'
    text+=f'\n本轮{q["ownT_mu_rows"]}行mu、{q["same_RT_specimen_budget_nT_rows"]}行同样品预算nT及{q["RT_fullchain_query_rows"]}行RT完整链查询全部保留。检验{q["transport_scope_states"]}个不同预算/模型类别的mu状态，{q["majority_sigma_kl_lt1"]}个过半电导权重在kl<1。kl=2E tau/hbar，E为带底动能、tau为弛豫时间；这是弱散射诊断刻度，不是普适材料下界。512/256节点mu差{q["maximum_moment_replay_error"]:.3g}dex，最大外来原子误差{q["maximum_foreign_atom_relative_error"]:.3g}。束缚平衡响应/自由屏蔽之比最高{q["maximum_bound_response_over_free_screen"]:.3g}，不能未经动态尺度论证就把束缚项加入散射屏蔽。未删除任何行，没有认证Born材料机制。\n'
    text+='\n![完整链权衡及输运适用域](figures/property_scope.png)\n'
    text+='\n## 8. 理论怎样被实验修正\n\n旧alpha的log-log斜率幅度≤1/ln10，新logit可超过这一界，但二者并非嵌套。旧s0>0,s1>0的反向softplus在高剂量有非零平台1−exp(−s0)，低剂量趋于1；非恒定affine-logit不能同时恢复这两个极限。因此解除某个斜率限制不等于完整保留旧函数族再增加能力。反向包含又被斜率反例排除。原RT旧s0全部近零（最大约6.7e-10），所以失去非零平台也不能充分解释该组失败。\n'
    text+='\n90条同训练上下文/同损失比较中，相对旧反向softplus，原RT的15个上下文有14个新训练误差更低，联合为11个更低、4个更高。新原RT斜率b约−5.41到−1.36，联合约−6.12到.34。它们留出预测却较差，提示应检查复杂度先验、跨来源分布和未报告变量，而不是直接推断“需要更多参数”。该比较不使用外层标签选参数，但其解释仍属回顾性诊断，不能唯一证明某种过拟合机制。\n'
    text+='\n![训练损失更低并不保证来源泛化更好](figures/training_diagnostic.png)\n'
    text+='\n## 9. 数值门槛、失败与独立使用\n'
    text+=f'\n最终{len(g["checks"])}项组合梯度/守恒/根/标签门槛通过，最大缩放差{g["maximum_scaled_derivative_error"]:.3g}<5e-6。初次662检查有2个零更新比较失败，原代码/结果保留：它把多次参数扰动后热求根路径的状态与初始状态混为接口零更新误差。零更新改与连续相同输入比较，原1e-9阈值不变；不同热路径漂移最大4.05e-9dex，按既有2e-6回放标准另记。模型公式、解析梯度和接受阈值未改，训练在全部门槛通过后开始。\n'
    text+=f'\n{r["saved_bundles"]}模型束、{r["independent_rows"]}独立输出，无查询标签或性质表读取。保存来源坐标回放nH差{r["maximum_fixed_carrier_error"]:.3g}dex、输运{r["maximum_fixed_transport_error"]:.3g}；仅凭允许锚点重新推断坐标，输出差仍约{r["maximum_source_refit_transport_error"]:.3g}dex，log-odds坐标差{r["maximum_logodds_coordinate_error"]:.3g}。这是实际预测复用证据，不是新来源材料精度。这里回放的是主任务样品预算；旧曲线温度锚点表由另一审计逐行重建，不能混称。\n'
    text+='\n首次runtime实际worker输出和verification已保存通过，但父进程最后打印JSON时numpy布尔量报错、退出1。完整记录保留；V2仅清理打印标量并用新目录完整重跑，进程退出0，无重训或物理变化。来源脚本V1/V2在执行前的代码核对改为同一次状态返回nH/mu/sigma，保留未执行V1协议；不是事后调整材料拟合。\n'
    text+='\n新预测API可以从保存束加载、用0/1/2个用户提供nH锚点校准并预测，不读取实验表。推荐CheckedLogOddsResearchPredictor：24项域门槛检查其对非Si/Sn、无效剂量/温度/H状态/来源坐标/正则化的拒绝，两条支持分支委托同一数值映射。它不能把未验证的稀土元素套入Si/Sn激活关系。README给出API和命令。这些是交叉验证折模型，不是全5来源部署模型，输入域/适用级别必须保留。\n'
    text+='\n## 10. 文献初筛与下一步实际预测工作\n\n新增检索未确认可加入主任务的新Hall样品。[IWGO2026 MoP-18作者摘要](https://avsconferences.org/IWGO2026/Sessions/Schedule/88739)虽然有Sn SIMS与生长条件，终点却是Hg-probe C-V净电荷；不能直接改名Hall载流子浓度。[2025厚漂移层论文](https://doi.org/10.1063/5.0302735)的相关UID化学浓度只有检测限约束，不能把上限作为确定剂量。2018 PLD论文10.1063/1.5047214的全文配对仍待审计，索引/下载日期不能当2026新发表；alpha相材料也不混入beta能量目录。读取深度、直开失败和候选状态均见literature_leads.json，没有宣称新全文系统复现。\n'
    text+='\n下一步先按完整goal逐项核对剩余交付，训练全来源并导出证据支持的强预测模型及输入适用域，再审计独立化学剂量/Hall来源；确需继续学习改进时，用训练内正则化与可精确恢复强参考的分支控制增量复杂度。当前不自动增加神经网络或反应PDE。新理论与后续方案见ML_SANDBOX_GENERALIZATION_NEXT_20261018.md，尚未执行。主方向仍是ML＋Sandbox预测掺杂性质；所有负面结果和旧强对照保留，goal继续active。\n'
    dest=OUT/'ML与Sandbox_分配学习与来源泛化检验.md';dest.write_text(text)
    evidence=[OUT/p for p in files]+[old,OUT/'validation/primary_summary.csv',audit.source.OUT/'validation/primary_summary.csv',
        OUT/'validation/temperature_summary.csv',nt,mu,OUT/'property_evaluation/full_chain_summary.csv',
        BASE/'phase12_stock_adaptation/physical_scope_v2/summary.csv',OUT/'literature_leads.json']
    energy.write(OUT/'report_claims.json',dict(code_sha256=energy.sha(Path(__file__)),report_sha256=energy.sha(dest),
        evidence_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in evidence},numeric_claims=CLAIMS,methods=14,
        anchor_protocols_explicitly_separated=True,universal_prediction_gain_established=False,goal_complete=False))
    print(dest);print('numeric claims',len(CLAIMS))


if __name__=='__main__':main()
