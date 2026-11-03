"""Assemble the conserved-inventory prediction checkpoint without hiding controls."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv as read
import phase17_audit as audit
import phase2_energy as energy

study=audit.study;OUT=study.OUT;BASE=energy.BASE
CLAIMS=[]


def scalar(path,selectors,column='MAE'):
    data=read(path)
    for key,value in selectors.items():data=data[data[key].eq(value)]
    assert len(data)==1,(path,selectors,len(data))
    v=float(data[column].iloc[0])
    CLAIMS.append(dict(path=str(path.relative_to(ROOT)),selectors=selectors,column=column,value=v))
    return v


def short(method):
    if method=='activation_sigmoid_RT':return '直接sigmoid／RT'
    if method.startswith('Direct_ridge_cap_'):return '直接Ridge含Tcap／'+method.split('_')[-1]
    if method.startswith('Matched_activation_'):
        return method.replace('Matched_activation_','直接同形／').replace('_','／')
    return (method.replace('Inventory_reference_','无分配参考／')
            .replace('Inventory_selected_','内选库存／').replace('Output_selected_','内选输出／')
            .replace('Activation_selected_','内选同形／').replace('Stock_','')
            .replace('binding_joint','E(C)').replace('original_RT','原RT')
            .replace('native_constant','原生常数').replace('complex_constant','复合常数')
            .replace('complex_dose_plus','复合剂量＋').replace('complex_dose_minus','复合剂量－')
            .replace('output_constant','输出常数').replace('output_dose_plus','输出剂量＋')
            .replace('output_dose_minus','输出剂量－').replace('_','／'))


def table(path,methods,scope='all_query',kind=None,old=False):
    rows=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for method in methods:
        values=[]
        for b in (0,1,2):
            selector=dict(method=method,budget=b,scope=scope)
            if kind:selector['kind']=kind
            values.append(scalar(path,selector))
        rows.append('| '+(method if old else short(method))+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    return '\n'.join(rows)+'\n'


def figures():
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,3.9))
    methods=[('Inventory_selected_original_RT','Conserved inventory'),('Activation_selected_RT','Matched activation'),
             ('activation_sigmoid_RT','Direct sigmoid'),('Direct_ridge_cap_RT','Ridge with cap')]
    new=read(OUT/'validation/primary_summary.csv')
    old=read(BASE/'phase15_coherent_hall/validation/primary_summary.csv')
    for ax,scope in zip(axes,('all_query','common18')):
        for method,label in methods:
            d=new[new.method.eq(method)&new.scope.eq(scope)].sort_values('budget')
            ax.plot(d.budget,d.MAE,'o-',label=label,ms=4)
        d=old[old.method.eq('original_RT')&old.scope.eq(scope)].sort_values('budget')
        ax.plot(d.budget,d.MAE,'s--',label='Retained original model',ms=4,color='black')
        ax.set(xticks=[0,1,2],xlabel='Permitted nH specimen anchors',ylabel='Source-equal nH MAE (dex)',
               title='29 / 23 / 18 queries' if scope=='all_query' else 'Identical common 18 specimens')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'carrier_prediction.{ext}',dpi=170)
    plt.close(fig)
    methods=['Inventory_reference_original_RT','Stock_native_constant_original_RT',
             'Stock_complex_constant_original_RT','Stock_complex_dose_minus_original_RT','activation_sigmoid_RT']
    s=read(OUT/'validation/primary_by_source.csv');s=s[s.scope.eq('all_query')&s.budget.eq(0)]
    p=s.pivot(index='method',columns='family',values='MAE').reindex(methods)
    fig,ax=plt.subplots(figsize=(8.9,3.7));im=ax.imshow(p.to_numpy(),vmin=0,vmax=4,cmap='magma')
    ax.set(yticks=range(5),yticklabels=['Conditional cap reference','Native only','Complex constant','Complex dose minus','Direct sigmoid'],
           xticks=range(5),xticklabels=[{'10.1002/aelm.202300679':'TEOS','10.1007/s10853-015-9693-6':'Baldini',
              '10.1007/s40843-022-2167-x':'Xiang','10.1063/5.0027870':'Mauze','10.1063/5.0101132':'McCandless'}[f] for f in p.columns],
           title='Retain every source: zero-anchor nH error')
    plt.setp(ax.get_xticklabels(),rotation=15,ha='right',fontsize=8)
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{p.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white')
    fig.colorbar(im,ax=ax,label='MAE (dex)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'source_errors.{ext}',dpi=170)
    plt.close(fig)
    mu=read(OUT/'property_evaluation/summary.csv');scope=read(OUT/'property_evaluation/transport_scope_summary.csv')
    methods=['Inventory_reference_original_RT','Stock_native_constant_original_RT',
             'Stock_complex_constant_original_RT','Stock_complex_dose_minus_original_RT','Inventory_selected_binding_joint','activation_sigmoid_RT']
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    for m in methods:
        x=new[new.method.eq(m)&new.budget.eq(0)&new.scope.eq('all_query')].MAE.iloc[0]
        y=mu[mu.method.eq(m)&mu.budget.eq(0)&mu.kind.eq('RT')&mu.scope.eq('all_query')].MAE.iloc[0]
        label={'Inventory_reference_original_RT':'Reference','Stock_native_constant_original_RT':'Native',
            'Stock_complex_constant_original_RT':'Complex constant','Stock_complex_dose_minus_original_RT':'Complex dose minus',
            'Inventory_selected_binding_joint':'Selected inventory E(C)','activation_sigmoid_RT':'Direct sigmoid'}[m]
        axes[0].scatter(x,y,s=35);axes[0].annotate(label,(x,y),xytext=(4,4),textcoords='offset points',fontsize=7)
    axes[0].set(xlabel='nH MAE, 29 specimens / 5 DOI (dex)',ylabel='muH MAE, 5 specimens / 2 DOI (dex)',title='Different endpoint cohorts; no universal win')
    axes[0].grid(alpha=.2)
    d=scope.groupby('kind')[['states','majority_sigma_kl_lt1']].sum()
    axes[1].bar(range(len(d)),d.states,color='lightgray',label='All examined states')
    axes[1].bar(range(len(d)),d.majority_sigma_kl_lt1,color='tab:orange',label='Majority conductivity at k*l < 1')
    axes[1].set(xticks=range(len(d)),xticklabels=d.index,title='Transport scope remains conditional',ylabel='Prediction states; no filtering')
    axes[1].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'property_scope.{ext}',dpi=170)
    plt.close(fig)


def main():
    evidence_rel=['inventory_audit/contract_audit.json','numerical_gates.json','validation/contract_audit.json',
        'research_runtime_v2/verification.json','property_evaluation/contract_audit.json']
    checks={p:json.loads((OUT/p).read_text()) for p in evidence_rel}
    assert all(v['passed'] for v in checks.values())
    a=checks['validation/contract_audit.json'];r=checks['research_runtime_v2/verification.json'];q=checks['property_evaluation/contract_audit.json']
    gates=checks['numerical_gates.json'];methods=list(audit.ALL_METHODS)
    figures()
    npath=OUT/'validation/primary_summary.csv';mupath=OUT/'property_evaluation/summary.csv'
    text=r'''# ML与Sandbox：守恒库存学习与跨来源掺杂性质预测

## 1. 目标、问题与本轮结论

研究方向保持为“实际掺杂输入→跨条件共享的ML内部关系→物理Sandbox→掺杂性质预测”。当前主要终点为29片Si/Sn、5个DOI的室温Hall载流子浓度nH；0/1/2个样品nH锚点对应29/23/18个查询，固定共同18个样品用于比较适应收益。迁移率mu和派生电导率sigma继续属于完整目标。理论分析服务于模型改进，不以解释性文档完成整个goal。

上轮将热库存温度限制在记录条件后，预测载流子严重不足。本轮不再用更高的有效冻结温度吸收全部失配，而训练一个在固定条件温度下改变掺杂原子分配的共享函数；随后重新求解电荷中性与同态Hall输运。它已实际训练、输出留出预测、保存并独立复用，尚未建立优于全部强基线的预测方法。

关键结果：库存分配显著减小记录约束参考的数dex失配，但旧强基线和直接sigmoid对照仍更好，迁移率也没有同步一致改善。所有固定模型和内选模型均报告；新模型不是观测到的真实原子反应或新微观速率。没有新物理样品、DFT或MD；GPU1授权保留，本轮使用CPU。

## 2. 数据、条件输入与先做的库存诊断

原29片/5DOI及锚点划分不变。7片Si旧样品的142个变温nH读数来自2DOI，只作补充；训练曲线温度上限275K。完整mu只有5个室温化学/Hall/mu配对、2DOI，以及3个旧样品的54个变温读数、1DOI。52组条件迁移率队列没有混入完整掺杂输入预测。sigma没有独立标签。

热审计18片有样品或同组温度、11片Mauze只有700–900°C区间；完整实际峰值与冷却轨迹为0。Tcap为明确记录所支持的条件上界，Mauze900°C是区间情景，不能称作每片实际生长温度。固定a=6，使用log10(Tf)=log10(600)+[log10(Tcap)−log10(600)]sigmoid(6)。不是重新自由拟合Tf。所有来源氧分压缺失，统一1e-5atm是假设；未知H使用旧零H情景；C未进入现有缺陷目录。

先从冻结参考重求58个RT状态，再做174个原生库存因子1/.1/0的固定参数反事实。TEOS中纯原生母类只承担模型负电荷约0.37%–0.93%；Si/Sn参与的Ga缺失复合体可承担主要补偿。移除纯原生库存并不能代表全部失配。这是当前模型内诊断，不是测得真实复合体浓度，亦非全局优化后的不可达定理。因此保留“只抑制原生库存”为对照，增加外来原子守恒的复合体分配候选。

## 3. ML到底学习什么：非负库存分配

母类i的参考库存c_i是cm^-3；母类是同一原子组成/结构下所有电荷态的集合。矩阵A的每列为一个母类含有的Si、Sn、H原子数，输入总外来原子浓度为D，故A c=D。原生集合N的A列全为零，可用exp(−s_n)缩放而不改变外来原子总数。

复合体集合L按净化学计量判定：含k_i>0个当前掺杂X原子，Ga变化d_Ga<−k_i，O变化为0，不含其他外来原子。接收集合由目录中两个孤立X_Ga母类组成；每个含1个X，替代1个Ga。r_h=c_h/(sum接收池c_h)是参考接收权重，u_i=k_i在L内、其他为0。定义非负可训练坐标s，转移比例alpha=1−exp(−s)，D_L为L的对角选择矩阵：

\[
 R_\alpha=I-\alpha D_L+\alpha r u^T,\qquad c'=R_\alpha c.
\]

I是母类空间的单位矩阵，u^T c是复合体所含X原子总浓度。R把减少的复合体X原子送入两个已有孤立母类；不是在最终nH上直接加修正。因为0≤alpha<1、r≥0，R逐项非负，所以c'≥0。由于A D_L=e_X u^T、A r=e_X：

\[
 A R_\alpha=A-\alpha e_Xu^T+\alpha e_Xu^T=A,\qquad A c'=D.
\]

e_X是外来元素空间中X对应的单位向量。Si/Sn/H逐项守恒；H混合复合体不参加这个最小转移。R_0=I，零分配准确恢复参考库存。Ga/O是开放生长端点库存，代码记录其变化，不能把此映射称为封闭Ga/O守恒反应、已测解离路径或动力学耗散定理。

本轮共享关系为s(C)=s0，或s0+s1 softplus(±x)，s0,s1≥0，x=log10(C/10^18cm^-3)，softplus(x)=ln(1+exp(x))。C是实际掺入浓度；测试样品的实测nH不作为共享关系输入。常数、剂量正向和反向均固定报告，也用训练来源内选择比较。原RT结构固定施主统计，E(C)结构另学习共享电离能/屏蔽关系。没有新增DFT能量标签。

## 4. 从库存到性质：完整导数与条件唯一性

参考热态由F_hot(v,p)=0求得；v含费米能级与化学势，p是原始输入和参数。雅可比F_v可逆时，隐函数定理给dv/dp=−F_v^-1 F_p。参考库存c随v和p同时变化。对于任一学习参数或物理输入t：

\[
 \frac{dc'}{dt}=R_\alpha\frac{dc}{dt}
 +\alpha\frac{dr}{dt}(u^Tc)
 +\frac{d\alpha}{dt}(ru^T-D_L)c,
 \qquad
 \frac{dr_h}{dt}=\frac{dc_h/dt-r_h\sum_{j\in I}dc_j/dt}{\sum_{j\in I}c_j}.
\]

接收比例r自身的导数不能省略。新封装在原方程上做局部Newton精度抛光，保留旧库和结果；旧2e-9量级热残差在强补偿小自由电荷中可能放大。此举改变数值精度，不改变热态物理方程。

测量温度T下冻结c'，普通母类电荷概率pi_iq正比于g_iq exp[−(A_iq+q EF)/(kB T)]。A_iq为参考一致的自由能截距、q为正元电荷单位的电荷数、g为简并权重、EF为费米能级、kB为玻尔兹曼常数。归一化保证sum_q pi_iq=1。总电荷方程与导数为

\[
 Q(E_F)=\sum_i c'_i\langle q\rangle_i+p(E_F)+N_d-n(E_F)=0,
 \qquad
 Q_{E_F}=p_{E_F}-n_{E_F}-\frac{\sum_i c'_i\chi_i}{k_BT}<0.
\]

n是自由电子浓度、p是模型声明的空穴项之和、Nd为有效正背景。普通母类chi_i=Var(q)；固定局域/永久电离子群的施主则chi=ell f(1−f)，其中ell为固定局域比例、f为局域中性概率。不能使用混合总体的ell f(1−ell f)，因为两子群不在测量中互相转换。正温度下电子压缩率为正、空穴导数非正，所以给定库存至多一个根；仍须检查实际括区异号来保证存在。很小的Q_EF仍会带来病态灵敏度。

对任一参数t，dEF/dt=−Q_t/Q_EF。完整梯度包含库存、T、背景、绑定能与能带参数的显式项，再传入实际平方电荷散射预算Nq²、自由电子屏蔽和Hall矩；观测温度对输运的直接导数也单独加入。固定库存下所有电荷态截距共同平移不改变归一化概率，不能据此说生长库存完全不依赖能量。

最终nH=n/rH、muH=rH mu_d，电导率sigma=e n mu_d=e nH muH；e是正元电荷，浓度cm^-3、迁移率cm²/Vs给sigma单位S/cm。恒等式保证读出相容，不证明Born/RTA或材料机制正确。自由电子屏蔽、标量能带、固定205cm²/Vs额外率均沿用；205是历史经验先验，不是本轮声子计算。

## 5. 训练、模型选择与公平性

两个基础各含无分配参考、原生常数、复合常数/正向/反向、输出常数/正向/反向，加一个直接sigmoid，共17个基础方法。另加6个同形直接激活和2个获得显式Tcap输入的Ridge，个体方法合计25。六个族选择器分别在库存、输出、同形激活的常数/正/反成员间联合选择成员与原7个来源收缩lambda。所有选择只看内层整DOI留出nH，分数1e-10舍入同分优先更大lambda，再优先常数/正/反预定顺序。没有按外层结果选“最佳基础”。

原RT只监督nRT；E(C)监督nRT+.1nT，曲线限275K；每块先片内读数、再DOI内样品、再DOI等权。当前批无新mu监督，也没有mu锚点；旧阶段16联合监督结果完整保留。外层排除整个来源的所有标签；锚点仅允许对应预算的nRT样品。共享函数在所有测试来源复用，来源更新为一个收缩nH输出偏移，未部署新的内部来源库存更新。

同容量输出控制在log10 nH上加c0+c1 softplus(±x)，不改物理mu；sigma随校准nH同步变化，显式标记不是同一未校准物理状态。参数数相同不意味着函数容量、先验范围或运行时间相同。匹配直接激活为nH=Nd+(C+H)[1−exp(−s(C))]；另一直接sigmoid为(C+H+Nd)sigmoid(a+b x)，背景是否处于激活括号内不同，不能把两者当作精确相同函数族。Ridge含显式Tcap、剂量、温度、元素及已知H/C标记，标准化只用训练来源。

初始协议在训练前固定；补充同形/Tcap控制协议在基础训练运行中、任何新外层评分之前写定；六个族选择规则也在外层评价前写定。这些均是已经多次查看历史队列后的回顾性研发，不能称预注册或盲测。相同信息预算、候选数、参数数和实际成本分开说明。
'''
    text+=f'\n本轮新增基础非线性拟合{a["new_base_nonlinear_fits"]}、匹配非线性拟合{a["new_matched_nonlinear_fits"]}，共{a["initial_nonlinear_optimizer_starts"]}个起点；Ridge凸拟合{a["new_convex_Ridge_fits"]}，参考复用{a["inherited_references"]}，合计{a["fit_records"]}条记录。均按预设停止条件终止，0状态失败、0主查询缺失；优化器停止不是全局最优证明。独立复算{a["inner_scores"]}条内层分数，{a["RT_saved_rows"]}条RT及{a["nT_saved_rows"]}条变温nH保存行，最大目标重算差{a["maximum_objective_replay_error"]:.3g}。\n'
    text+='\n## 6. 载流子主结果：全部31个方法\n\n以下是DOI等权log10误差MAE，单位dex；0.3dex约为2倍误差量级。每预算各自查询集不同，跨预算评价须同时看后面的共同18。\n\n'+table(npath,methods)
    keymethods=['Inventory_selected_original_RT','Inventory_selected_binding_joint','Output_selected_original_RT',
        'Activation_selected_RT','Activation_selected_joint','activation_sigmoid_RT','Direct_ridge_cap_RT']
    text+='\n### 同一共同18片\n\n'+table(npath,keymethods,'common18')
    text+='\n![实际跨来源载流子预测；旧强对照继续保留](figures/carrier_prediction.png)\n'
    oldpath=BASE/'phase15_coherent_hall/validation/primary_summary.csv'
    old=read(oldpath);oldmethods=list(old[old.scope.eq('all_query')&old.budget.eq(0)].method)
    text+='\n### 保留的旧基础对照全表\n\n以下结构、温度范围和Hall算子并非全与新记录约束模型相同；其作用是避免只击败严重失配参考就宣称最佳预测。原研究主任务与信息预算保持。\n\n'+table(oldpath,oldmethods,old=True)
    p12=BASE/'phase12_stock_adaptation/physical_scope_v2/summary.csv'
    d=read(p12)
    scalar(p12,dict(model='constant_stock',budget=1,scope='all_query'))
    scalar(p12,dict(model='constant_offset',budget=2,scope='all_query'))
    scalar(BASE/'phase15_coherent_hall/validation/temperature_summary.csv',dict(method='binding_screen_joint',budget=0,scope='common40_125'))
    scalar(BASE/'phase16_thermal_joint/property_evaluation/summary.csv',dict(method='Thermal_binding_joint_effective_g1',budget=0,scope='all_query',kind='curve'))
    text+='\n阶段12全部来源适应结果继续保留于physical_scope_v2/summary.csv。其1锚点常数库存＋内部适应0.229552dex、2锚点常数库存＋输出适应0.250297dex也没有被新模型稳定超过；不能把后者从比较中删除。\n'
    text+='\n### 来源差异与可成立的结论\n\n'+(read(OUT/'validation/primary_by_source.csv')
        .query("scope == 'all_query' and budget == 0")
        .loc[lambda t:t.method.isin(['Inventory_selected_original_RT','Inventory_selected_binding_joint'])]
        .pivot(index='family',columns='method',values='MAE').rename(columns=short).to_markdown(floatfmt='.6f'))+'\n'
    text+='\n![不删除任何来源的固定模型误差](figures/source_errors.png)\n'
    text+=r'''
零锚点原RT内选库存0.353275dex，明显优于无分配参考2.868890和原生常数1.065793；E(C)内选库存0.379779也优于其参考3.197473。这个信号说明在给定热情景下改变掺杂分配比只乘纯原生存留因子更有预测价值。但直接sigmoid0.312982、旧HallII原RT0.262469、旧原RT0.266289仍更好。新方法不能宣称总精度领先；E(C)和原RT的来源改善也不一致。

原RT内选库存全查询0/1/2误差0.353275/.316399/.430459，在共同18为.391626/.380918/.430459；两个锚点反而变差。固定剂量反向模型2锚点.404366好于族选择器.430459，表明小训练来源下成员选择本身存在代价，不能事后用外层好成员替换内层选择。

部分新库存模型优于本轮同形激活和输出模型，并不证明底层缺陷能量不可替代：模型函数容量、非线性与温度路径不完全等同，且更强直接sigmoid仍胜出。仅看到某个对照失败不能建立普遍物理增益或唯一微观解释。
'''
    ntpath=OUT/'validation/temperature_summary.csv'
    text+='\n## 7. 补充变温载流子预测\n\n仍为原7片Si、2DOI、142读数；共同40–125K有80读数。所有方法完整输出在temperature_summary.csv，以下固定报告关键对照，共同温区按相同样品/来源权重。\n\n'+table(ntpath,keymethods,'common40_125')
    text+='\n旧E(C)在共同温区约0.308611dex，仍优于当前原RT库存约0.876785与E(C)库存约1.053019（0锚点）。不能用局部温度拟合或样本数重复增加来声称新增独立验证。\n'
    text+='\n## 8. 完整性质链：全部31方法的零锚点mu\n\n逐读数在其实际T求解；不输入查询实测nH。室温mu为5配对/2DOI；变温mu为54读数/3旧片/1DOI。表中各列队列不同。其余预算、逐来源及共同查询全表已保存，不以零锚点表替代它们。\n\n| 方法 | 室温mu MAE | 变温mu MAE |\n| --- | --- | --- |\n'
    for method in methods:
        vals=[scalar(mupath,dict(method=method,budget=0,scope='all_query',kind=kind)) for kind in ('RT','curve')]
        text+='| '+short(method)+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |\n'
    text+='\n### 共同2片室温mu的预算比较\n\n'+table(mupath,keymethods,'common_RT2','RT')
    text+='\n### 共同1片/24读数变温mu的预算比较\n\n'+table(mupath,keymethods,'common_curve','curve')
    text+='\n无分配原RT参考muRT0.751185，内选库存1.023539反而更差；变温mu0.886541→0.315388改善，但E(C)库存0.197736仍不及旧阶段16有效E(C)gamma1约0.153771。不存在nH、mu同时全面改善。物理单体模型mu对nH输出校准不变；族模型不同预算可能选择不同成员，直接CT模型则会对校准后预测nH重新计算mu。查询组成变化也会改变平均误差，所以共同查询表不可省略。CT在低温沿用经验外推，不代表其300–500K校准范围得到扩展。\n'
    fullpath=OUT/'property_evaluation/full_chain_summary.csv'
    text+='\n### 同一室温化学/Hall/mu配对上的完整链（0锚点）\n\n| 方法 | nH MAE | mu MAE | 派生sigma MAE |\n| --- | --- | --- | --- |\n'
    for method in keymethods+['Inventory_reference_original_RT']:
        vals=[scalar(fullpath,dict(method=method,budget=0),column=c) for c in ('nH_MAE','mu_MAE','sigma_MAE')]
        text+='| '+short(method)+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |\n'
    text+='\n这里nH只限5配对样品，不能与29片主任务直接混用。sigma从已观测的RT nH×mu派生，误差可因两项偏差抵消，不是新增独立验证端点。\n'
    text+='\n## 9. 物理适用域没有被低误差自动修复\n\n相同能量分辨输运中k l=2E tau/hbar，k为波矢模、l为平均自由程、E为相对带底动能、tau为弛豫时间、hbar为约化普朗克常数。对所有基础物理/输出控制的零锚点mu状态统计kl<1的电导积分权重；只是声明的弱散射自洽诊断，不能当作所有材料的普适严格界。\n'
    text+=f'\n共{q["transport_scope_states"]}个状态，其中{q["majority_sigma_kl_lt1_states"]}个超过一半电导权重在kl<1；512与256节点mu重算最大差{q["maximum_moment_replay_error"]:.3g}dex，最大原生母类占位{q["maximum_native_parent_fraction"]:.3g}，最大外来原子相对误差{q["maximum_foreign_atom_relative_error"]:.3g}。未删除任何读数。\n'
    text+=f'\n固定库存束缚电荷的平衡响应Σc chi与自由电子屏蔽浓度之比最高{q["maximum_bound_equilibrium_response_over_free_screen"]:.3g}。这说明仅自由电子屏蔽是需要审计的近似；束缚响应能否参与实际散射屏蔽取决于时间/频率与空间尺度，不能直接把静态响应全部加入并宣称已修正真实屏蔽。physical_valid旧字段仅表求解域，不等于Born近似或材料机制已验证。\n'
    text+='\n![完整链的预测权衡与输运适用域](figures/property_scope.png)\n'
    text+='\n## 10. 数值验证、保留的失败与独立使用\n'
    text+=f'\n最终{len(gates["checks"])}项数值与合同门槛全部通过，最大缩放差{gates["maximum_scaled_derivative_error"]:.3g}，门槛不变为5e-6。初次有3项输出控制剂量输入差分失败，误差7.26e-6、7.24e-6、6.63e-6；完整代码/日志/失败门槛保留。输出控制的四阶差分步长从.002调为.005后通过，物理公式、解析梯度与接受阈值未改。更早强补偿精度探针也保留。没有在失败门槛下启动本轮训练。\n'
    text+=f'\n导出{r["saved_bundles"]}个来源排除模型束，独立子进程禁止读取实验、拟合、选择、热审计与库存审计表，使用物理输入和允许的nH锚点复算{r["independent_rows"]}行；nH最大差{r["maximum_carrier_error"]:.3g}dex，输运{r["maximum_transport_error"]:.3g}dex，电荷预算{r["maximum_charge_error"]:.3g}dex，来源偏移{r["maximum_source_offset_error"]:.3g}dex。查询标签未进入任务文件。它证明保存的学习关系可复用，不能替代新来源材料精度。\n'
    text+=f'\n首次独立回放在JSON写出时遇到numpy布尔标量不可序列化，进程失败。原代码、协议、465束和日志完整保留于research_runtime；独立V2仅在JSON编码端转换numpy标量，不改模型、拟合或选择。V2回放通过后才生成{q["ownT_mu_predictions"]}条own-T mu评价和{q["fullchain_RT_query_rows"]}条RT完整链查询输出。\n'
    text+='\n模型束为交叉验证折模型，未重新训练全5来源部署模型。运行命令、输入字段和直接API见本目录README.md；协议、逐样本输出、耗时、失败和源文件哈希均保留。独立运行守卫不是材料物理验证。\n'
    text+=r'''
## 11. 文献依据与创新的实际边界

架构中强制解析守恒已有先例，不能把R_alpha本身当作新ML算法。Beucler等在物理代理网络中比较架构/损失约束，本工作采用的守恒映射属于相关思想的具体实现；硬约束不构成任意任务误差必降定理。[Beucler等，PRL126,098302](https://arxiv.org/abs/1909.00912)。本轮缓存21页全文，重点核对正文约束构造与局部误差论证，未宣称逐项复现其补充实验。

Johnson等的STEM、DFT与DLOS讨论支持Ga2O3中补偿复合体的重要性；其阳离子间隙—双空位结构不能直接标定本轮净计量分组的alpha。相同净Ga/Sn变化不唯一对应原子几何，DLOS有限时间读出也不是唯一母类库存标签。[Johnson等，PRX9,041027](https://arxiv.org/abs/1907.00563)。本轮缓存27页并重点核对第7–9页相关讨论，没有把未复现的全部实验当作本模型的独立验证。

TEOS原文讨论Si表面富集与潜在偏析，并观察晶粒、孪晶/层错等结构差异，但未据此唯一证明当前样品中某个SiO2中性库。缺失C、非电活性化学库存或晶界输运均只是需数据检验的候选；不能为了扩大自由度编造中性相或形成能。缓存读取深度与SHA见literature/reading_registry.json。

因此当前真正已交付的是：共享库存学习与外来原子守恒、电荷IFT、同态Hall链的可训练连接；明确协议下的真实历史数据比较；可复用预测模型和失败边界。新算法独创性、能量层独有信息增益、跨元素普适性或可靠稀土排序尚未建立。

## 12. 实验怎样修正理论，并指向下一轮预测研发

第一项修正已经实施：纯原生因子不足以覆盖模型补偿，故将掺杂复合体分配接入内部状态，保留原生对照。第二项修正来自本轮误差：当前正softplus坐标对转移比例有固定斜率限制。对剂量反向关系令t=exp(−x)>0、s=s0+s1 ln(1+t)，在alpha>0处

\[
 \frac{d\ln\alpha}{d\ln C}
 =-\frac{s_1 t/(1+t)}{\ln(10)(e^s-1)}.
\]

因e^s−1≥s≥s1 ln(1+t)，且ln(1+t)≥t/(1+t)，得到

\[
 -\frac1{\ln10}\le\frac{d\ln\alpha}{d\ln C}\le0.
\]

故alpha的反向log-log斜率幅度最多约0.4343。正向关系有相同绝对上限。它只约束alpha，不是完整nH的斜率定理：参考库存、Tcap、H、电荷重排和Hall均会额外改变nH，不能据此宣称某条真实曲线不可能拟合。

下一候选是可学习log-odds分配：g=a+b log10(C/C0)、s=softplus(g)、alpha=sigmoid(g)。保持R非负与原子守恒，但dln(alpha)/dln(C)=b(1−alpha)/ln10，可学习斜率范围。两个关系参数不增加，仍须比较相同形状直接激活、输出控制、旧强参考和同标签预算；零分配是g→−infinity的极限，不能说a=b=0就恢复参考。来源log-odds平移也需新协议、完整梯度和同锚点检验，不能用本轮输出校准结果冒充已经验证。

这一候选尚未实现或训练。本轮失败也可能涉及参考能量/化学库存、热输入或输运近似，不能把斜率限制当作唯一成因。先写定新版本数学和验证协议，再以原29/5/共同18执行实际预测；理论、学习对象和参数化要随可重复训练内证据修正。继续保持ML＋Sandbox掺杂性质预测的主方向，整个goal仍为active。
'''
    dest=OUT/'ML与Sandbox_守恒库存学习与跨来源预测.md';dest.write_text(text)
    sources=[OUT/p for p in evidence_rel]+[npath,mupath,ntpath,oldpath,fullpath,p12,
        BASE/'phase15_coherent_hall/validation/temperature_summary.csv',BASE/'phase16_thermal_joint/property_evaluation/summary.csv',
        OUT/'validation/primary_by_source.csv',OUT/'property_evaluation/transport_scope_summary.csv',
        OUT/'inventory_audit/state_contributions.csv',OUT/'family_selection_protocol.json']
    energy.write(OUT/'report_claims.json',dict(code_sha256=energy.sha(Path(__file__)),
        report_sha256=energy.sha(dest),evidence_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in sources},
        numeric_claims=CLAIMS,methods=31,universal_gain_established=False,goal_complete=False,
        scope='actualdose-to-propertyprediction;all31nH/all31zeroanchorownTmu;allCSVbudgets retained;retrospective'))
    print(dest);print('numeric claims',len(CLAIMS))


if __name__=='__main__':main()
