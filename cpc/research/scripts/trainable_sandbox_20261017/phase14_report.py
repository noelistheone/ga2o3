"""Record joint internal-statistics prediction and its transport unit gate."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from csv_io_v2 import read_result_csv as read
import phase2_energy as energy
import phase14_refinement as ref

OUT=ref.ORIGINAL;FINAL=ref.OUT/'validation';SOURCE=OUT/'source_adaptation/validation'
LABELS={
    'original_RT':'原目录RT', 'original_joint':'原目录联合',
    'localized_fixed_joint':'永久局域＋固定绑定能', 'binding_screen_joint':'原共享绑定能E(C)',
    'fraction_constant_RT':'常数局域比例RT', 'fraction_dose_RT':'共享局域比例RT',
    'fraction_constant_joint':'常数局域比例联合', 'fraction_dose_joint':'共享局域比例联合',
    'fraction_binding_joint':'共享比例＋常数绑定能',
    'output_linear_RT':'同容量输出RT', 'output_linear_joint':'同容量输出联合',
    'direct_ridge_RT':'直接Ridge RT', 'direct_ridge_joint':'直接Ridge联合',
    'dose_line_RT':'剂量直线', 'activation_prior_RT':'中位激活比',
    'identity_unbounded':'n=C，无收缩偏移', 'identity_H_unbounded':'n=C+已报H，无收缩偏移',
    'fraction_screen_constant_joint':'新：E(C)＋常数比例',
    'fraction_screen_joint':'新：E(C)＋共享比例',
    'output_screen_joint':'新：E(C)＋同容量输出',
}
VARIANTS={
    'original_output':'原目录：输出适应','original_stock':'原目录：库存适应','original_Tf':'原目录：Tf适应',
    'binding_output':'E(C)：输出适应','binding_stock':'E(C)：库存适应','binding_Tf':'E(C)：Tf适应',
    'joint_output':'联合：输出适应','joint_stock':'联合：库存适应','joint_Tf':'联合：Tf适应',
    'joint_localized':'联合：局域比例适应',
}


def table(frame,scope='all_query',value='MAE',labels=LABELS,column='method'):
    d=frame if scope is None else frame[frame.scope.eq(scope)]
    p=d.pivot(index=column,columns='budget',values=value)
    lines=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for method,label in labels.items():
        lines.append('| '+label+' | '+' | '.join(f'{p.loc[method,b]:.6f}' for b in (0,1,2))+' |')
    return '\n'.join(lines)+'\n'


def figures(primary,temperature,source):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    methods={'original_RT':'Original RT','binding_screen_joint':'E(C) parent',
        'fraction_binding_joint':'Fraction + constant E',
        'fraction_screen_joint':'E(C) + fraction','output_screen_joint':'E(C) + matched output'}
    fig,axs=plt.subplots(1,3,figsize=(12.8,3.5))
    for ax,d,scope,title in zip(axs,(primary,primary,temperature),('all_query','common18','common40_125'),
        ('RT: all queries','RT: fixed common18','Same 7 Si specimens: temperature')):
        for method,label in methods.items():
            a=d[d.method.eq(method)&d.scope.eq(scope)].sort_values('budget')
            ax.plot(a.budget,a.MAE,'o-',label=label,ms=3,lw=1)
        ax.set(xlabel='Allowed specimen anchors',ylabel='Source-equal MAE (dex)',xticks=[0,1,2],title=title)
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=6.5);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'joint_prediction.{ext}',dpi=190)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11.8,3.6))
    chosen={'original_output':'Original: output','original_Tf':'Original: Tf','joint_output':'Joint: output',
        'joint_localized':'Joint: localized','joint_stock':'Joint: stock'}
    for ax,scope,title in zip(axs,('all_query','common18'),('29/23/18 queries','Same common18')):
        for method,label in chosen.items():
            a=source[source.variant.eq(method)&source.scope.eq(scope)].sort_values('budget')
            ax.plot(a.budget,a.MAE,'o-',label=label,ms=4,lw=1)
        ax.set(xlabel='Allowed specimen anchors',ylabel='Source-equal MAE (dex)',xticks=[0,1,2],title=title)
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'source_adaptation.{ext}',dpi=190)
    plt.close(fig)
    bank=ref.load_composite()
    choices=json.loads((ref.OUT/'methods/fraction_screen_joint/selections.json').read_text())
    c=np.linspace(15.8,20.2,250);fig,axs=plt.subplots(1,2,figsize=(11.8,3.6))
    for choice in choices:
        if choice['budget']:continue
        record=bank[choice['fit_key']];w=dict(zip(record['parameter_names'],record['parameters']))
        # The saved parameter names are the authoritative parameter convention.
        ell=expit(w['localized_logit']-w['localized_slope']*(c-18))
        ed=10**w['log10_binding_E0']*np.exp(-w['binding_screen_s']*10**((c-18)/3))
        axs[0].plot(c,ell,label=choice['held'],lw=1)
        axs[1].plot(c,ed*1000,lw=1)
    axs[0].set(ylabel='Conditional localized fraction',ylim=[-.02,1.02])
    axs[1].set(ylabel='Conditional binding energy (meV)')
    for ax in axs:
        ax.set(xlabel=r'$\log_{10}(C/\mathrm{cm}^{-3})$');ax.grid(alpha=.2)
    axs[0].legend(fontsize=6);fig.suptitle('Five source-excluded learned relations; no microscopic truth labels')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'learned_relations.{ext}',dpi=190)
    plt.close(fig)


def sensitivity():
    differences=[];changed=0
    for method in LABELS:
        a=json.loads((OUT/'methods'/method/'selections.json').read_text())
        b=json.loads((ref.OUT/'methods'/method/'selections.json').read_text())
        for x,y in zip(a,b):
            changed+=x['selected']['lam']!=y['selected']['lam'] or x['selected']['config']!=y['selected']['config']
        a=read(OUT/'methods'/method/'primary_predictions.csv')
        b=read(ref.OUT/'methods'/method/'primary_predictions.csv')
        assert a[['sample_id','budget']].equals(b[['sample_id','budget']])
        differences.extend(abs(a.predicted_log_nH-b.predicted_log_nH))
    result=dict(passed=True,changed_outer_choices=int(changed),changed_prediction_rows=int(np.sum(np.array(differences)>1e-10)),
        maximum_prediction_change=float(max(differences)),criterion='all failed-start records,not testscoreselection')
    energy.write(OUT/'selection_sensitivity.json',result);return result


def main():
    base=json.loads((FINAL/'contract_audit.json').read_text())
    src=json.loads((SOURCE/'contract_audit.json').read_text())
    runtime=json.loads((OUT/'research_runtime/verification.json').read_text())
    charge=json.loads((OUT/'research_runtime/charge_budget_audit.json').read_text())
    transport=json.loads((OUT/'transport_audit/gates.json').read_text())
    gates=json.loads((OUT/'model_gates.json').read_text())
    assert all(j['passed'] for j in (base,src,runtime,charge,transport,gates))
    primary=read(FINAL/'primary_summary.csv');temperature=read(FINAL/'temperature_summary.csv')
    source=read(SOURCE/'summary.csv');chains=read(OUT/'research_runtime/full_chain_summary.csv')
    figures(primary,temperature,source);sense=sensitivity()
    def score(frame,m,b,scope='all_query',col='method',metric='MAE'):
        a=frame[(frame[col]==m)&frame.budget.eq(b)]
        if scope is not None:a=a[a.scope.eq(scope)]
        assert len(a)==1
        return float(a.iloc[0][metric])
    claims={
        'joint_allquery':[score(primary,'fraction_screen_joint',b) for b in (0,1,2)],
        'original_allquery':[score(primary,'original_RT',b) for b in (0,1,2)],
        'joint_localized_allquery':[score(source,'joint_localized',b,col='variant') for b in (0,1,2)],
        'original_Tf_allquery':[score(source,'original_Tf',b,col='variant') for b in (0,1,2)],
        'binding_temperature_common80':[score(temperature,'binding_screen_joint',b,'common40_125') for b in (0,1,2)],
        'joint_temperature_common80':[score(temperature,'fraction_screen_joint',b,'common40_125') for b in (0,1,2)],
    }
    texts=[
        '# ML＋Sandbox：联合内部统计、来源适应与性质预测核验\n',
        '本阶段持续推进用户要求的“ML＋Sandbox预测掺杂性质”。载流子是首个主终点，完整迁移率、电导率和有依据的候选排序仍在目标内。数学、机制与可识别性分析用于改进预测；本报告不是最终goal完成声明。GPU1授权有效，本轮使用CPU，没有新DFT、MD、样品或仪器实验。\n',
        '## 1. 目标检查与本阶段结论\n',
        '系统goal保持active，要求实际训练进入物理内部的共享关系、保存后复用于新输入，并在相同来源、锚点及覆盖集上与固定物理、常数校准、分层经验和直接ML比较。它没有退回只解释测量机制的研究。完整掺杂性质链仍需继续改进，不能以数值门槛或报告数量验收。\n',
        '本轮新增3种联合结构、45个拟合及10种来源适应对照。联合E(C)与局域比例没有带来跨预算稳定优势；同一联合模型的局域比例内部适应在两锚点有条件改善，但未超过原模型Tf适应及此前强常数库存对照。独立模型回放通过；新输运参考的SI/FD核验也通过，但尚未接入这些性质预测。以下完整保留有利和不利结果。\n',
        '## 2. ML具体学什么，Sandbox具体解什么\n',
        r'给定实际化学浓度\(C\)，共同学习绑定能\(E_D\)与局域子群比例\(\ell\)：'+'\n',
        r'''\[
E_D(C)=E_0\exp\!\left[-s\left(\frac{C}{10^{18}\,\mathrm{cm}^{-3}}\right)^{1/3}\right],\qquad
\ell(C)=\operatorname{sigmoid}\!\left[a-b\log_{10}\!\left(\frac{C}{10^{18}\,\mathrm{cm}^{-3}}\right)\right].
\]
''',
        r'\(E_0\)单位eV；\(s\)为无量纲衰减强度；\(a\)是参考浓度的比例logit，\(b\ge0\)是每dex的logit斜率。它们是跨训练条件共享的有效关系，不是新的HSE能量标签、实测Mott阈值或直接观测的杂质带人口。sigmoid把比例限制在0与1之间；训练范围a∈[-12,12]、b∈[0,12]不等于严格包含端点。旧固定ell=0/1另列为候选。'+'\n',
        r'''\[
x=\ln2+\frac{E_F-E_C+E_D(C)}{k_BT},\quad
f_0=\frac1{1+e^{-x}},\quad
A_{\rm eff}=G_+-\ell(C)k_BT\ln(1+e^x).
\]
''',
        r'\(E_F\)是电子化学势，\(E_C\)是导带边，\(k_B\)为eV/K单位的Boltzmann常数，\(T\)为K，\(G_+\)是含+1态参考及库存乘子的形成自由能；\(f_0\)是局域子群的中性占据。固定输入子群自由能用于高温库存及冷态电荷再平衡。冷热求解同时满足声明的化学供应与电荷守恒，完整IFT把学习权重对性质损失的导数传回训练。'+'\n',
        '新增模型为：E(C)＋常数比例5参数、E(C)＋浓度相关比例6参数、E(C)＋同容量输出6参数；父E(C)模型4参数。另17个冻结方法全部保留。两个关系的浓度导数和全部实际训练参数链已通过381项数值检查，最大归一化差分误差7.58e-9。这是实现门槛，不能解释为381项材料实验。\n',
        '## 3. 数据、训练及来源隔离\n',
        '主任务仍为29片Si/Sn、5个DOI来源，0/1/2每来源样品锚点，29/23/18查询及共同18查询。外层整DOI排除；其余来源内再排除DOI选择收缩。MAE先来源内平均，再5来源等权，单位dex。五DOI不是五个独立实验室。历史队列已多轮暴露，这是回顾性研发，不称盲测或预注册。\n',
        '温度补充仍是原7片Si、2DOI、142个n读数，没有新增独立样品。联合训练损失为来源等权RT平方误差＋0.1倍来源/样品/读数等权的275K以下曲线误差。RT选出的lambda直接迁移到原温度锚点/guard，共同40—125K为80读数；不能和温度专门选参的阶段10结果混成纯结构消融。\n',
        '所有物理模型使用实际Si/Sn、已报H、显式材料profile及pO2=1e-5atm情景。29条主记录无实测pO2；部分生长温度是来源代理。C缺陷能量不在KROGER目录，未补造。局域比例和绑定能关系只用化学浓度，不能说生长温度、衬底、缺失形貌等已被这些关系学会。\n',
        '15个3/4来源上下文新增45物理拟合、90起点；全部两个声明起点、每起点120评估、ftol/xtol/gtol为1e-7、相同IFT及参数范围。对照405记录复用确切路径/哈希；加入新模型后450记录、210物理拟合。直接模型有不同实际成本，候选上限相同不等于耗时相同。逐起点与求解成本表均保存。\n',
        '## 4. 全部20种基础结构：原主任务\n',
        '下表是各预算全部查询的来源等权MAE（dex）。保持全部简单和直接学习对照，不能只挑某个最优单格。\n',table(primary),
        '联合E(C)＋共享比例为0.271963/0.264131/0.279582，原RT为0.266289/0.251591/0.257810，三个预算都未超过原RT。同容量输出更差，只能说明所测内部关系优于这一输出参数化，不能据此宣称整体方法战胜全部基线。常数比例几乎退回原E(C)，新增自由度未稳定产生增量。\n',
        '## 5. 固定查询与温度信号\n',
        '共同18查询固定每个预算的评估对象，避免锚点增加后因查询组成改变而误读收益。全部20方法如下。\n',table(primary,'common18'),
        '![同一主任务与温度补充的结构比较；误差越小越好](figures/joint_prediction.png)\n',
        '温度共同80读数的全部方法见下一表，按样品再来源等权。新增E(C)＋共享比例0锚点0.502636，比前轮“比例＋常数绑定能”0.635130有所改善，但仍不及原E(C)的0.308611。新增常数比例为0.308579，与父模型几乎相同；同容量输出0.304781的小幅差别未转化为室温主任务优势。不能预设联合两个有效关系一定更好。\n',table(temperature,'common40_125'),
        '![各排除来源保存的关系；不同曲线不是置信带，也不是真实微观参数标签](figures/learned_relations.png)\n',
        '## 6. 来源内部适应：相同锚点的10组对照\n',
        '冻结基础共享参数，原RT、父E(C)、联合模型各比较输出/库存/Tf坐标，联合再比较局域比例logit坐标。来源标量z只使用允许的0/1/2样品锚点。输出为logn−z，不重求状态；库存、Tf、局域坐标重求冷热守恒状态。内部z范围[-6,6]，物理映射及惩罚单位预先固定，不能将同一lambda数值叫作相同微观先验。\n',
        r'''\[
J(z)=\sum_{j\in\mathcal A}[\log_{10}n_H(x_j;z)-y_j]^2+\lambda z^2,
\qquad \lambda\in\{0,.001,.01,.1,1,10,\infty\}.
\]
''',
        'A是合法锚点集合，y是锚点log10 nH。输出用闭式解，内部用25节点及逐局部极小精化，内层整来源选择lambda。该算法不证明连续全局最优；局部I/(I+lambda)指标不是微观后验置信区间。250个来源情境、54714个标量候选目标和1800项闭式核对通过；3000项内层评分、870行保存输出、所有查询均有输出。\n',
        '下表为全部查询的来源等权MAE（dex），零锚点对应各自基础结构，不混为同一个起点。\n',table(source,labels=VARIANTS,column='variant'),
        '同一联合模型两锚点局域适应0.259271优于输出0.279582，但仍未超过原RT的Tf适应0.255610，以及阶段12常数库存输出的0.250297。阶段12常数库存在一锚点为0.229552仍保留；其基础训练/优化设置不同，作为已知强对照，不称本轮纯结构消融。\n',
        '局域两锚点适应的5个来源中，两个lambda无穷大而完全不更新；Baldini选择无惩罚z约2.89，Xiang选择无惩罚并触碰z=-6边界，McCandless选择lambda=.1、z约.202。改善并不证明来源存在已测到的真实局域比例差异。边界、无更新及失配均保存。\n',
        '固定共同18的10种来源对照如下；图同时显示两个查询协议。\n',table(source,'common18',labels=VARIANTS,column='variant'),
        '![共享模型固定后比较来源内部与输出连接；无跨预算稳定统治关系](figures/source_adaptation.png)\n',
        '## 7. 收敛、数值风险及原版保留\n',
        '新90起点仅一个三来源联合拟合有未收敛起点，120评估时目标0.02450688165。按“所有失败起点记录”标准统一360复核，本次只涉及该1记录；两个原始起点加保存最佳点，共3新起点，全部停止成功，目标降至0.02450425085。不按外层误差选择复核对象。\n',
        f'阶段14实际新增93起点，继承实际333，总实际426；当前复合记录含422起点。当前最佳/外层选中未收敛者均0，但停止标志不是全局最优证明。V1完全保留；外层配置/lambda改变{sense["changed_outer_choices"]}项，最大预测变化{sense["maximum_prediction_change"]:.3g}dex。\n',
        f'独立审计重建8520项内层评分、1740主输出与8940温度输出，最大训练目标差{base["maximum_objective_reconstruction_error"]:.3g}，经验重建差{base["maximum_empirical_reconstruction_error"]:.3g}；原锚点/查询/共同18和来源排除通过。经验输出与物理状态分开标志，禁止用校准后的n冒充重新求出的中性状态。\n',
        '输运门槛初次运行因MaterialProfile属性Tref写错而在后段中断；原脚本和失败日志保存在development_runs/initial_attribute_error，修正为实际T_ref后重新运行并通过。该错误不改变训练模型或已有CT结果。数字化、来源与求解失败不从历史记录中删除。\n',
        '## 8. 独立调用与完整性质链\n',
        f'导出315个保存包：14物理基础结构＋7内部适应变体，各5来源×3预算。独立子进程只读保存关系、物理输入与合法锚点，禁读性质表、拟合/选择/验证表。重新求状态及内部来源校准，1827条载流子最大差{runtime["maximum_carrier_error"]:.3g}dex，来源坐标差{runtime["maximum_source_coordinate_error"]:.3g}，电荷预算最大差{charge["maximum_charge_budget_error"]:.3g}dex。它证明学到的关系可以复用，不是查表返回答案，也不是新材料验证。\n',
        f'231条配对mu/sigma回放最大差{runtime["maximum_transport_error"]:.3g}dex。全部20基础＋7额外内部变体的完整链297行，仍仅5条化学/Hall/mu明确配对、2DOI；0/1/2预算相应5/4/2查询。52条件对/11引用组的源外CT头训练与从化学输入预测mu的任务不同，不能冒充52条完整链。\n',
        '全部完整链的mu来源等权MAE如下。来源适应的输出基线已由相应基础方法覆盖；其余7种额外内部变体列在表末。\n',table(chains,scope=None,value='mu_MAE',labels={**LABELS,**{k:v for k,v in VARIANTS.items() if not k.endswith('output')}}),
        'sigma=e nH muH是同一Hall量派生恒等式，没有独立sigma标签，分量误差可能抵消。当前CT头零锚点的表达范围误差下界仍为1.234605dex，原链实际1.332676，不能只改善n就宣称完整输运已解决。研究包不是全五来源重训部署模型，尚无新稀土可靠排序。\n',
        '## 9. 新输运参考：数学核验已完成，性质验证尚未完成\n',
        '为继续改进mu及nH的一致预测，新增独立ionized_transport.py。旧transport.py与全部冻结结果不改。假设单各向同性抛物带、独立静态点电荷、弹性Born散射及弛豫时间近似；同一声明质量用于DOS、速度和散射。没有声子、晶界、杂质带或多通道材料验证。\n',
        r'''\[
\tau^{-1}_{II}(E)=\frac{N_{q^2}e^4}{16\sqrt2\pi\epsilon^2\sqrt m\,E^{3/2}}
\left[\ln(1+b)-\frac b{1+b}\right],\quad
b=\frac{8mE}{\hbar^2q_s^2},\quad q_s^2=\frac{e^2 n_{\rm screen}}{\epsilon k_BT}.
\]
''',
        r'此式所有能量为J、密度为m^-3、\(m\)为kg、\(\epsilon=\epsilon_0\epsilon_r\)为F/m，\(e\)为C，\(\tau^{-1}\)为s^-1；接口密度cm^-3显式乘1e6。\(N_{q^2}=\sum_j N_j Z_j^2\)是实际平方电荷预算，不能用绝对电荷或响应chi替代。\(n_{\rm screen}=k_BT\,\partial n/\partial\mu_J\)来自自由电子压缩率，不是总化学浓度。'+'\n',
        r'''\[
I_j=\int_0^\infty \tau(E)^j E^{3/2}\left(-\frac{\partial f}{\partial E}\right)dE,
\quad \mu_d=\frac em\frac{I_1}{I_0},\quad
r_H=\frac{I_2 I_0}{I_1^2},\quad n_H=\frac n{r_H},\quad \mu_H=r_H\mu_d.
\]
''',
        r'\(f\)为电子Fermi函数，\(j=0,1,2\)是矩阶，\(\mu_d\)为漂移迁移率，\(r_H\)为无量纲Hall因子，\(n_H,\mu_H\)为相容Hall读出。常数库仑对数的非简并极限给前因子\(128\sqrt{2\pi}\)，旧BH函数在同一SI约定下比它少\(16\pi^2\approx157.914\)。这不是把所有真实mu误差乘158即可修复；代表能量b系数8/24也必须单独区分。'+'\n',
        '63项门槛通过：独立角积分、FGR绝对前因子、匹配L的9组旧函数对照、Maxwell Gamma/Hall极限、FD屏蔽导数、正矩/Cauchy、独立DOS电导积分、密度缩放及尾积分收敛。原published Nc=3.33e18在单标量带对应DOS质量0.26016457m0；旧输运0.28m0对应Nc=3.717997e18。真实各向异性中不同质量可有物理意义；单各向同性模型必须显式统一或解释，不能悄悄混用。\n',
        '[RSC作者补充材料S1](https://www.rsc.org/suppdata/d0/nr/d0nr03432k/d0nr03432k1.pdf)用于SI系数核对，其材料为MAPbI3、b取24，不是Ga验证。[Neal2018](https://doi.org/10.1063/1.5034474)是Ga载流子与含Hall输运的相关先例；本轮没有复现其作者拟合。完整推导保存在ML_SANDBOX_TRANSPORT_UNIT_DERIVATION_20261017.md。\n',
        '## 10. 下一项改进与goal的真实完成边界\n',
        '不再无依据扩大同一浓度函数的容量。下一项是把经过核验的FD统计、自由电子屏蔽、平方电荷预算和Hall矩接到同一冷热Sandbox状态，保留并重新训练跨条件共享关系。至少区分“旧统计+rH1”“一致FD+rH1”“同一FD+Hall矩”三组，使统计修正与观测读出修正各有对照。\n',
        '改变rH必须同步改变原nH=n/rH损失并重做29/5、0/1/2和共同18，不能保留旧载流子成绩而只替换mu定义。新增通道应先合并能量依赖速率，再算Hall矩；常数tau有效通道不能命名已求解的Fröhlich声子。插值和梯度必须与实际函数一致，并通过冷热IFT与读出全链核验后再训练。该耦合尚未实现或取得新的材料精度。\n',
        '低mu薄膜可能超出均匀带输运假设，但不能删难例后宣称全队列准确，也不能由一次拟合唯一归因晶界、接触或局域输运。新框架必须保存适用性标志、失败及完整强对照。完整mu、独立来源迁移和新掺杂排序仍未解决；goal继续active，不能由本阶段报告代替最终预测工具。\n',
        '## 11. 数学补充：自由能一致性与可微训练\n',
        '在给定C时，ell与E_D是外部条件函数；对EF求导时它们固定。先对softplus求导，再用+1形成自由能的EF斜率，逐步得到：\n',
        r'''\[
\frac{\partial A_{\rm eff}}{\partial E_F}=1-\ell f_0=\bar q,\quad
\frac{\partial^2 A_{\rm eff}}{\partial E_F^2}
=-\frac{\ell f_0(1-f_0)}{k_BT}=-\frac\chi{k_BT}.
\]
''',
        r'固定子群响应为\(\chi=\ell f_0(1-f_0)\ge0\)。对+1/0子群实际\(\overline{q^2}=\bar q\)，而非chi。热态库存\(c=N\exp[-A_{\rm eff}/(k_BT)]\)随EF也变化，链式求导得到：'+'\n',
        r'''\[
\frac{\partial(c\bar q)}{\partial E_F}
=-\frac c{k_BT}(\bar q^2+\chi)\le0.
\]
''',
        r'冷态固定库存时只剩\(-c\chi/(k_BT)\)。加上电子正压缩率与空穴项，在声明的条件下电荷函数随EF严格下降；但联合库存乘子的可微性还需整个根雅可比可逆。若守恒方程为\(R(v,\theta;C)=0\)，则：'+'\n',
        r'''\[
\frac{dv}{d\theta}=-R_v^{-1}R_\theta,\qquad
\frac{dy}{d\theta}=y_\theta-y_vR_v^{-1}R_\theta.
\]
''',
        r'\(v\)含实际求得的化学势与库存乘子，\(\theta\)是保存的共享参数；\(R_v,R_\theta\)为相应偏导。对外部剂量C求导时还必须包含ell与E_D的两条浓度链。381项检查验证实际采用的映射/来源/输入导数，不是用形式上写出IFT代替验证。参数或关系近共线仍可能导致有限数据下不可识别，预测留出结果因此单独报告。'+'\n',
    ]
    target=OUT/'ML与Sandbox_联合统计与来源适应预测验证.md';target.write_text('\n'.join(texts))
    energy.write(OUT/'report_claims.json',dict(created_utc=energy.now(),claims=claims,
        evidence_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (FINAL/'primary_summary.csv',FINAL/'temperature_summary.csv',
            SOURCE/'summary.csv',FINAL/'contract_audit.json',SOURCE/'contract_audit.json',OUT/'research_runtime/verification.json',
            OUT/'research_runtime/charge_budget_audit.json',OUT/'transport_audit/gates.json')},
        methods=20,source_variants=10,new_fits=45,actual_new_starts=93,actual_total_starts=426,composite_starts=422,
        main_gradient_checks=381,transport_reference_checks=63,independent_bundles=315,
        stable_primary_increment_established=False,new_transport_used_for_predictions=False,goal_complete=False))
    print(target)


if __name__=='__main__':main()
