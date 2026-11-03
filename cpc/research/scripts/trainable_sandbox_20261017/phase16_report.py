"""Assemble the complete thermal/joint prediction record, including failures."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv as read
import phase16_prediction as study
import phase2_energy as energy

OUT=study.OUT


def label(m):
    if m.startswith('mu_head_'):return label(m[len('mu_head_'):])+'＋μ均值头'
    if m.startswith('mu_control_'):
        return {'fixed205':'固定205先验','constant_logmean':'同标签常数均值','dose_temperature_line':'同标签剂量/温度线性','direct_ridge':'同标签直接Ridge'}[m[len('mu_control_'):]]
    d=study.STRUCTURES[m]
    return ('原RT' if d['reference_method']==study.REFERENCES[0] else 'E(C)')+'／'+('有效Tf' if d['thermal_mode']=='effective' else '记录约束')+f'／γ={d["gamma"]:g}'


def table(frame,methods,scope='all_query',kind=None):
    d=frame[frame.scope.eq(scope)]
    if kind is not None:d=d[d.kind.eq(kind)]
    p=d.pivot(index='method',columns='budget',values='MAE')
    out=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for m in methods:out.append('| '+label(m)+' | '+' | '.join(f'{p.loc[m,b]:.6f}' for b in (0,1,2))+' |')
    return '\n'.join(out)+'\n'


def figures(n,mu,scope):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.))
    for ax,ref,title in zip(axes,study.REFERENCES,('Original structure','Shared E(C) structure')):
        for m,d in study.STRUCTURES.items():
            if d['reference_method']!=ref:continue
            f=n[n.method.eq(m)&n.scope.eq('common18')].sort_values('budget')
            ax.plot(f.budget,f.MAE,'o'+('-' if d['thermal_mode']=='effective' else '--'),ms=4,
                label=('effective' if d['thermal_mode']=='effective' else 'record cap')+f', gamma={d["gamma"]:g}')
        ax.set(title=title,xlabel='Allowed nH specimen anchors',ylabel='Source-equal nH MAE, common18 (dex)',xticks=[0,1,2]);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'carrier_common18.{ext}',dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.8,4.2))
    for m,d in study.STRUCTURES.items():
        x=float(n[n.method.eq(m)&n.scope.eq('all_query')&n.budget.eq(0)].MAE.iloc[0])
        y=float(mu[mu.method.eq(m)&mu.scope.eq('all_query')&mu.kind.eq('RT')&mu.budget.eq(0)].MAE.iloc[0])
        color='tab:blue' if d['reference_method']==study.REFERENCES[0] else 'tab:orange'
        ax.scatter(x,y,c=color,marker='o' if d['thermal_mode']=='effective' else 's',s=40)
        ax.annotate(('O' if color=='tab:blue' else 'E')+(' eff ' if d['thermal_mode']=='effective' else ' cap ')+f'{d["gamma"]:g}',(x,y),xytext=(4,3),textcoords='offset points',fontsize=7)
    ax.set(xlabel='nH MAE, 29 specimens / 5 DOI (dex)',ylabel='muH MAE, 5 paired specimens / 2 DOI (dex)',
        title='Zero-anchor prediction trade-off: fixed ablations, no outer model choice');ax.grid(alpha=.2);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'property_tradeoff.{ext}',dpi=170)
    plt.close(fig)
    p=scope[scope.kind.eq('RT')].pivot(index='method',columns='sample_id',values='fraction_sigma_kell_lt1').reindex(list(study.STRUCTURES))
    fig,ax=plt.subplots(figsize=(8.6,5.1));im=ax.imshow(p.to_numpy(),aspect='auto',vmin=0,vmax=1,cmap='magma')
    short=[]
    for m in p.index:
        d=study.STRUCTURES[m];short.append(('Orig' if d['reference_method']==study.REFERENCES[0] else 'E(C)')+(' eff' if d['thermal_mode']=='effective' else ' cap')+f' g{d["gamma"]:g}')
    ax.set(yticks=range(len(p)),yticklabels=short,xticks=range(len(p.columns)),
        xticklabels=[s.split('::')[-1] for s in p.columns],title='Fraction of predicted conductivity weight with k*l < 1')
    fig.colorbar(im,ax=ax,label='Weight fraction; diagnostic only, no rows removed');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'transport_scope.{ext}',dpi=170)
    plt.close(fig)


def main():
    files=['thermal_audit/audit.json','numerical_gates.json','validation/contract_audit.json','research_runtime/verification.json',
        'property_evaluation/contract_audit.json','mu_controls/contract_audit.json','transport_scope/contract_audit.json',
        'parameter_scope/contract_audit.json','shared_state_head_control/contract_audit.json']
    payloads={p:json.loads((OUT/p).read_text()) for p in files}
    for p in files[1:]:assert payloads[p]['passed'],p
    a=payloads['validation/contract_audit.json'];gate=payloads['numerical_gates.json'];runtime=payloads['research_runtime/verification.json'];scopeaudit=payloads['transport_scope/contract_audit.json']
    n=read(OUT/'validation/primary_summary.csv');nt=read(OUT/'validation/temperature_summary.csv')
    mu=pd.concat([read(OUT/'property_evaluation/summary.csv'),read(OUT/'shared_state_head_control/summary.csv')],ignore_index=True)
    scope=read(OUT/'transport_scope/diagnostics.csv')
    figures(n,mu,scope)
    methods=list(study.STRUCTURES);controls=['mu_control_'+m for m in ('fixed205','constant_logmean','dose_temperature_line','direct_ridge')]
    heads=['mu_head_'+m for m,d in study.STRUCTURES.items() if d['gamma']==0]
    text=r'''# ML与Sandbox：温度输入闭合与多性质联合预测研究

## 1. 研究方向与本阶段实际交付

目标仍然是从掺杂元素、实际掺入量与明确输入出发，通过可训练ML＋Sandbox预测掺杂性质。共享学习作用于内部温度/库存与E(C)关系；nH、mu和派生sigma由同一状态读出。热历史、可识别性与数值分析用于改进这条预测链，不能代替留出预测，也不预设一定优于简单模型。本轮没有改论文，没有新增物理样品、DFT或MD。

本阶段完成29片热记录审计、新温度映射及完整梯度、载流子—迁移率联合训练、同标签简单/直接ML对照与独立预测器导出。原29个Si/Sn样品、5个DOI、0/1/2样品nH锚点、29/23/18查询及固定共同18保持不变；条件迁移率52组数据没有混入完整掺杂链。

## 2. 热输入审计：记录、代理与实际热历史

| 本组薄膜生长记录与后处理 | 来源 | 样品数 |
| --- | --- | --- |
| 本组825°C；另一个875°C系列不混用；明确无金属化后退火。 | TEOS | 3 |
| 当前所有样品850°C。1000°C/60min是沉积前衬底处理；875°C属于文中引用的其他工艺研究。 | Baldini | 8 |
| 全部830°C；接触退火470°C、N2、1min。 | Xiang | 4 |
| Fig4b仅给700–900°C范围，无逐点映射；旧700°C为下端代理。沉积前800°C处理不当作掺杂层热史；接触退火500°C、N2、1min。 | Mauze | 11 |
| TableI分别523/502/523°C。900°C/30min是真空中衬底预处理，发生在UID缓冲层和掺杂薄膜之前。 | McCandless | 3 |

18片有样品/同组温度记录，11片只有范围；完整实际峰值与冷却历史已知数为0。新情景取已报薄膜生长温度或范围上端及已报后退火的较大者作Tcap；Mauze900°C仅是条件区间上端，不是逐片温度标签。旧输入和全部样品保留。所有来源的热仪器校准、升降温轨迹及氧化学势仍不完备，因此“记录约束”不是对实际Tmax的恢复。

逐片文件、SHA及原文定位见thermal_audit/。原始论文： [TEOS](https://doi.org/10.1002/aelm.202300679)、 [Baldini](https://doi.org/10.1007/s10853-015-9693-6)、 [Xiang](https://doi.org/10.1007/s40843-022-2167-x)。

另外两篇来源为 [Mauze](https://doi.org/10.1063/5.0027870) 和 [McCandless](https://doi.org/10.1063/5.0101132)。本轮读的是已有缓存全文的相关实验段与图表；DOI直开有403/不可访问，未宣称重新取得全部在线全文。

## 3. 可训练温度律：原理、每个符号与完整导数

令a为无量纲共享学习坐标，s(a)=1/(1+exp(-a))为sigmoid，l0=log10(600K)，lc=log10(Tcap/K)。记录约束模型为

\[
 \ell_f=\ell_0+(\ell_c-\ell_0)s(a),\qquad T_f=10^{\ell_f}\mathrm K.
\]

Tf为冻结库存计算使用的条件温度。600K是声明的模型下界；Tcap是逐片显式输入。sigmoid严格在0和1之间，因此600<Tf<Tcap。共享坐标a由训练来源标签学习，未给每个测试样品另设自由温度。

\[
 \frac{\partial\ell_f}{\partial a}=(\ell_c-\ell_0)s(1-s),\qquad
 \frac{\partial\ell_f}{\partial\ell_c}=s.
\]

第一式用sigmoid导数s(1-s)得到；第二式把a固定后对输入边界求导。旧冻结模型用l1=log10(1950K)和旧坐标ã。定义q=(lf-l0)/(l1-l0)、ã=ln(q/(1-q))，可准确复用旧状态求解器：

\[
 \frac{\partial\tilde a}{\partial a}=
 \frac{(\ell_c-\ell_0)s(1-s)}{(\ell_1-\ell_0)q(1-q)},\qquad
 \frac{\partial\tilde a}{\partial\ell_c}=
 \frac{s}{(\ell_1-\ell_0)q(1-q)}.
\]

令隐式状态方程为F(x,theta)=0，x含高温费米能级/化学势与测量温度电荷状态，theta为共享参数。对方程微分得Fx dx+Ftheta dtheta=0；在Fx可逆时有dx/dtheta=−Fx^{-1}Ftheta。旧求解器的导数包含高温库存、冷却后电荷再平衡、屏蔽和Hall读出，乘以上述坐标导数得到新链。仅修改输出公式或漏掉库存对温度的导数都会错误。effective参考分支准确复用阶段15，不改能量和FD表；Tcap=1950K时新映射和梯度恢复旧映射。

## 4. 同一状态的nH、mu、sigma与联合损失

固定统计模型中，n为自由电子浓度，Nq²为每类电荷数q的平方加权缺陷浓度之和再加背景带电中心。自由电子屏蔽由自由能带的压缩率计算。离子散射率和固定205cm²/Vs对应的常数tau额外率相加，能量积分给出漂移迁移率mu_d与Hall因子rH。读出满足nH=n/rH、muH=rH mu_d及

\[
 \sigma=e n\mu_d=e n_H\mu_H.
\]

e是正元电荷，浓度以cm^-3、迁移率以cm²/Vs代入时sigma单位为S/cm。该恒等式只保证读出相容，不能验证散射机制。205是历史经验先验，不是新算出的声子参数。单标量能带、自由电子屏蔽、独立点电荷Born/RTA近似仍需适用域限制。

每条性质在自己的观测温度T求解；mu输入不含实测nH。原RT结构保持原来的nRT监督；E(C)结构保持nRT+.1 nT监督。预设gamma0/.1/1只改变mu监督：

\[
 L=L_{n,RT}+\beta_nL_{n,T}+\gamma(L_{\mu,RT}+.1L_{\mu,T}),
 \quad \beta_n=0\ (\mathrm{RT}),\ .1\ (E(C)).
\]

L的每个块依次对同片读数、同DOI样品和训练DOI平均平方log10误差。所有温度曲线训练限T≤275K。现有muRT只有5配对/2DOI，muT只有54读数/3旧片/1DOI，nT仍为7旧片/2DOI/142读数。无标签的块不补造观测；一个3来源上下文排除两个mu来源，8个非零gamma结构在该上下文没有mu训练标签。未知跨端点/读数协方差，因此这是复合预测目标，不能称为独立高斯似然或微观后验。

## 5. 固定验证设计与计算记录

两种基础结构×两种温度律×三个gamma共12结构。effective/gamma0复用30个旧拟合；10新增结构在15个3/4来源上下文训练150次，两个原始坐标起点共300新起点、120次求值上限、1e-7停止阈值。限制温度不增加参数；原结构仍2个共享参数，E(C)仍4个。

外层整DOI排除其所有n/mu标签；内层用训练来源nH查询选原7个输出偏移lambda。gamma逐项固定报告，不按外层误差选“最优gamma”。测试来源只允许预定的0/1/2个nH样品锚点，没有mu锚点。mu同标签对照包含固定205、常数对数均值、剂量/温度线性与直接Ridge；Ridge标准化和alpha选择只用训练来源。没有mu训练标签时明确回退205。
'''
    text+=f'\n实际最终记录{a["final_fit_records"]}个，新起点{a["actual_new_optimizer_starts"]}个；需要统一数值复核的初轮拟合{a["V1_failed_start_fits"]}个。独立复算{a["inner_scores"]}个内层分数，主查询缺失{a["missing_primary_queries"]}。停止成功不等于已证明全局最优或材料模型正确。\n'
    text+=r'''
## 6. 数学与运行门槛，包括初次失败

初次945项门槛中有76项未通过，记录和当时代码保留，训练没有在失败门槛下启动。强补偿低自由载流子状态中，2e-5两点差分把浮点舍入放大；代表性状态的步长扫描和冷/暖启动误差见numerical_precision_diagnostic.json。改用四阶Richardson差分解析核验，没有修改物理公式、解析导数或3e-6误差门槛。它是数值求导验证方案修订，不是新的物理实验。
'''
    text+=f'\n最终{len(gate["checks"])}项通过，最大缩放导数差{gate["maximum_scaled_derivative_error"]:.3g}。参数、logTcap输入、stock/Tf/offset来源链、旧参考/零更新、损失权重和实际联合目标梯度均有核验。\n'
    text+=r'''
## 7. 载流子主终点：全部结果

以下为5个DOI等权MAE，单位dex。第一表使用每个预算的全部查询，第二表固定同一共同18样品；不能用不同查询组成的均值变化当作适应收益。
'''
    text+='\n'+table(n,methods)+'\n### 固定共同18\n\n'+table(n,methods,'common18')
    text+='\n![共同18上的全部温度律与gamma对照](figures/carrier_common18.png)\n'
    old=read(energy.BASE/'phase15_coherent_hall/validation/primary_summary.csv')
    selected=old[old.scope.eq('all_query')&old.method.isin(['original_RT','direct_ridge_RT','direct_ridge_joint','dose_line_RT'])]
    p=selected.pivot(index='method',columns='budget',values='MAE')
    text+='\n### 保留的旧对照\n\n| 旧方法 | 0锚点 | 1锚点 | 2锚点 |\n| --- | --- | --- | --- |\n'
    for method in p.index:text+='| '+method+' | '+' | '.join(f'{p.loc[method,b]:.6f}' for b in (0,1,2))+' |\n'
    text+='\n阶段12常数库存＋内部库存适应在1锚点达到0.229552dex，2锚点常数库存＋输出适应达到0.250297dex，均继续保留。旧26基础模型与旧来源适应全表在前阶段结果，不因新增结构而删除。\n'
    text+='\n记录约束的全部90个拟合均接触logitTf=6和log10背景=18上界；E(C)的屏蔽斜率也均接触5上界。348个外层RT预测状态有18个低于模型声明的1e13cm^-3警示线，均保留。这是给定参数域和输入情景的失配证据，不是无噪声全局不可达定理或唯一失效机制证明。\n'
    text+='\n## 8. 完整链迁移率与派生电导率\n\n以下muRT以2个DOI等权，只有5/4/2个预算查询；无实测nH输入，也无mu锚点。所有12物理模型及mu标签匹配对照都报告。简单mu对照自身不使用额外n标签，因此它们不等于共享全部n/mu信息的多任务ML。为补这个比较边界，在主结果出来后另加4个“同一个仅n训练状态＋一参数mu均值头”诊断：保留完全相同n拟合，使用相同mu训练标签，只做闭式输出校准，不更新物理库存。该后加诊断有独立协议，不能称为预注册或盲测。sigma只从已有RT nH/mu配对派生，不是第三类独立验证标签。\n\n'+table(mu,methods+controls+heads,kind='RT')
    text+='\n![零锚点载流子与迁移率预测权衡；两坐标队列不同，不能直接当统一损失](figures/property_tradeoff.png)\n'
    text+='\n## 9. 变温迁移率：相同旧片，在自身温度预测\n\n54个读数来自McCandless的3片/1DOI，外层预测整组排除了该DOI所有n/mu标签；训练中看不到该来源mu(T)。按样品内读数平均后取来源均值。RT锚点预算只改变查询样品组成，mu自身不作来源偏移；固定共同曲线仍为同一片24个读数。\n\n'+table(mu,methods+controls+heads,kind='curve')
    text+='\n### 固定共同曲线\n\n'+table(mu,methods+controls+heads,'common_curve','curve')
    text+='\n## 10. 输运适用域：预测误差与物理自洽必须分别报告\n\n抛物线能带中k=sqrt(2mE)/hbar、v=sqrt(2E/m)、平均自由程l=v tau，因此kl=2E tau/hbar。E是相对带底动能、m是有效质量、hbar是约化普朗克常数、tau是该能量下弛豫时间。若承担主要电导的能量区间kl很小，现有弱散射半经典图像缺少自洽支持。我们用同一能量积分计算kl<1区域对电导和Hall矩的权重比例；1仅是声明的诊断刻度，不是所有材料普适的严格电导下界。该边界问题也见[Gunnarsson等综述摘要](https://arxiv.org/abs/cond-mat/0305412)，本轮未把摘要检索称为全文系统综述。\n'
    text+=f'\n检查了{scopeaudit["states"]}个预算0的实际mu预测状态，其中{scopeaudit["states_majority_sigma_kell_below1"]}个状态的电导权重有超过一半来自kl<1区域。512节点复算mu与256节点读出最大差{scopeaudit["maximum_moment_replay_error"]:.3g}dex。没有删点或改变评分。保存的physical_valid旧字段仅是求解域标记，不是Born机制已验证；数值相容、预测误差和机制适用域不可混为一谈。\n'
    text+='\n![五个RT配对样品的输运自洽诊断，全部模型保留](figures/transport_scope.png)\n'
    text+='\n## 11. 独立使用与复现\n'
    text+=f'\n导出{runtime["saved_bundles"]}个整DOI排除的模型束，在禁止打开实验表、拟合表、选择表和热输入审计表的子进程复算{runtime["independent_rows"]}行。最大状态差{runtime["maximum_state_error"]:.3g}dex、输运差{runtime["maximum_transport_error"]:.3g}dex、nH锚点来源坐标差{runtime["maximum_source_shift_error"]:.3g}dex。任务文件只有物理输入与授权nH锚点，没有查询nH或mu标签。模型束是交叉验证折模型，不是重新训练的全5来源部署模型。\n'
    text+='\n运行命令与直接API见本目录README.md；全部拟合/来源选择/逐样品预测、输入SHA和失败记录都保留。最终阶段16清单同时验证前15阶段哈希。本轮采用CPU4，未使用GPU；GPU1授权仍保留，GPU0未使用。\n'
    text+='\n## 12. 结果如何改变下一步预测模型研发\n\n'+(OUT/'interpretation_zh.md').read_text()
    dest=OUT/'ML与Sandbox_温度闭合与联合性质预测.md';dest.write_text(text)
    evidence=[OUT/p for p in files]+[OUT/'validation/primary_summary.csv',OUT/'validation/temperature_summary.csv',
        OUT/'property_evaluation/summary.csv',OUT/'transport_scope/diagnostics.csv',OUT/'interpretation_zh.md',
        OUT/'shared_state_head_control/summary.csv',OUT/'parameter_scope/summary.csv',
        energy.BASE/'phase15_coherent_hall/validation/primary_summary.csv',energy.BASE/'phase12_stock_adaptation/physical_scope_v2/summary.csv']
    energy.write(OUT/'report_claims.json',dict(code_sha256=energy.sha(Path(__file__)),
        evidence_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in evidence},
        scope='allfixedablationresults;retrospective actualdose-to-propertyprediction,not puremechanismreport or successful universalmethod',
        primary='unchanged29/5/0-1-2/common18',independent_sigma_labels=0,goal_complete=False))
    print(dest)


if __name__=='__main__':main()
