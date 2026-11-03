"""Evidence report for learned defects, coherent Hall and source adaptation."""
from pathlib import Path
import sys,json,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv as read
import phase15_prediction as study
import phase2_energy as energy

OUT=study.OUT;FINAL=OUT/'validation';SOURCE=OUT/'source_adaptation/validation'
LABELS={
 'original_RT':'原目录RT','original_joint':'原目录联合','localized_fixed_joint':'永久局域固定能量','binding_screen_joint':'原共享E(C)',
 'fraction_constant_RT':'常数比例RT','fraction_dose_RT':'共享比例RT','fraction_constant_joint':'常数比例联合','fraction_dose_joint':'共享比例联合',
 'fraction_binding_joint':'比例＋常数绑定能','output_linear_RT':'同容量输出RT','output_linear_joint':'同容量输出联合',
 'direct_ridge_RT':'直接Ridge RT','direct_ridge_joint':'直接Ridge联合','dose_line_RT':'剂量直线','activation_prior_RT':'中位激活比',
 'identity_unbounded':'n=C无收缩','identity_H_unbounded':'n=C+H无收缩','fraction_screen_constant_joint':'E(C)＋常数比例',
 'fraction_screen_joint':'E(C)＋共享比例','output_screen_joint':'E(C)＋同容量输出',
 'FD_original_RT':'新FD原结构，rH1','FD_binding_joint':'新FD E(C)，rH1',
 'HallII_original_RT':'纯II Hall原结构','HallII_binding_joint':'纯II Hall E(C)',
 'HallExtra_original_RT':'II＋常数tau原结构','HallExtra_binding_joint':'II＋常数tau E(C)',
}
SHORT={'FD_original_RT':'新FD原','FD_binding_joint':'新FD E(C)','HallII_original_RT':'II原','HallII_binding_joint':'II E(C)',
 'HallExtra_original_RT':'II＋额外率原','HallExtra_binding_joint':'II＋额外率E(C)'}
SOURCELABEL={method+'_'+suffix:label+'：'+kind for method,label in SHORT.items() for suffix,kind in (('output','输出'),('stock','库存'),('Tf','Tf'))}


def table(frame,labels,scope='all_query',metric='MAE',column='method'):
    d=frame if scope is None else frame[frame.scope.eq(scope)]
    p=d.pivot(index=column,columns='budget',values=metric)
    lines=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for method,label in labels.items():lines.append('| '+label+' | '+' | '.join(f'{p.loc[method,b]:.6f}' for b in (0,1,2))+' |')
    return '\n'.join(lines)+'\n'


def figures(primary,temperature,source,chain):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    chosen={'original_RT':'Old original','binding_screen_joint':'Old E(C)','FD_original_RT':'New FD, rH1',
        'HallII_original_RT':'II Hall, original','HallExtra_original_RT':'II + extra, original','HallII_binding_joint':'II Hall, E(C)'}
    fig,axs=plt.subplots(1,3,figsize=(12.9,3.6))
    for ax,d,scope,title in zip(axs,(primary,primary,temperature),('all_query','common18','common40_125'),
        ('RT all queries','RT fixed common18','Same 7 Si specimens: temperature')):
        for m,label in chosen.items():
            h=d[d.method.eq(m)&d.scope.eq(scope)].sort_values('budget');ax.plot(h.budget,h.MAE,'o-',ms=3,lw=1,label=label)
        ax.set(xlabel='Allowed specimen anchors',ylabel='Source-equal MAE (dex)',xticks=[0,1,2],title=title);ax.grid(alpha=.2)
    axs[0].legend(fontsize=6.5);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'coherent_prediction.{ext}',dpi=190)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11.8,3.5))
    for ax,scope in zip(axs,('all_query','common18')):
        for m in ('FD_original_RT','HallII_original_RT'):
            for kind,style in (('output','-'),('stock','--'),('Tf',':')):
                d=source[source.variant.eq(m+'_'+kind)&source.scope.eq(scope)].sort_values('budget')
                ax.plot(d.budget,d.MAE,'o'+style,lw=1,ms=3,label=m.replace('_original_RT','')+': '+kind)
        ax.set(xlabel='Allowed specimen anchors',ylabel='Source-equal MAE (dex)',title=scope,xticks=[0,1,2]);ax.grid(alpha=.2)
    axs[0].legend(fontsize=6.5);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'coherent_source.{ext}',dpi=190)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.9,3.8))
    for m,label in {'original_RT':'Old raw CT','FD_original_RT':'New FD, forced rH1','HallII_original_RT':'II Hall','HallExtra_original_RT':'II + extra Hall'}.items():
        d=chain[chain.method.eq(m)&chain.budget.eq(0)]
        ax.scatter(d.observed_log_mu,d.predicted_log_mu,s=32,label=label)
    ax.plot([-1,4],[-1,4],color='black',lw=.8)
    ax.set(xlabel='Measured log10 Hall mobility (cm2/Vs)',ylabel='Predicted log10 Hall mobility (cm2/Vs)',
        title='5 chemical/Hall/mobility pairs, 2 DOI; no new mu targets; fixed priors disclosed')
    ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'coherent_mobility.{ext}',dpi=190)
    plt.close(fig)


def main():
    a=json.loads((FINAL/'contract_audit.json').read_text());s=json.loads((SOURCE/'contract_audit.json').read_text())
    g=json.loads((OUT/'numerical_gates.json').read_text());r=json.loads((OUT/'research_runtime/verification.json').read_text())
    state_scope=json.loads((OUT/'state_scope/contract_audit.json').read_text())
    assert all(v['passed'] for v in (a,s,g,r,state_scope))
    primary=read(FINAL/'primary_summary.csv');temperature=read(FINAL/'temperature_summary.csv');source=read(SOURCE/'summary.csv')
    chain=read(OUT/'research_runtime/full_chain_predictions.csv');chain_summary=read(OUT/'research_runtime/full_chain_summary.csv')
    tm=read(OUT/'temperature_transport/summary.csv');trows=read(OUT/'temperature_transport/predictions.csv')
    ablation=read(OUT/'scattering_ablation/summary.csv')
    figures(primary,temperature,source,chain)
    fig,axs=plt.subplots(1,3,figsize=(12.8,3.6))
    temp_methods={'FD_binding_joint_output':'State E(C), rH1','HallExtra_binding_joint_output':'State E(C), extra Hall',
        'HallII_original_RT_output':'State original, II','unit_dose_ionized_extra':'Unit dose + extra','constant205':'Constant205'}
    for ax,sample in zip(axs,sorted(trows.sample_id.unique())):
        q=trows[trows.sample_id.eq(sample)&trows.budget.eq(0)]
        y=q[q.method.eq('constant205')].sort_values('T_K');ax.plot(y.T_K,10**y.observed_log_mu,'k.',label='Measured')
        for method,label in temp_methods.items():
            h=q[q.method.eq(method)].sort_values('T_K');ax.plot(h.T_K,10**h.predicted_log_mu,lw=1,label=label)
        ax.set(xlabel='Temperature (K)',ylabel='Hall mobility (cm2/Vs)',yscale='log',title=sample.rsplit('::',1)[-1]);ax.grid(alpha=.2)
    axs[0].legend(fontsize=6);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'temperature_mobility.{ext}',dpi=190)
    plt.close(fig)
    get=lambda m:[float(primary[primary.method.eq(m)&primary.budget.eq(b)&primary.scope.eq('all_query')].MAE.iloc[0]) for b in (0,1,2)]
    source_get=lambda m:[float(source[source.variant.eq(m)&source.budget.eq(b)&source.scope.eq('all_query')].MAE.iloc[0]) for b in (0,1,2)]
    claims=dict(original=get('original_RT'),newFD=get('FD_original_RT'),HallII=get('HallII_original_RT'),HallExtra=get('HallExtra_original_RT'),
        source={m:source_get(m) for m in SOURCELABEL})
    claims['RT_mu']={m:[float(chain_summary[chain_summary.method.eq(m)&chain_summary.budget.eq(b)].mu_MAE.iloc[0]) for b in (0,1,2)]
        for m in ('original_RT',*study.NEW_METHODS)}
    claims['temperature_mu']={m:[float(tm[tm.method.eq(m)&tm.budget.eq(b)&tm.scope.eq('all_query')].MAE.iloc[0]) for b in (0,1,2)]
        for m in ('FD_binding_joint_output','HallExtra_binding_joint_output','constant205','unit_dose_rH1','unit_dose_ionized_extra')}
    claims['temperature_query_reads']=[int(tm[tm.method.eq('constant205')&tm.budget.eq(b)&tm.scope.eq('all_query')].queries.iloc[0]) for b in (0,1,2)]
    claims['state_scope']=state_scope
    mu_tablelabels={**LABELS,**{m:l for m,l in SOURCELABEL.items() if not m.endswith('_output')}}
    theory=(ROOT/'docs/ML_SANDBOX_COHERENT_HALL_THEORY_20261018.md').read_text()
    theory=theory[theory.index('## 1.'):].split('## 5.',1)[0]
    theory=re.sub(r'(?m)^## ([1-5])\.',r'### 10.\1',theory)
    texts=[
      '# ML＋Sandbox：同态Hall输运与掺杂性质预测\n',
      '本阶段实际打通“化学掺入量→共享内部学习→冷热守恒状态→Hall载流子/迁移率/电导率”。系统goal保持active，用户要求的掺杂性质预测方向不变。原29Si/Sn/5DOI与0/1/2锚点为主，7旧Si变温曲线只作补充。GPU1授权有效，本轮CPU，无新DFT、MD、材料样品或仪器实验。\n',
      '## 1. 本轮增加的真实能力\n',
      '此前完整mu使用经验CT头，新SI散射只独立通过单位检查。本轮用新的不可变profile统一自由电子Fermi统计、DOS/速度/散射质量；从同一求解状态的自由电子压缩率和Nq平方预算算正散射率与Hall矩。计算的rH进入nH训练，muH/sigma随同一状态输出，未保留旧nH分数只更换mu公式。\n',
      'ML仍学习跨条件复用的共享E(C)及有效全局参数；关系进入冷热自由能与隐式梯度。所有新拟合都在合法训练来源上重新优化。Hall一致拟合本身有文献先例，不是本文新算法；本阶段的材料增量必须看后面的源外结果。\n',
      '模型有三种明确读出：新FD＋rH1对照、纯离化杂质II Hall、II加固定有效常数tau Hall；每种各配原结构RT和共享E(C)联合训练，共6新模型。额外通道预设mu0=205cm²/Vs、独立于T，不是已求解的声子、晶界或新测微观速率。\n',
      '## 2. 理论、数值修正及失败记录\n',
      '新的Fermi数学表12801节点只来自积分，没有性质标签；区间内C2 logF样条、低端级数、高端Sommerfeld及C2过渡。独立45位polylog核对函数与一/二阶导数，最大归一化误差6.55e-8；测试范围的logF斜率保持正，运行时也检查该斜率，极低人口允许数值下溢。Nc=3.33e18对应单标量质量.26016457m0。原STH两人口公式明确保持旧有效近似，没有把它们称为精确微观统计。\n',
      '首轮数值检查passed=false：实际接口误把读出logNq²导数接到logT列；3201节点的二阶导数精度也不足。保留初版库、脚本、表和完整失败日志；修正连接并加密数学表后重跑。源零更新相对不同求根初值的差异约1e-14，现分别检查同缓存严格恒等和跨初值浮点误差，未放宽物理梯度门槛。错误修正发生在训练前。\n',
      f'最终{len(g["checks"])}项门槛全部通过。完整原状态IFT最大2.89e-8，实际载流子/性质链参数梯度最大2.08e-8/1.95e-8；来源梯度最大4.56e-9，读出三个输入链最大1.17e-10。包括独立自适应积分、正Hall矩、能量网格/尾收敛、冷热同温恒等、profile序列化和固定来源零更新。它们是声明模型的数学/实现核验，不是材料准确率。\n',
      '## 3. 数据、训练和验证契约\n',
      '原29片Si/Sn、5DOI；0/1/2每来源样品锚点；29/23/18查询和固定共同18保持。MAE来源内平均后等权。外层整DOI排除，内层选择7点lambda网格；历史队列已暴露，是回顾性研发，不称盲测/预注册。5DOI不是5独立实验室。29条均无实测pO2，1e-5atm仍为情景，H与C目录缺失等证据边界不变。\n',
      '6新模型×15个3/4来源上下文=90拟合，180声明起点，120评估上限与1e-7停止阈值，全部停止成功，中间状态失败0，因此没有触发统一360复核。继承450记录后为540记录、300物理拟合；复合602起点，实际历史累计606。停止成功不等于全局最优。数值、输入、代码及数学表哈希在训练前固定。\n',
      '原结构仅RT训练；E(C)模型为来源等权RT平方损失加.1倍温度补充损失。温度仍旧7Si/2DOI、142n读数，共同40—125K为80查询，RT选出的lambda迁移，不是独立温度选参。90新拟合没有用mu标签；同一nH监督产生的物理mu属于待检验输出。growth_T_C保留为追溯输入，但当前共享Tf不是其已验证的动力学函数。\n',
      f'独立审计重建{a["nested_candidate_inner_evaluations"]}项内层评分、{a["outer_saved_rows"]}主输出、{a["temperature_saved_rows"]}温度输出，来源、锚点、权重、目标和读出一致性通过，主查询无缺失。最大训练目标差{a["maximum_objective_reconstruction_error"]:.3g}。相同候选/评估上限不等于相同运行时间，完整成本表保存。\n',
      '## 4. 原主任务：全部26种基础方法\n',
      '下表是全部查询的来源等权MAE，单位dex。旧20种方法用冻结拟合及确切哈希复用；强简单与直接ML对照全部保留。\n',table(primary,LABELS),
      '新FD+rH1几乎复现旧原结构。纯II原结构零/一锚点.262469/.244683优于旧.266289/.251591，两锚点.261397却劣于旧.257810；也未超过此前一锚点常数库存.229552、两锚点常数库存输出.250297。纯II共享E(C)室温变差，额外常数tau也没有稳定增益。不能把修复理论约定说成全面提高材料精度，不能唯一归因改善来自真实微观散射。\n',
      '共同18保持同一批查询，完整结果如下。\n',table(primary,LABELS,'common18'),
      '![新自由统计及Hall反馈与原基线：不同预算、固定查询及补充温度任务](figures/coherent_prediction.png)\n',
      '温度补充的几个关键结构列在下表，全部26种在temperature_summary.csv。纯II E(C)共同80零锚点.304282略低于旧.308611，但两锚点没有保持；额外率E(C).323503更差。不同温区/任务的局部信号不能替代原室温主终点。\n',table(temperature,{m:LABELS[m] for m in ('original_RT','binding_screen_joint',*study.NEW_METHODS)},'common40_125'),
      '## 5. 同样合法锚点下的内部来源适应\n',
      '在首批结果暴露后，为全部6新结构固定第二个来源协议，各比较output/stock/Tf，共18变体，未按外层最好单格筛掉差结构。全局共享权重冻结，内部z[-6,6]按相同25节点/局部精化拟合载流子锚点，再在训练来源内选择lambda。mu/sigma不参与选参。局部I/(I+lambda)不是后验，z也不是真实过程标签。\n',
      f'450来源情境，{s["profile_candidate_objectives"]}个内部候选目标、{s["closed_checks"]}项闭式核对、{s["inner_candidate_evaluations"]}项内层评分、{s["outer_saved_rows"]}行输出独立审计。查询缺失{s["missing_query_rows"]}。输出6基线与首批选择/预测复现最大{s["maximum_output_replay_error"]:.3g}dex。全部查询与共同18分别如下。\n',
      table(source,SOURCELABEL,column='variant'),table(source,SOURCELABEL,'common18',column='variant'),
      '![同一原结构的新FD与纯II读出在来源输出、库存、Tf适应下的完整预算变化](figures/coherent_source.png)\n',
      '来源适应是否改善nH必须和同一结构输出适应、历史强对照同时看；边界及不更新均保存。内部适应重求库存/电荷/散射，输出偏移不重求状态。即使nH改善，也不自动保证mu改善，不能把从nH锚点拟合的有效库存命名为独立识别的真实散射中心浓度。\n',
      '## 6. 完整性质链：同态输出与实际精度分开\n',
      '新模型与内部来源变体的mu来自同一状态，本轮没有用mu标签重训；旧20方法在本表用固定原始CT头，分开52条件mu标签带来的额外学习信息。固定205/CT数值本身仍含历史经验迁移率知识；历史锚定点与当前队列的逐片独立性未重新核实，不能把不读取mu表说成完全没有mu信息。纯II分支不含205通道。此前源外校准CT结果在阶段14，用了52条件对/11引用组，信息预算又不同。详见COHERENT_PRIOR_SCOPE说明。\n',
      '室温完整链依旧仅5条明确化学/Hall/mu配对、2DOI，0/1/2锚点对应5/4/2查询。下表为mu来源等权MAE dex，完整nH/mu/sigma逐样本和逐来源表均保存。\n',table(chain_summary,mu_tablelabels,scope=None,metric='mu_MAE'),
      '![保留全部5条配对，包括低mu薄膜；均匀单带近似是否适用需由数据判断](figures/coherent_mobility.png)\n',
      'sigma=e nH muH由相同Hall量派生，没有独立sigma标签；分量误差可能抵消。输出偏移后sigma随经验nH派生且标明不对应新中性状态；内部stock/Tf则保持同一状态。新物理读出可越过CT原固定输出下限，但这不是准确性保证，缺失形貌、边界或接触输入不能由数值正确补齐。\n',
      '### 6.1 迁移率改善究竟来自什么\n',
      '新FD原结构rH1的室温mu MAE为.883023，旧原始CT为1.501275；真正使用Hall矩的额外率原结构为.968433，纯II则为1.570473。前者是单位Hall近似对照，不能把它等同完整Hall矩的优势；与旧源外校准CT1.332676的对比还涉及不同mu信息预算。五配对/两DOI仍不足以证明一般实用精度。\n',
      '为区分散射函数变更与库存信息，固定同一个新FD/rH1状态、屏蔽、质量和205额外率，仅换Nq²/Nabs/自由n/C+H+背景代理。只在rH1对照做此消融，因其nH算子不依赖散射；没有偷换其他Hall模型的nH成绩。室温原结构结果如下。\n',
      table(ablation[ablation.endpoint.eq('RT')&ablation.model.eq('FD_original_RT')],
        {'Nq2':'同一模型Nq²预算','Nabs':'Nabs代理','free_n':'自由n代理','chemical_plus_H_plus_background':'C+H+背景代理','constant205':'固定mu205'},scope=None,column='proxy'),
      '同一状态的Nq²相对化学代理0锚点1.051611改善到.883023，说明这5配对里不只是替换函数的效果；但不能据此唯一识别真实补偿中心，未知结构/接触和未校准库存仍可能被有效状态吸收。消融1270行全部保存，原读出精确复现；不是重新训练的完整Hall消融或新独立来源。\n',
      '### 6.2 54条既有变温mu的同温预测\n',
      '新增一项冻结验证使用阶段8已经提取的54个mu读数，实际只有3个旧Si样品、1个McCandless DOI。每条在其自己的T_K预测，不拿近温nH当作同温真值，不派生新增sigma标签。共享模型完全排除该DOI；mu不进入任何训练或选参。2916个独立模型输出均成功；含4简单对照的完整表3564行。\n',
      '原RT样品角色决定mu查询：0/1/2预算分别3/2/1曲线、54/38/24读数。下表先曲线等权；预算间查询样品改变，不能将误差降低自动归于适应。该DOI内部来源坐标都未更新，输出偏移不改mu；固定共同24读数时各预算的mu误差相同。\n',
      table(tm,{**SOURCELABEL,'constant205':'固定mu205','unit_dose_rH1':'单位剂量rH1','unit_dose_ionized':'单位剂量纯II','unit_dose_ionized_extra':'单位剂量II＋额外率'},column='method'),
      '![三条旧样品曲线的零锚点预测；低温/不利读数全保留，非新增独立样品](figures/temperature_mobility.png)\n',
      '共享E(C)/rH1零锚点.167879、额外率Hall .168960，比常数205的.188158略好，但单位剂量rH1已为.169903、单位剂量额外率.169983，不能将大部分曲线信号归于复杂Sandbox。纯II .842601—1.313354明显过高。变温消融中原结构Nq².180438甚至不及自由n代理.170959；共享E(C)的Nq².167879仅略优于n代理.169048。库存收益随任务变化，不能宣称统一优势。\n',
      '## 7. 独立预测复用\n',
      f'保存270包：6新基础＋12额外内部来源变体，各5排除来源×3预算。独立子进程只读物理输入、合法锚点、数学/能量参考及保存关系，禁止性质表/拟合/选择/验证表。重新求状态与来源坐标，{r["carrier_rows"]}载流子复现最大{r["maximum_carrier_error"]:.3g}dex、电荷预算{r["maximum_charge_error"]:.3g}；同态mu/sigma最大{r["maximum_transport_error"]:.3g}。\n',
      f'{r["chemical_paired_transport_replays"]}条化学配对查询同样独立回放，最大{r["maximum_paired_transport_error"]:.3g}dex。没有查询标签或mu标签进入任务。完整对照表{r["full_chain_rows"]}行不代表新增独立样品。研究包未全五来源重训，不认证新稀土或器件性能。\n',
      '## 8. 对目标的实质推进与剩余问题\n',
      '本轮已经把原先分开的缺陷状态与输运参考连接成可训练、可独立调用的完整性质链，完成原主任务和固定锚点对照。理论严谨性服务于可运行预测，未将工作退回纯机制解释；完整mu、真实跨研究迁移及新元素适用范围仍需分别验证。\n',
      '状态适用域审计重求1566状态且预测不变：原生母类最大占位3.71e-4、原生事件总数/化学式单元最大2.07e-4，没有用浓度饱和获得收益。然而1566状态的有效Tf均高于表中生长温度，最大差1005.7K；生长记录未必是完整峰值热历史，因此目前Tf只能作为有效校准参数，不能宣传已识别真实冻结温度或由真实热历史预测了库存。平衡束缚响应/自由屏蔽最大42.95，是否需并入束缚屏蔽取决于未知响应时标；free-only是显式近似。\n',
      '目前没有跨预算稳定优于全部强基线。下一项先审计热历史/代理输入，在有效Tf与记录热上界约束情景下检验nH/mu联合监督；之后再固定共享正额外率与同容量mu输出对照，避免额外自由度先吸收温度解释问题。保持主nH、相同DOI/锚点和简单/直接ML对照。联合损失是显式预测目标，未知协方差下不称独立Gaussian似然或唯一微观识别；sigma不作第三标签。详细未执行方案见JOINT_N_MU_NEXT。\n',
      '## 9. 文献和复现入口\n',
      '[Ma2016作者机构原文](https://djena.engineering.cornell.edu/papers/2016/apl16_nan_GaOx_transport.pdf)用于式2/Hall定义及II率/系数核对；其模型含不同质量、束缚态屏蔽与多通道，这里没有完整复现。[Neal2018作者原文](https://arxiv.org/abs/1809.01230)已有自洽nH/muH拟合先例，不能把这一原则本身称为新发明。literature/reading_record.json区分实际阅读深度和用途，没有新增数字化训练标签。\n',
      '源代码coherent_fermi.py、coherent_transport.py、coherent_hall_model.py；脚本phase15_gates/prediction/audit/source_adaptation/source_audit/runtime；输入/代码/数学表哈希及全部失败版本保存。CSV用csv_io_v2读取，防数字ID/哈希转换。使用ga2o3环境、PYTHONPATH=src及CPU线程1；冻结后回放写新目录，见research_runtime/README.md。\n',
      '## 10. 训练前写定的物理与导数论证\n',
      '以下保留原协议的数学部分；训练门槛与预测对照已在第2—6节给出，完整原文件与哈希另行存档。R0式中的mu0须用SI值205×1e-4m²/Vs，代码已显式换算。实现的一处high-temperature注释应读高eta简并Sommerfeld展开，公式与数值无误；详见implementation_errata.json。\n',theory,
    ]
    target=OUT/'ML与Sandbox_同态Hall输运与掺杂性质预测.md';target.write_text('\n'.join(texts))
    energy.write(OUT/'report_claims.json',dict(created_utc=energy.now(),claims=claims,
        evidence_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (FINAL/'primary_summary.csv',FINAL/'temperature_summary.csv',
            SOURCE/'summary.csv',OUT/'research_runtime/full_chain_summary.csv',OUT/'research_runtime/verification.json',OUT/'numerical_gates.json',
            OUT/'temperature_transport/contract_audit.json',OUT/'temperature_transport/summary.csv',OUT/'scattering_ablation/summary.csv',OUT/'state_scope/contract_audit.json')},
        numerical_checks=864,new_fits=90,new_starts=180,base_models=26,source_variants=18,independent_bundles=270,
        new_mu_supervision=False,independent_sigma_labels=0,stable_primary_increment_established=False,goal_complete=False))
    print(target)


if __name__=='__main__':main()
