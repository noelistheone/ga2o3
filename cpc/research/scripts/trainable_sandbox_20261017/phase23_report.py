"""Traceable frozen-source property forecasts and process-input theory report."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv as read
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase23_process_inputs';PRED=OUT/'source_prediction';CLAIMS=[];CACHE={}


def value(path,selector,column,fmt='.6f'):
    if path not in CACHE:CACHE[path]=read(path)
    d=CACHE[path]
    for key,v in selector.items():d=d[d[key].eq(v)]
    assert len(d)==1,(path,selector,len(d));x=float(d[column].iloc[0])
    CLAIMS.append(dict(path=str(path.relative_to(ROOT)),sha256=e.sha(path),selector=selector,column=column,value=x))
    return format(x,fmt)


def table(context,tier,scenario,methods,prop='nH',scope='common_query'):
    lines=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |'];maxb=1 if tier=='same_specimen_prose_2' else 2
    for name in methods:
        cols=[value(PRED/'summary.csv',dict(context=context,tier=tier,scenario=scenario,method=name,budget=b,scope=scope),prop+'_MAE') for b in range(maxb+1)]
        if maxb==1:cols.append('无独立查询')
        lines.append('| '+name+' | '+' | '.join(cols)+' |')
    return '\n'.join(lines)


def figures():
    d=read(PRED/'predictions.csv');q=d[d.context.eq('full5')&d.tier.eq('process_calibrated_12')&d.scenario.eq('regime_rates')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.7));styles=[('OriginalHall',0,'#2864a0'),('InventoryEnsemble',2,'#b97400'),('UnitDose',0,'#777777'),('DirectRidge',2,'#7a4696')]
    observed=q[q.method.eq('OriginalHall')&q.budget.eq(0)].sort_values('incorporated_cm3')
    for ax,prop,title in zip(axes,['nH','mu'],['Hall carrier density','Hall mobility']):
        ax.scatter(np.log10(observed.incorporated_cm3),observed['observed_log_'+prop],color='black',s=26,label='Published observations',zorder=5)
        for r in observed.itertuples():ax.annotate(r.sample_label,(np.log10(r.incorporated_cm3),getattr(r,'observed_log_'+prop)),xytext=(3,4),textcoords='offset points',fontsize=8)
        for method,budget,color in styles:
            g=q[q.method.eq(method)&q.budget.eq(budget)].sort_values('incorporated_cm3')
            ax.plot(np.log10(g.incorporated_cm3),g['predicted_log_'+prop],color=color,lw=1.4,marker='.',ms=4,label=method+f' ({budget} anchors)')
        ax.set_xlabel(r'$\log_{10} C_{\rm proxy}\ [{\rm cm}^{-3}]$');ax.set_title(title);ax.grid(alpha=.2)
        ax.set_ylabel(r'$\log_{10} n_H\ [{\rm cm}^{-3}]$' if prop=='nH' else r'$\log_{10}\mu_H\ [{\rm cm}^{2}/{\rm Vs}]$')
    axes[0].legend(fontsize=7,loc='upper left');fig.suptitle('Frozen forecasts, primary process-input scenario (not actual-dose truth)',fontsize=10)
    fig.tight_layout();folder=OUT/'figures';folder.mkdir(exist_ok=True,parents=True)
    for ext in ('png','pdf'):fig.savefig(folder/f'new_source_forecasts.{ext}',dpi=180)
    plt.close(fig)
    e.write(folder/'provenance.json',dict(prediction_path=str((PRED/'predictions.csv').relative_to(ROOT)),prediction_sha256=e.sha(PRED/'predictions.csv'),
        selection=dict(context='full5',tier='process_calibrated_12',scenario='regime_rates'),shown_methods=styles,
        lines='connecting evaluated discrete inputs only;not interpolation uncertainty',illustrative_selection_not_hyperparameter_choice=True))


def main():
    audit=json.loads((PRED/'audit/contract_audit.json').read_text());raw=json.loads((OUT/'process_audit/audit.json').read_text());gates=json.loads((OUT/'process_checks/gates.json').read_text())
    assert audit['passed'] and raw['passed'] and gates['passed'];protocol=json.loads((PRED/'protocol.json').read_text())
    full=sorted(c['method'] for c in protocol['configs'] if c['context']=='full5');excluded=sorted(c['method'] for c in protocol['configs'] if c['context']=='exclude_related_McCandless')
    figures();blocks=['# ML与Sandbox：过程输入审计与新来源掺杂性质预测',r'''## 1. 目标、实际工作与结论

主方向始终是可训练 ML＋Sandbox 预测掺杂性质：实际掺入量及可追溯条件 → 载流子，再检验相容迁移率、电导率和有证据支持的候选排序。原系统 goal 实际核对为 active；理论、可识别性和数据审计服务于预测改进，不单独完成目标。

本轮解析新来源原始日志并实际执行 7320 次冻结性质预测，全部有数值；没有新共享参数训练或新来源超参数选择。保留完整旧5来源上下文的34方法与排除相关 McCandless 来源上下文的27方法，后者没有伪造旧7能量模型。原29样品/5DOI、0/1/2锚点、common18及全部旧结果不变；新来源未并入训练。

结果没有建立稳定新增内部学习优势。12样品过程代理层上，部分两锚点物理来源适应可改善，但这种表现不跨上下文成立；两个粗略同样品化学条件仍暴露高掺杂低估。无法将所有失败都归给过程浓度代理，也不能凭输入不完整就断言 ML＋Sandbox 无法改进。

本阶段使用 CPU，GPU1 授权保留；无新 DFT、MD 或实体实验。下一步先审计独立化学堆栈能否支持一个真实过程输入学习层，再连接并检验完整性质预测，不继续只增加来源先验复杂度。''',
r'''## 2. 读了哪些原始证据，新增数据是什么

[Azizie等2023作者全文](https://djena.engineering.cornell.edu/papers-new/2023/azizie2023silicon.pdf)的制备方法、TableII、Fig2标定及SIMS段落已核对；[原数据仓库](https://data.paradim.org/doi/zsda-pa72/)的生长日志和XRR校准原始归档已取得。文献DOI10.1063/5.0139622。

重要修正：TableII有12片Hall样品a–l，非a–k；m/n是两片SIMS堆栈。e/k被正文明确指定为同样品SIMS与Hall，但正文只有约2e18/3e19cm⁻³的化学尺度，精确SIMS值未重建。全部12片有室温nH和muH；只有两片具有本轮可读的粗略独立化学输入，不能把12片都并入严格实际掺入量验证队列。

该论文与旧McCandless来源共享团队。全旧5来源与排除相关McCandless的两个冻结上下文分别报告，均不称独立实验室盲测。当前DOI不在共享训练清单中；这不单独证明物理样品从未跨文献复用。来源已被人工阅读，属于回顾性开发验证。

Ge2021 DOI10.1063/5.0059657仍未取得可核验全文。正式元数据和作者发表目录存在，不等于已建立Ge化学/Hall配对、目录覆盖或预测域。未将其计为新元素验证。补充材料TableS1的完整公开版本尚未核实；XRR扫描不等于已校准的唯一厚度/密度解。''',
r'''## 3. 生长日志审计：哪些信息已读出，哪些仍不能配对

23份嵌套日志、13584个目录项完成清点；直接挡板状态优先于未经验证的位掩码说明。elapsed time明确为秒，按事件保持区间而非采样条数汇总；负哨兵值、量程外值和缺失通道均保留。1123缺少SiO2及其挡板通道，没有用另一通道偷偷替代。

源温相近只产生13个候选对应，已核实日志ID到论文样品字母的映射仍为0。源炉温度不是基底温度，光学高温计数值也不能自动覆盖论文所报525°C。接触区后续再生长的630°C不能当作主层生长温度或完整淬火历史。实际臭氧环境不能直接换成平衡O2化学势。

独立事件与时间积分核验160项通过，生成51行条件入射剂量记录（包括缺失记录）；14份具有Ga/Si/主挡板共同开放时段的日志中，源温很稳定，逐时指数通量积分与平均源温代入的最大相对差仅约1.24e-6。故本数据中源炉短时波动本身没有显示可解释大幅Hall差异的信息。该结果仍以原日志源温数值按°C解释、印刷通量律及事件保持为条件，不能提升为实际剂量测量。长时间预热区的较大非线性平均差不用于主层性质预测。

17份XRR扫描和多个保存模型均保留。扫描角度2Theta-Omega中的2Theta不能直接当Omega；CuKα1波长为1.540598Å。保存模型厚度/密度存在不同拟合，未按Hall好坏挑解，也未由此生成新化学真值。''',
r'''## 4. 输入层、方程与固定锚点

A层：same_specimen_prose_2。e/k化学输入固定三情景：(2e18,3e19)、(1.5e18,2.5e19)、(2.5e18,3.5e19)cm⁻³。它们是正文精度/解释敏感性，不是置信区间。仅0/1锚点，低剂量e为锚，固定共同查询k；两样品不能制造2锚点后的独立查询。

B层：process_calibrated_12。Fig2明确以XRR/SIMS拟合的印刷通量为

\[
J_s(T_s)=4.6\times10^{28}\exp[-5.3\ {\rm eV}/(k_BT_s)],\qquad C_{\rm proxy}=J_s/v.
\]

Ts为SiO2源炉的K温度，摄氏值加273.15；kB=8.617333262145e-5eV/K。J的约定单位为atoms cm⁻²s⁻¹，v以cm/s计，1μm/h=1e-4/3600cm/s。5.3eV是源标定斜率，不是掺杂电离能。正文低源温Hall反推的星形通量未用于重标定。

三个速度情景预先固定：SIMS堆栈所报0.86/1.3μm/h、统一1μm/h、前者乘1.4。堆栈速度不等于每片Hall膜实测速率；某片STEM厚度高约40%不证明全部膜误差都是40%。Cproxy属于独立校准支持的过程代理，包含掺入和生长约定，不能当12片实际SIMS浓度。

锚点身份仅由主情景输入排序确定：d和随后相隔至少0.10dex的c；两者目标Hall相同也照样保留。所有情景复用这两片，0/1/2锚点的全部查询数12/11/10；共同查询始终排除c/d，固定10片。模型接收当前已知输入集合，但不接收查询标签。

全部推断用300K比较约定、文献525°C生长输入和旧1e-5atm有效O2情景。后者未由实际O3标定。冻结模型内的有效冻结温度也不是本轮测得的热历史。过程输入和这些物理假设的限制必须与预测结果同时保留。''',
r'''## 5. 相同输入a/j带来的严格误差下界

a与j在TableII的已报告输入相同，Hall浓度相差40.7407倍，即1.610029dex。任意只接收这些输入并进行同一来源条件化的确定性预测f，必须满足

\[
|f(x)-y_a|+|f(x)-y_j|\ge|y_a-y_j|.
\]

y为log10 Hall浓度；右式来自三角不等式。因此这对样品平均MAE至少0.805014dex；完整12片均值至少0.134169dex，固定10片共同查询至少0.161003dex。这里限定为已报告输入集，不能据此证明原表错误、真实样品状态相同或更完整工艺输入无效。

实际逐方法/情景/预算核验a/j的相同输入预测，最大差为0；nH、mu和派生sigma均满足各自三角下界。两片均保留，没有调换源温、删点或按目标反配日志。''',
'## 6. 全部载流子预测：过程代理层\n\n单位dex，固定共同10片。full5为旧5来源冻结模型，排除相关来源使用旧4来源训练和先验。并列上下文是相关来源敏感性检验，不能当独立重复制备。\n\n### 完整旧5来源：34方法\n\n'+table('full5','process_calibrated_12','regime_rates',full),
'### 排除相关McCandless来源：27方法\n\n'+table('exclude_related_McCandless','process_calibrated_12','regime_rates',excluded),
r'''必须保留的有利与不利事实：full5的FieldAnchorSafe在2锚点达0.361363，优于同预算UnitDose0.476460、MedianActivation0.435070及直接Ridge0.483574；但排除相关来源时它为0.609933，且0/1锚点没有相同收益。不能抹去这个条件改进，也不能提升为跨来源共享学习增量。当前没有新共享训练。

InventoryEnsemble full5在2锚点0.545530，较其零更新0.617816改善，但仍不如简单对照；其排除相关来源结果为0.602050。UnitDose零锚点0.321556较好，但输入本身是过程代理且同输入样品有下界，不能解释为普遍100%掺杂激活的证明。

all_query和固定common指标都已保存。不同预算的all_query样本组成变化，不可直接称为锚点帮助或损害；上表使用固定共同集合。只有一个新来源，不能计算有说服力的跨来源显著性。''',
'## 7. 两个粗略同样品化学条件\n\n主输入为正文粗略尺度。共同查询仅k，一点误差不代表稳定泛化。完整结果同时保留三个化学输入情景。\n\n### 完整旧5来源\n\n'+table('full5','same_specimen_prose_2','published_rounded',full),
'### 排除相关来源\n\n'+table('exclude_related_McCandless','same_specimen_prose_2','published_rounded',excluded),
r'''UnitDose零锚点在主情景的k恰为0误差，是粗略化学值3e19与目标Hall数值3e19恰好相等；不能当作高精度物理验证。另两个输入情景为0.079181/0.066947dex。OriginalHall三个情景的k误差均约1.16–1.17dex，即使承认输入粗略，这也暴露当前高掺杂读出的实际预测缺口，不能全部用过程代理解释掉。''',
'## 8. 同次迁移率与派生电导\n\n以下是相同预测状态/经验头输出的muH误差，固定共同查询，单位dex。经验方法使用旧CT头；物理模型使用既有相干FD/Born-RTA及固定经验额外散射，并未在本轮学习新输运。',
    ]
    selected=['OriginalHall','FieldAnchorSafe','InventoryEnsemble','SharedReadout','UnitDose','MedianActivation','DirectRidge']
    for context in ('full5','exclude_related_McCandless'):
        blocks.append('### 过程代理层：'+context+'\n\n'+table(context,'process_calibrated_12','regime_rates',selected,prop='mu'))
    blocks.append(r'''sigma由e*nH*muH派生，独立sigma标签为0。所有数值请求逐项满足该恒等式。nH与mu误差可能相加或相消：A层k的OriginalHall中，载流子与迁移率同向低估，sigma误差达1.457526dex；直接Ridge两项误差方向相反，sigma误差更小不等于新独立验证成功。全部7320条nH/mu/sigma向量及1830行完整分层汇总保存，未筛去较差读出。

新来源12对Hall n/mu不等于12个严格化学输入的完整性质配对。旧严格化学/输运队列、两个本轮粗略化学条件和12个过程条件层分别保留，没有把它们合并扩大验证计数。

![冻结新来源性质预测；连线仅连接已评估输入，黑字为样品字母。a/j横坐标相同，观测相差明显。](figures/new_source_forecasts.png)''')
    sensitivity=['| 方法及输入层 | 0锚点三情景范围 | 1锚点三情景范围 | 2锚点三情景范围 |','| --- | --- | --- | --- |']
    for tier in ('process_calibrated_12','same_specimen_prose_2'):
        for m in ('OriginalHall','FieldAnchorSafe','InventoryEnsemble','UnitDose','MedianActivation','DirectRidge'):
            cols=[]
            for b in range(3):
                if b==2 and tier=='same_specimen_prose_2':cols.append('无独立查询');continue
                sel=dict(context='full5',method=m,tier=tier,budget=b,scope='common_query',property='nH')
                cols.append(value(PRED/'audit/scenario_sensitivity.csv',sel,'minimum_scenario_MAE')+'–'+value(PRED/'audit/scenario_sensitivity.csv',sel,'maximum_scenario_MAE'))
            sensitivity.append('| '+m+('（过程）' if tier=='process_calibrated_12' else '（粗化学）')+' | '+' | '.join(cols)+' |')
    blocks.extend(['## 9. 预先固定输入情景的敏感性\n\n下表full5共同查询nH MAE的最小–最大值，三个输入全部报告，不选取最好速度或化学值作为新主结果。它们是条件情景范围，不是置信区间。\n\n'+'\n'.join(sensitivity),
r'''## 10. 独立核验与下一项预测研发

83985项独立协议检查核对模型/代码/输入哈希、来源排除、7320请求、锚点身份与标签隔离、完整查询集合、单位剂量控制、三个性质及相同输入下界。全部通过；它们证明当前记录和数值执行一致，不证明材料模型正确。所有失败/拒绝规则保留，本轮实际拒绝0。此前冻结阶段共43295个清单项SHA未变。

本轮没有证据支持“生长日志一取得就能训练高可信微观过程模型”。但仍有具体的预测改进路径：优先核对Fig3两个SIMS堆栈的源温/层序列和化学平台值，检验独立化学数据能否约束过程到掺入量的共享有效关系；然后将其接入同一电子/输运Sandbox，并与同输入、同化学数据和同锚点的简单/直接模型比较。只有两个相关堆栈时不能声称跨来源过程泛化。见docs/ML_SANDBOX_CHEMICAL_PROCESS_NEXT_20261018.md。

过程守恒应从实际供给与库存推导。例如候选有效表面库存G、有效掺入分数eta和正速率kappa满足

\[
\dot G=\eta J-\kappa G,\quad J_{\rm inc}=\kappa G,\quad J_{\rm loss}=(1-\eta)J,\quad
\dot G+J_{\rm inc}+J_{\rm loss}=J.
\]

G为atoms/cm²，J及各通量为atoms cm⁻²s⁻¹，kappa为s⁻¹，eta无量纲且在[0,1]。该式是下一阶段候选有效库存模型，尚未实现或验证；不等于已识别真实表面吸附/解吸，也不自动保证详细平衡。若只有稳态平台，kappa不受约束，应移除时间层。必须先解决印刷通量本身已含掺入约定的问题，避免重复乘掺入因子。

不得用已看过的Hall结果反推温度、速度或化学量再声称预测改善。尚未建立的Ge全文/化学配对和日志身份继续作为未满足输入条件；原29/5及全部不利证据保留。整体goal仍active；本次报告和解释本身不足以结束研发。

复现入口：phase23_process_audit.py、phase23_process_gates.py、phase23_source_prediction.py、phase23_prediction_audit.py。预测器、共享权重和来源先验均来自冻结v8/v9包；运行时禁止读取查询目标表或拟合档。报告表格的来源路径、筛选、SHA和值保存在report_claims.json。报告生成器初版字符串语法失败及日志保留在report_v1_syntax_failure；首次数值报告PDF的上标负号字体缺失保留在report_v2_font_failure。最终单位改用显式数学排版，不改变预测。'''])
    report=OUT/'ML与Sandbox_过程输入与新来源预测.md';source='\n\n'.join(blocks)+'\n'
    source=source.replace('cm⁻³',r'\(\mathrm{cm}^{-3}\)').replace('atoms cm⁻²s⁻¹',r'\(\mathrm{atoms\,cm^{-2}\,s^{-1}}\)').replace('s⁻¹',r'\(\mathrm{s}^{-1}\)')
    report.write_text(source)
    e.write(OUT/'report_claims.json',dict(numeric_table_claims=len(CLAIMS),report_sha256=e.sha(report),claims=CLAIMS))
    print('phase23 report',len(CLAIMS),'traceable table values',flush=True)


if __name__=='__main__':main()
