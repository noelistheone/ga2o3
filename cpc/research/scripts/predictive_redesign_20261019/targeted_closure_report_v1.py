"""Write the bounded experiment decision from actual tables, without new fits."""
from pathlib import Path
import sys,json
from collections import defaultdict
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import targeted_closure_model_v1 as m

ROOT=c.ROOT.parent
OUT=m.OUT
DOC=ROOT/'轻量补强实验与定稿判断_20261021.md'
PAPER=c.ROOT/'paper/npj/targeted_closure_20261021_v1'


def selected(table,**fields):
    q=table
    for name,value in fields.items():q=q[q[name].eq(value)]
    assert len(q)==1,fields
    return q.iloc[0]


def source_tables(folder):
    p=c.read(folder/'predictions.csv');table=c.cases();external=c.read(c.BASE/'external_rate_cohort_v2/scenarios24.csv')
    meta=pd.concat([table,external],ignore_index=True).set_index('case_id')
    rows=[]
    for cohort in p.cohort.unique():
        for scope in (('all_query','common20') if cohort=='old_cohort_heldout' else ('query',)):
            q=p[p.cohort.eq(cohort)].copy()
            if scope=='common20':q=q[q.primary_common20.eq(True)]
            for (variant,family),z in q.groupby(['variant','family']):
                r=dict(cohort=cohort,scope=scope,variant=variant,family=family,conditions=z.sample_id.nunique(),
                    logdose_min=float(np.log10(meta.loc[z.case_id,'incorporated_cm3']).min()),
                    logdose_max=float(np.log10(meta.loc[z.case_id,'incorporated_cm3']).max()))
                r['nH_signed_bias']=float(z.assign(bias=z.predicted-z.observed).groupby('sample_id').bias.mean().mean())
                for endpoint in ('nH','mu','rho'):r[endpoint+'_MAE']=c.group_score(z.to_dict('records'),endpoint,paired=endpoint!='nH',group_equal=False)
                rows.append(r)
    frame=pd.DataFrame(rows);frame.to_csv(folder/'by_source.csv',index=False);return frame


def old_grid_counts():
    stats=defaultdict(lambda:dict(nodes=0,feasible=0,feasible_amax_positive=0,better_training_nodes=0))
    base=c.BASE/'proposal_followup_20261021_v1/constrained_block_refinement_v2/records'
    for path in base.glob('*.json'):
        j=c.document(path);parent=c.document(c.ROOT/j['parent_path']);value=parent['baseline_rho_objective'];r=stats[j['method']]
        for p in j['probes']:
            r['nodes']+=1;r['feasible']+=int(p['feasible']);r['feasible_amax_positive']+=int(p['feasible'] and p['beta'][1]>0)
            r['better_training_nodes']+=int(p['feasible'] and p['objective']<value-1e-7)
    result=dict(stats);c.write(OUT/'old_grid_feasibility_diagnostic.json',result);return result


def main():
    assert c.document(OUT/'independent_audit_v1/audit.json')['passed']
    inventory=c.read(OUT/'inventory_training_v1/summary.csv');rate=c.read(OUT/'rate_scale_v1/summary.csv')
    sources=source_tables(OUT/'inventory_training_v1');rate_sources=source_tables(OUT/'rate_scale_v1');grid=old_grid_counts()
    title='# 轻量补强实验与定稿判断\n\n'
    text=title+'本轮采用“先做两项有限实验，再决定模型与论文写法”的路线。主要目标仍是可训练 ML＋Sandbox 预测掺杂性质。未创建持久 goal，未覆盖上一轮文档、论文或代码；这是回顾性的开发与验证，不能称盲测。\n\n'
    text+='## 1 先给出结论\n\n'
    text+='两项机制在数学上都可实现，但本轮没有产生新的稳定预测增益：软铰链没有改善原线性共享库存；公共率缩放严格保住了 nH，却没有学到可跨来源迁移的输运修正。维持原共享库存与变温内部关系作为当前已支持的预测实例，不把新候选升级为主模型。\n\n'
    text+='本轮新增 16 个上游拟合：6 个软铰链（5 个关联组留出＋1 个全训练）和10个三组线性库存（用于第二阶段的严格内层来源排除），35次优化器启动。输运阶段为87次岭回归线性求解、99份常数／岭模型记录；两类数量不应混称为新增物理样品。两项计算均在 CPU 完成。\n\n'
    text+='## 2 软铰链：实现、假说与真实结果\n\n'
    text+='### 2.1 同容量替换了什么\n\n'
    text+='设 ℓ 是以 cm⁻³ 为数值单位的掺入浓度的十进制对数。固定阈值 c=18.4、平滑宽度 Δ=0.25 dex，不按新外层或外部误差选参。将旧特征 (ℓ−18)/2 换为\n\n'
    text+=r'''\[
h(\ell)=\frac{\Delta}{2}\left[\ln(1+e^{(\ell-c)/\Delta})-\ln(1+e^{(18-c)/\Delta})\right].
\]

\[
h(18)=0,\quad h'(\ell)=\frac12\operatorname{sigmoid}((\ell-c)/\Delta).
\]

'''
    text+='这样高浓度斜率尺度与原特征相同，低浓度导数趋于零；仍为六参数、同库、同标签和同损失。剂量系数允许正负，阈值没有作为第七参数学习。它只让剂量依赖变平：截距、生长、元素、有效温度、背景及电荷可行性门槛仍起作用，不能保证低浓度回到两参数物理参照。\n\n'
    text+='完整参数及输入导数以旧物理解的精确重参数链传播。66项前置检查通过，最大导数差分误差8.55×10⁻⁸以内。第一版四个标签隔离判据因重复求根1.3×10⁻¹⁵—1.6×10⁻¹⁴的舍入差未通过；原模型不改，第二版增加输入键完全相同检查并采用1×10⁻¹¹读出容差。原失败与训练前拒绝记录均保留。\n\n'
    text+='### 2.2 固定共同查询与外部结果\n\n'
    text+='以下均为来源等权 MAE，单位 dex；μ只在对应真实配对条件上计算。\n\n| 队列 | 模型 | nH MAE | μ MAE | 派生σ MAE |\n| --- | --- | ---: | ---: | ---: |\n'
    for cohort,scope,label in [('old_cohort_heldout','common20','共同20条件'),('external3_conditional','query','外部3条件')]:
        for variant,name in [('FixedNHReference','两参数物理参照'),('LinearShared','原线性共享库存'),('DiluteHinge','软铰链共享库存')]:
            r=selected(inventory,cohort=cohort,scope=scope,variant=variant)
            text+=f'| {label} | {name} | {r.nH_MAE:.5f} | {r.mu_MAE:.5f} | {r.rho_MAE:.5f} |\n'
    q=sources[sources.cohort.eq('old_cohort_heldout')&sources.scope.eq('common20')].pivot(index='family',columns='variant',values='nH_MAE')
    wins=int((q.DiluteHinge<q.LinearShared-1e-10).sum());losses=int((q.DiluteHinge>q.LinearShared+1e-10).sum())
    text+=f'\n软铰链相对原线性库存，6个DOI中{wins}个改善、{losses}个变差。它仍优于该队列物理参照的平均误差，但未证明相对原共享模型的新增收益；结论只针对本次固定阈值、宽度及模型族，不能推为所有非线性库存函数无效。\n\n'
    text+='## 3 来源分解纠正了原来的物理归因\n\n'
    text+='### 3.1 主要平均增益不是压低过高的预测\n\n'
    text+='下表带符号偏差定义为预测减观测；正值为高估，负值为低估。每个物理条件先平均缺失输入情景，再在同来源内平均。\n\n| 来源与队列 | 模型 | 带符号nH偏差 / dex | nH MAE / dex |\n| --- | --- | ---: | ---: |\n'
    for cohort,scope,fam,label in [('old_cohort_heldout','common20','10.1063/1.5047214','1.5047214，共同查询'),
        ('external3_conditional','query','10.1063/1.5017616','Rafique，外部'),('external3_conditional','query','10.1063/1.5109678','Feng，外部')]:
        for variant,name in [('FixedNHReference','物理参照'),('LinearShared','线性库存'),('DiluteHinge','软铰链')]:
            r=selected(sources,cohort=cohort,scope=scope,family=fam,variant=variant)
            text+=f'| {label} | {name} | {r.nH_signed_bias:+.5f} | {r.nH_MAE:.5f} |\n'
    text+='\n1.5047214的参照是明显低估，共享模型把低估减轻；不能把0.607 dex改善解释成“增加补偿压低原来过高的载流子”。外部Rafique为低估，Feng为高估，也不支持两篇都由过度补偿低估的统一解释。这里没有据此唯一认定替代微观机制。\n\n'
    text+='### 3.2 关联组留出不是同一条全局曲线\n\n'
    text+='每个外层关联组使用不同的四组训练模型。评价1.5047214时，该DOI及其关联组标签不参与其模型训练，因此它自身的标签不能被说成该留出模型补偿斜率的直接驱动力。它可以影响其他外层折，但因果归因需要相应消融。\n\n'
    text+='同样，低浓度软铰链只使学习特征的剂量导数变小；真实输出还有参考库存、电荷阈值和冷态统计的直接剂量导数。总体三升三降与剂量范围可以并列展示为探索诊断，不能直接升级为独立验证的“稀薄／补偿相区”定律。\n\n'
    text+='## 4 公共散射率缩放：严格解耦成立，学习泛化未成立\n\n'
    text+='### 4.1 第二阶段的训练与来源隔离\n\n'
    text+='固定上一轮FrozenPrimary（包括明确有限的结合logit−12和零纹理），以d=βᵀφ、a=10ᵈ>0乘全部能量处的全部散射率。φ仅含常数、固定缩放的log剂量、生长温度和Sn指示。监督只有11个配对物理条件的logμ，情景行不当独立样品；没有再把派生σ当第三标签。\n\n'
    text+='常数均值、中位数和特征Ridge均保留。Ridge截距不惩罚，特征惩罚α∈{0.001, 0.01, 0.1, 1, 10, 100, 10⁶}，由训练组内部留出μ误差选择；平分偏向强收缩。十个新增三组库存保证内层被留来源的nH与μ均未进入对应上游或缩放拟合。原五个外层主干不重训。\n\n'
    text+='这个实验检验的是所测元数据和线性尺度关系，不排除其他有物理依据的状态特征或函数族。把a写进内核与对logμ和logσ施加相同残差在这里代数等价，因此不能仅以实现位置宣称新散射机制。\n\n'
    text+='### 4.2 实际结果\n\n| 队列 | 模型 | nH MAE | μ MAE | 派生σ MAE |\n| --- | --- | ---: | ---: | ---: |\n'
    for cohort,scope,label in [('old_cohort_heldout','common20','共同20条件'),('external3_conditional','query','外部3条件')]:
        for variant,name in [('FrozenPrimary','冻结主干'),('RateConstantMean','公共尺度：均值'),('RateConstantMedian','公共尺度：中位数'),('RateFeatureRidge','公共尺度：特征Ridge')]:
            r=selected(rate,cohort=cohort,scope=scope,variant=variant)
            text+=f'| {label} | {name} | {r.nH_MAE:.5f} | {r.mu_MAE:.5f} | {r.rho_MAE:.5f} |\n'
    comp=c.document(OUT/'rate_scale_v1/complete.json');mom=c.document(OUT/'rate_scale_v1/real_state_moment_checks.json')
    text+=f'\nnH逐点变化严格为{comp["maximum_nH_change_dex"]:.1f} dex；{mom["checks"]}项真实拟合状态上的全速率矩重建也全部通过。公共率缩放具有正确的数学保护结构，但学习后μ明显变差，不能用“nH约束锁死了优化器”来解释这一实验，因为此处μ更新从结构上不消耗nH漂移预算。\n\n'
    text+='五个外层上下文有三个选择α=10⁶，部署模型也选择10⁶，说明这些折里的特征修正被压向常数；另外两折选较弱惩罚但没有挽救整体留出误差。常数均值和中位数同样不胜未缩放主干。不能把这个负结果推广为所有特征、所有散射或全部迁移率学习无价值。\n\n'
    text+='公共缩放可把模型推向更差的局域弱散射条件；它并不恢复Born／RTA物理适用域。common20输运仍只有4个配对条件、3个DOI、2个关联组；外部仍为3条件、2论文、1关联团队，均不是新材料盲测。\n\n'
    text+='## 5 对另外两项建议的数学判断\n\n'
    text+='### 5.1 损失增量不能直接开方当作预测漂移\n\n'
    text+=r'''令旧训练残差为r₀，新旧预测差为d，加权矩阵为W，则

\[
\Delta L=(r_0+d)^TW(r_0+d)-r_0^TWr_0
=2r_0^TWd+d^TWd.
\]

'''
    text+='因此ΔL≤0.0001 dex²不蕴含RMS漂移≤0.01 dex；交叉项可正可负。第一阶段是带正则、边界的最优点，也不保证对新增二阶段方向残差正交。原模型还另有最大漂移0.03 dex约束；缺少实测噪声协方差，不能说它比实验噪声低一个数量级。\n\n'
    text+='旧375节点中实际有145个可行，其中91个amax>0，而非全部非零候选都被约束挡回。分解如下（amax>0不等于已产生足够强的实际团簇纹理）：\n\n| 模型 | 节点 | 可行 | 可行且amax>0 | 改善基线训练目标的可行点 |\n| --- | ---: | ---: | ---: | ---: |\n'
    for name in ['ClusterCoupled','UniformBackground','ConstantTexture']:
        r=grid[name];text+=f'| {name} | {r["nodes"]} | {r["feasible"]} | {r["feasible_amax_positive"]} | {r["better_training_nodes"]} |\n'
    text+='\n冻结六个主干参数后放宽二参数约束，并不等价于原八参数联合重训；不能预设一条ε曲线必然连接原0.338／0.778与联合0.544／0.553端点。若以后需要空间Pareto分析，应明确模型族、两种约束及训练／留出坐标，不能先规定有利终点。本轮不扩展这项扫描。\n\n'
    text+='### 5.2 一维混合是合理经验对照，但没有无过拟合保证\n\n'
    text+='将f_phys与logC混合时，消去平移后的设计向量为P(f_phys−logC)。若两条曲线在锚点只差一个常数，该向量为零，连一维都不可识别；非零时也只是一个观测对比，不能唯一解释成“补偿／单位激活”的机制标签。\n\n'
    text+='有限偏移先验、观测误差、λ先验及[0,1]截断都会影响结果；有限偏移先验下的Schur补也不再是自由平移的同一个秩k−1正交投影。秩上界不保证一锚点退化被彻底消除，0.23440对0.23849也不是经验精度上限或无过拟合证明。混合本身属于输出校准，并不自动拥有一个对应的内部库存状态。本轮不把它添加为追逐新均值的第三项实验。\n\n'
    text+='## 6 模型和论文的明确取舍\n\n'
    text+='1. 保留原线性共享库存作为已支持的零锚点预测实例：同库载流子MAE0.41477→0.33772，但逐来源与外部不利结果同时报告。软铰链不替换它。\n'
    text+='2. 保留既有变温内部E(C)实例，与同轮加强Ridge、平移及温度线比较；不把它和室温库存写成一个已联合训练的生产模型。\n'
    text+='3. 公共率缩放作为严格保nH的输运对照及本轮负结果，不写成新增精度贡献。仍可检验有独立物理依据的关系；当前结果没有支持立即通过再调几个门槛得到全链突破。\n'
    text+='4. 将投影秩、IFT和Hall矩关系用于设计与解释，不单独把常规代数命题包装成全新ML定理；主贡献要落到实际学习的内部关系及可复现的预测用途。\n'
    text+='5. 可以开始聚焦预测论文的完整写作，但不采用“全部三项机制已被彻底解释”“空间团簇已解决低迁移率”“两锚点达经验上限”等表述。当前main.tex不覆盖，新英文结果更新单独保存。\n\n'
    audit=c.document(OUT/'independent_audit_v1/audit.json')
    text+=f'## 7 审计、文件与复现\n\n独立复核{audit["checks"]:,}项全部通过，重建评分、加权Ridge系数、内层分数和选参，核对来源排除、标签、完整覆盖及Hall恒等式，并验证前轮531个冻结文件未变。它是实现与记录审计，不是发表保证或新的材料精度认证。\n\n'
    links=[('本文阅读版PDF',DOC.with_suffix('.pdf')),('冻结的本轮计划',c.ROOT/'docs/ML_SANDBOX_TARGETED_CLOSURE_PLAN_20261021.md'),
        ('软铰链全部结果',OUT/'inventory_training_v1/summary.csv'),('库存逐来源结果',OUT/'inventory_training_v1/by_source.csv'),
        ('公共率缩放全部结果',OUT/'rate_scale_v1/summary.csv'),('输运逐来源结果',OUT/'rate_scale_v1/by_source.csv'),
        ('内层来源选参',OUT/'rate_scale_v1/selections.json'),('独立审计',OUT/'independent_audit_v1/audit.json'),
        ('英文Results更新',PAPER/'results_update.md')]
    for label,path in links:text+=f'- [{label}]({path.relative_to(ROOT).as_posix()})\n'
    text+='\n## 8 结果图\n\n'
    text+='![固定队列的两项实验结果](Ga2O3-Sandbox/results/predictive_redesign_20261019/targeted_closure_20261021_v1/figures/closing_results.png)\n\n'
    text+='![逐来源增量与剂量范围](Ga2O3-Sandbox/results/predictive_redesign_20261019/targeted_closure_20261021_v1/figures/source_gains_and_dose.png)\n'
    DOC.write_text(text)
    paper(inventory,rate,sources)
    plots(inventory,rate,sources)
    c.write(OUT/'report_build.json',dict(created_utc=c.e.now(),document=str(DOC.relative_to(ROOT)),new_fits=0,
        notes='tables derived from completed outputs;old reports and drafts unchanged'))
    print(str(DOC))


def paper(inventory,rate,sources):
    PAPER.mkdir(exist_ok=True)
    text='''# Results update after the bounded closing experiments

This supplements the previous Results draft without changing that frozen text or main.tex. The scientific objective remains doping-property prediction using learned relations inside a defect Sandbox. The following candidates are controls that did not establish an additional predictive advantage.

## A low-dose-flat inventory feature does not improve transfer in this test

We replaced the linear dose feature with a centered soft hinge at log10(C/[1 cm^-3])=18.4 and width0.25 dex. The normalization preserves the old high-dose derivative scale1/2, while the low-dose derivative tends to zero. The model retains six parameters, the same defect overlay, labels, bounds and fixed carrier-only regularization. Five related-group holdouts and one full-training fit were evaluated, without selecting a threshold on the outer or external labels. This feature does not force recovery of the physical reference, because the remaining inventory and electronic parameters are still active.

On common20, carrier MAE is0.37052 dex, compared with0.33772 dex for the linear inventory and0.41477 dex for the physical reference. The hinge improves two of six publication-wise errors and worsens four relative to the linear inventory. On the external three-condition cohort it yields0.31257 dex, compared with0.30966 dex for the linear inventory and0.28866 dex for the physical reference. These results do not support replacing the current inventory relation by this fixed hinge.

Signed residuals also change the interpretation of the earlier average gain. For DOI10.1063/1.5047214 the reference carrier bias is−1.22477 dex and the linear-inventory bias is−0.61784 dex: the gain reduces underprediction. It is not evidence that learning reduced an excessive carrier prediction by adding compensation. External Rafique conditions are underpredicted whereas the Feng condition is overpredicted. Furthermore, each outer source is evaluated by a different source-excluded fit. A universal compensation-regime explanation cannot be inferred from the aggregate ranking.

## Exact carrier-preserving rate scaling does not ensure transport transfer

We froze the previous FrozenPrimary models and multiplied every transport rate by a common positive scale a(x)=10^(beta^T phi(x)). The fixed input features are an intercept, scaled logarithmic dose, growth temperature and a Sn indicator. At a fixed electronic state, both time moments scale by the corresponding powers of1/a, preserving the Hall factor and carrier readout exactly; logmu and logsigma both decrease by log10(a). This operation is equivalent to a common observable residual and does not identify a new microscopic scattering mechanism.

We fitted only the paired mobility response from11 physical conditions, comparing weighted-mean and weighted-median constants with feature Ridge. Ridge retains an unpenalized intercept. Its penalty is selected from seven fixed values by inner related-group mobility MAE, using ten newly fitted three-group inventories so that inner held-out carrier labels are also excluded upstream. Together with the six hinge fits, the round has16 new upstream fit records and35 optimizer starts. Scale fitting adds87 Ridge linear solves and99 constant/Ridge records, not additional material samples.

All292 rate-control prediction rows preserve nH exactly. Common20 carrier MAE therefore remains0.33772 dex, while mobility errors are0.77785 dex for no scaling,1.26778 dex for the mean constant,0.90837 dex for the median constant, and1.34792 dex for nested feature Ridge. On the external cohort, mobility error increases from0.20678 dex to0.74079,0.60732 and0.74079 dex, respectively. Feature Ridge selects the largest penalty in three of five outer contexts and for full-training deployment. Thus the exact mathematical decoupling works, but the tested metadata-scale relation does not supply a transferable transport correction.

## Limits of the mechanistic interpretation

For baseline residual r0 and prediction drift d, the training-loss increment is2 r0^T W d+d^T W d; its square root is not an RMS-drift bound. In addition, the previous block experiment imposed an independent0.03 dex maximum-drift constraint. Of375 previous grid nodes,145 were feasible, including91 with positive maximum texture amplitude. Consequently, the absence of a useful secondary fit cannot be attributed solely to all nonzero candidates being excluded. Relaxing a secondary loss budget with the primary parameters frozen also need not recover the endpoint of joint eight-parameter training.

The offset-profiled rank bound is a method-design lemma, not a guarantee of monotonic query accuracy or unique microscopic inference. A one-dimensional mixture of physical and dose predictions may itself have a zero projected design. Even when nonzero, its fitted direction does not uniquely identify a compensation regime. The earlier0.23440 versus0.23849 dex comparison is not an empirical accuracy ceiling or a proof against overfitting.

We retain the demonstrated shared-inventory and temperature-dependent internal-statistics experiments as the prediction cases. The hinge and rate controls delimit the present constitutive choices. Common20 transport still contains only four paired conditions from two related groups, and the external set is a retrospective three-condition test from one related team. No universal joint carrier-and-transport gain or unique spatial-barrier mechanism is established.

## Evidence

- [Hinge and matched controls](../../../results/predictive_redesign_20261019/targeted_closure_20261021_v1/inventory_training_v1/summary.csv)
- [Signed source residuals](../../../results/predictive_redesign_20261019/targeted_closure_20261021_v1/inventory_training_v1/by_source.csv)
- [Rate-scale controls](../../../results/predictive_redesign_20261019/targeted_closure_20261021_v1/rate_scale_v1/summary.csv)
- [Independent audit](../../../results/predictive_redesign_20261019/targeted_closure_20261021_v1/independent_audit_v1/audit.json)
'''
    import re
    text=re.sub(r'\b(is|with|and|yields|DOI|of|from|has|adds|All|remains|are|scaling|error|Of|including|width|scale|previous|to|independent|earlier|versus)(?=[\d−])',r'\1 ',text)
    text=re.sub(r',(?=\d)',', ',text)
    (PAPER/'results_update.md').write_text(text)


def plots(inventory,rate,sources):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
    target=OUT/'figures';target.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,3,figsize=(13,4.4))
    for a,cohort,scope,title in [(ax[0],'old_cohort_heldout','common20','共同20条件：载流子'),(ax[1],'external3_conditional','query','外部3条件：载流子')]:
        names=['FixedNHReference','LinearShared','DiluteHinge'];y=[selected(inventory,cohort=cohort,scope=scope,variant=v).nH_MAE for v in names]
        a.bar(['物理参照','线性库存','软铰链'],y,color=['#7d8792','#2b8068','#b25e48']);a.set_ylabel('来源等权 MAE / dex');a.set_title(title)
        for i,v in enumerate(y):a.text(i,v+.005,f'{v:.3f}',ha='center',fontsize=9)
        a.set_ylim(0,max(y)*1.17);a.grid(axis='y',alpha=.2)
    vals=[selected(rate,cohort='old_cohort_heldout',scope='common20',variant=v).mu_MAE for v in ['FrozenPrimary','RateConstantMean','RateConstantMedian','RateFeatureRidge']]
    ax[2].bar(['不缩放','均值','中位数','特征岭'],vals,color=['#2b8068','#9b8b73','#7d8792','#b25e48']);ax[2].set_title('共同配对4条件：迁移率');ax[2].set_ylabel('来源等权 MAE / dex')
    for i,v in enumerate(vals):ax[2].text(i,v+.025,f'{v:.3f}',ha='center',fontsize=9)
    ax[2].set_ylim(0,max(vals)*1.18);ax[2].grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(target/'closing_results.png',dpi=170);fig.savefig(target/'closing_results.pdf');plt.close(fig)
    q=sources[sources.cohort.eq('old_cohort_heldout')&sources.scope.eq('common20')];pivot=q.pivot(index='family',columns='variant',values='nH_MAE')
    fig,axes=plt.subplots(1,2,figsize=(11,5.5),sharey=True,gridspec_kw={'width_ratios':[1.5,1]});y=np.arange(len(pivot));labels=[x.split('/')[-1] for x in pivot.index]
    for shift,v,label,color in [(-.18,'LinearShared','原线性共享库存','#2b8068'),(.18,'DiluteHinge','软铰链','#b25e48')]:
        gain=pivot.FixedNHReference-pivot[v];axes[0].barh(y+shift,gain,height=.32,label=label,color=color)
    axes[0].axvline(0,color='black',lw=.8);axes[0].set_yticks(y,labels);axes[0].set_xlabel('参照MAE减去共享MAE / dex');axes[0].set_title('正值表示优于参照');axes[0].legend(fontsize=9)
    for i,family in enumerate(pivot.index):
        r=q[q.family.eq(family)].iloc[0];axes[1].plot([r.logdose_min,r.logdose_max],[i,i],color='#4d6984',lw=4,marker='o',ms=4)
    axes[1].set_xlabel(r'$\log_{10}(C/[1\,\mathrm{cm}^{-3}])$');axes[1].set_title('相同共同查询的化学浓度范围')
    for a in axes:a.grid(axis='x',alpha=.2)
    fig.suptitle('来源异质性诊断：剂量范围并不单独证明微观补偿机制',fontsize=12);fig.tight_layout()
    fig.savefig(target/'source_gains_and_dose.png',dpi=170);fig.savefig(target/'source_gains_and_dose.pdf');plt.close(fig)


if __name__=='__main__':main()
