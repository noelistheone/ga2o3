"""Append measured outcomes and make plots from frozen new-round outputs."""
from pathlib import Path
import sys,json,itertools,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

OUT=c.BASE/'proposal_followup_20261021_v1'
ROOT=c.ROOT.parent
DOC=ROOT/'新预测方案的理论核对与执行记录_20261021.md'


def main():
    refined=OUT/'constrained_block_refinement_v2';assert (refined/'complete.json').exists()
    gate=c.read(OUT/'profiled_adaptation_v2/summary.csv');space=c.read(c.BASE/'phase35_association_spatial/related_group_heldout_v1/summary.csv')
    controls=c.read(OUT/'overlay_nH_controls_v1/summary.csv');block=c.read(refined/'summary.csv')
    ext=c.read(OUT/'overlay_nH_controls_v1/external_schema_replay_v2/summary.csv')
    def one(table,**fields):
        q=table
        for k,v in fields.items():q=q[q[k].eq(v)]
        assert len(q)==1,fields
        return q.iloc[0]
    names={'FrozenShared':'冻结共享主干','NestedShift':'嵌套纯平移','NestedProfiledGate':'消元信息门控',
        'NestedPhysicsOrDose':'训练内物理或剂量选择','NestedUnitDose':'匹配单位剂量校准'}
    tail='\n## 9 本轮完成后的实际结果\n\n'
    tail+='### 9.1 门控复核\n\n下表为固定 common20、来源等权 nH MAE，单位 dex。v2 将平移正则规范和网格与原第 29 阶段对齐；v1 的不同规范结果和失败均保留。\n\n'
    tail+='| 方法 | 零锚点 | 一锚点 | 两锚点 |\n| --- | ---: | ---: | ---: |\n'
    for m,label in names.items():
        vals=[one(gate,variant=m,budget=b,scope='common20').nH_MAE for b in (0,1,2)]
        tail+='| '+label+' | '+' | '.join(f'{v:.5f}' for v in vals)+' |\n'
    tail+='\n两锚点门控改善了同主干纯平移，但未超过匹配剂量基线。物理或剂量的内层选择两锚点略优于剂量模型，只有约 0.00409 dex 差，一锚点仍较差。不能将这个结果写成全预算单调优势。零锚点与原结果重放差在舍入尺度。\n\n'
    tail+='第一版无收缩剂量的两锚点误差为 0.26728，而原 0.23849 使用了收缩；v2 匹配网格后精确复现 0.31681／0.23849。旧平移正则作用于归一化锚点损失，不能与直接把同一数值加到 k 样品求和损失的惩罚混同。\n\n'
    tail+='### 9.2 同库、同标签的零锚点主比较\n\n固定配置 nH-only，比较同一 overlay、同一来源划分、同样 nH 标签的两参数参照和六参数共享库存。后者没有 μ 监督，不能据此称完整输运共同获胜。\n\n'
    tail+='| 模型 | nH MAE | 配对 μ MAE | 派生 σ MAE |\n| --- | ---: | ---: | ---: |\n'
    for variant,label in [('FixedNHReference','固定物理参照'),('LearnedNHInventory','共享内部库存')]:
        r=one(controls,cohort='old_cohort_heldout',variant=variant,scope='common20')
        tail+=f'| {label} | {r.nH_MAE:.5f} | {r.mu_MAE:.5f} | {r.rho_MAE:.5f} |\n'
    a=one(controls,cohort='old_cohort_heldout',variant='LearnedNHInventory',scope='common20');b=one(controls,cohort='old_cohort_heldout',variant='FixedNHReference',scope='common20')
    tail+=f'\n共享内部库存的 nH 宏 MAE 下降 {(b.nH_MAE-a.nH_MAE)/b.nH_MAE*100:.2f}%；μ 略差，σ 作为派生一致读出改善。这是值得保留的预测增量，不能全部归给新增复合体库或新分块算法：新增模型还涉及固定正则、库与训练配置；同库参照用于限定归因。\n\n'
    p=c.read(OUT/'overlay_nH_controls_v1/predictions.csv');p=p[p.cohort.eq('old_cohort_heldout')&p.primary_common20.eq(True)]
    per=(p.assign(error=abs(p.predicted-p.observed)).groupby(['variant','related_cluster','family','sample_id']).error.mean()
         .groupby(level=['variant','related_cluster','family']).mean().unstack('variant'))
    per['improvement_reference_minus_learned']=per.FixedNHReference-per.LearnedNHInventory
    per.reset_index().to_csv(OUT/'matched_nH_by_family.csv',index=False)
    wins=int((per.improvement_reference_minus_learned>1e-10).sum());losses=int((per.improvement_reference_minus_learned< -1e-10).sum())
    tail+=f'逐 DOI 为 {wins} 个改善、{losses} 个变差，完整逐来源表单独保存。平均下降不等于每个来源一致改进，也不具有当前样本数尚未支持的总体统计保证。\n\n'
    tail+='### 9.3 第 35 阶段匹配空间比较\n\n下表固定 γ=1、α=0.1、零来源锚点。六十个拟合全部完成、126 个优化器启动、无最终未停止或无效训练拟合；输出 5,292 行。α=0 的敏感性和全部来源适应仍在完整结果中，不按外层挑最优。\n\n'
    tail+='| 同库模型 | nH MAE | 配对 μ MAE | 派生 σ MAE |\n| --- | ---: | ---: | ---: |\n'
    for method in ['OverlayReference','OverlayShareProcess','ConstantAssociation','ClusterCoupled','ConstantTexture','UniformBackground']:
        r=one(space,method=method,alpha=.1,adaptation='none',budget=0,scope='common20')
        tail+=f'| {method} | {r.nH_MAE:.5f} | {r.mu_MAE:.5f} | {r.rho_MAE:.5f} |\n'
    tail+='\nClusterCoupled 改善了一部分配对 μ，却损伤 nH；与 UniformBackground 非常接近，没有证实预设化学—背景纹理关联的独立优势。不能据此写出“团簇势垒已解决强散射”的结论。\n\n'
    tail+='### 9.4 正确分块训练与平坦起点排查\n\n先完成五个同库 nH-only 主干拟合，再冻结六个权重，执行十五个两参数约束子问题。第二阶段使用真实目标与不等式导数，约束训练 nH 最大漂移 0.03 dex、平方损失增加不超过 0.0001 dex²。六个目标／约束导数坐标核验通过，最大误差低于 7×10⁻¹⁰。\n\n'
    tail+='为排查零纹理处的偶对称平坦导数与 −12 结合 logit 的饱和，再对每个子问题检查预先固定 25 个训练可行性网格点；总计 375 个节点，不读取外层标签。只有出现更好可行训练目标才追加 SLSQP，保留原解。这个复核属于同一批数据上的优化诊断，不是独立确认。\n\n'
    tail+='| 分块模型 | nH MAE | 配对 μ MAE | 派生 σ MAE |\n| --- | ---: | ---: | ---: |\n'
    for variant in ['FrozenPrimary','ClusterCoupled','ConstantTexture','UniformBackground']:
        r=one(block,variant=variant,scope='common20')
        tail+=f'| {variant} | {r.nH_MAE:.5f} | {r.mu_MAE:.5f} | {r.rho_MAE:.5f} |\n'
    rr=c.document(refined/'complete.json')
    tail+=f'\n网格复核实际追加 {rr["optimizer_starts"]} 次优化器启动。当前受约束候选仍没有建立输运增量；保护训练约束没有自动产生新物理信息。这里的结合层只沿更多三掺杂中性化方向变化，不能将其失败推广为所有双向 q 平方库存方向均无效。\n\n'
    tail+='### 9.5 外部三条件的冻结模型复算\n\n同一两篇论文、一个关联团队、三个条件和二十四缺失输入情景。新模型只用原 35 条件训练；测试仍是回顾性的已有公开来源，没有新增独立材料。\n\n'
    tail+='| 模型 | nH MAE | μ MAE | 派生 σ MAE |\n| --- | ---: | ---: | ---: |\n'
    for variant,label in [('FixedNHReference','固定物理参照'),('LearnedNHInventory','共享内部库存')]:
        r=one(ext,variant=variant,scope='query');tail+=f'| {label} | {r.nH_MAE:.5f} | {r.mu_MAE:.5f} | {r.rho_MAE:.5f} |\n'
    tail+='\n外部 nH 未胜参照，μ 与 σ 有小幅平均改善。该结果必须与 common20 的有利 nH 结果共同报告。首版外部表缺少旧标签别名导致拒绝；第二版只补 NaN 标签别名并显式擦除 nH 标签，复用保存参数、零重训，原失败输出保留。\n\n'
    audit=c.document(OUT/'independent_audit_v1/audit.json');grid=c.document(OUT/'independent_audit_v1/trained_grid_summary.json')
    tail+='### 9.6 数值和结果审计\n\n'
    tail+=f'第一轮独立审计 {audit["checks"]:,} 项通过，涵盖角色、来源排除、最优有限解、评分重建、单／双锚点秩及查询标签扰动隔离。三种空间模型 α=0.1 的 147 个保存外推状态另作 128→256 网格复算，全部数值返回，最大读出变化 {grid["max_error_dex"]:.8f} dex。\n\n'
    tail+='部分状态仍有很高比例电导／电阻率权重落在局域弱散射风险区。网格稳定不构成弱散射适用性认证；后续附加审计记录同库参照、外部 schema 重放和网格优化复核的完整性。\n\n'
    tail+='### 9.7 对方案的最终判断\n\n本轮支持把不同信息预算和不同观测任务分开，保留可训练库存与变温统计的实际正面结果；没有得到“门控保证全预算领先”或“第 35 阶段必然实现低迁移率解释”的证据。\n\n'
    tail+='可继续发展预测论文，但应先把零锚点内部学习的增量、逐来源差异和变温有效关系写实，输运与空间作为经过严格对照的扩展。能量补全、温度实例和空间实例不包装成一个已统一验证的生产模型。下一项方法改进需要针对实际剩余失配提出新的可检验关系，保持 ML＋Sandbox 预测方向，而非靠换叙述宣布当前方案已成功。\n\n'
    tail+='## 10 文件和复现入口\n\n'
    links=[('第35阶段全部留出结果',c.BASE/'phase35_association_spatial/related_group_heldout_v1/summary.csv'),
        ('匹配信息门控结果',OUT/'profiled_adaptation_v2/summary.csv'),('正确分块与可行网格',refined/'summary.csv'),
        ('同库nH主比较',OUT/'overlay_nH_controls_v1/summary.csv'),('外部零重训schema复算',OUT/'overlay_nH_controls_v1/external_schema_replay_v2/summary.csv'),
        ('第一轮独立审计',OUT/'independent_audit_v1/audit.json'),('主要来源阅读范围',OUT/'primary_literature_notes.json'),
        ('新论文大纲',c.ROOT/'paper/npj/predictive_redesign_20261021/论文主线与证据映射.md')]
    for label,path in links:tail+=f'- [{label}]({path.relative_to(ROOT).as_posix()})\n'
    tail+='\n'
    text=DOC.read_text().split('\n## 9 本轮完成后的实际结果\n')[0];DOC.write_text(text+tail)
    plots(gate,space,controls,block,per)
    print(str(DOC))


def plots(gate,space,controls,block,per):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(8,4.5))
    for v,label in [('FrozenShared','冻结主干'),('NestedShift','纯平移'),('NestedProfiledGate','消元信息门控'),('NestedPhysicsOrDose','训练内物理或剂量'),('NestedUnitDose','匹配单位剂量')]:
        q=gate[gate.variant.eq(v)&gate.scope.eq('common20')].sort_values('budget');ax.plot(q.budget,q.nH_MAE,marker='o',label=label)
    ax.set_xticks([0,1,2]);ax.set_xlabel('允许的物理样品 Hall 锚点数');ax.set_ylabel('common20 来源等权 MAE / dex');ax.set_title('消去平移后的信息门控实际结果');ax.grid(alpha=.2);ax.legend(fontsize=9,ncol=2)
    fig.tight_layout();target=OUT/'figures';target.mkdir(exist_ok=True);fig.savefig(target/'profiled_gate_results.png',dpi=180);fig.savefig(target/'profiled_gate_results.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6));gain=per.improvement_reference_minus_learned
    labels=[i[1].split('/')[-1] for i in per.index];ax.barh(labels,gain,color=['#2f826a' if x>=0 else '#b15545' for x in gain])
    ax.axvline(0,color='black',lw=.8);ax.set_xlabel('固定物理误差减去共享库存误差 / dex');ax.set_title('零锚点同库匹配比较  正值表示共享库存更好');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(target/'matched_nH_source_gains.png',dpi=180);fig.savefig(target/'matched_nH_source_gains.pdf');plt.close(fig)
    DOC.write_text(DOC.read_text()+'\n## 11 结果图\n\n'+
        '![共同查询的信息门控结果](Ga2O3-Sandbox/results/predictive_redesign_20261019/proposal_followup_20261021_v1/figures/profiled_gate_results.png)\n\n'+
        '![同库nH零锚点逐来源增量](Ga2O3-Sandbox/results/predictive_redesign_20261019/proposal_followup_20261021_v1/figures/matched_nH_source_gains.png)\n')


if __name__=='__main__':main()
