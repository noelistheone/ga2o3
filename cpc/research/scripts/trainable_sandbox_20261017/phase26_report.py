"""Build traceable full prediction results after actual evidence is reviewed."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

OUT=e.BASE/'phase26_pool_share';PRIMARY='ShareEffectiveProcess_internal';CLAIMS=[];CACHE={}
NEW=(PRIMARY,'ShareEffectiveConstant','ShareEffectiveDose','ShareEffectiveProcess',
    'ShareEffectiveConstant_internal','ShareEffectiveDose_internal','ShareGrowthConstant','ShareGrowthDose','ShareGrowthProcess',
    'ShareGrowthConstant_internal','ShareGrowthDose_internal','ShareGrowthProcess_internal',
    'GrowthReference','ActivationReference','ActivationConstant','ActivationDose','ActivationProcess')
OLD=('FixedPhysical','InventoryConstant','InventoryDose','InventoryProcess','InventoryConstant_internal',
    'InventoryDose_internal','InventoryProcess_internal','OutputConstant','OutputDose','OutputProcess','DirectDose','DirectProcess',
    'DirectRidge','UnitDose','MedianActivation','DoseLine','DoseLineAffine','RidgeAffine','PhysicalAffine')
VARIANTS=NEW+OLD
CHAIN=(PRIMARY,'ShareEffectiveProcess','FixedPhysical','InventoryProcess_internal','ActivationProcess','MedianActivation','UnitDose','DirectRidge')
AUDITS=('numerical_gates.json','learning_gates_v2.json','reachability/complete.json','source_gates.json',
    'source_smoke/audit.json','interface_smoke/audit.json','training_audit.json','prediction/audit.json',
    'delivery/cohort_replay/audit.json','external_Azizie/audit.json','analysis/audit.json','physical_scope/audit.json',
    'delivery/final_checks/audit.json','parent_verification_working.json')


def value(path,selector,column,fmt='.6f'):
    if path not in CACHE:CACHE[path]=read(path)
    d=CACHE[path]
    for k,v in selector.items():d=d[d[k].eq(v)]
    assert len(d)==1,(path,selector,len(d))
    x=float(d[column].iloc[0]);CLAIMS.append(dict(path=str(path.relative_to(ROOT)),sha256=e.sha(path),
        selector=selector,column=column,value=x if np.isfinite(x) else None))
    return format(x,fmt) if np.isfinite(x) else '未完整覆盖'


def table(scheme='cluster',scope='common20',column='DOI_macro_nH_MAE',variants=VARIANTS):
    rows=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for v in variants:
        rows.append('| '+v+' | '+' | '.join(value(OUT/'analysis/all_method_metrics.csv',
            dict(scheme=scheme,variant=v,budget=b,scope=scope),column) for b in (0,1,2))+' |')
    return '\n'.join(rows)


def external_table(context,tier,variants=VARIANTS):
    budgets=(0,1) if tier=='rounded_same_Hall_2' else (0,1,2)
    header='| 方法 | '+' | '.join(str(b)+'锚点' for b in budgets)+' |'
    rows=[header,'| '+' | '.join('---' for _ in range(1+len(budgets)))+' |']
    for v in variants:
        rows.append('| '+v+' | '+' | '.join(value(OUT/'external_Azizie/summary.csv',
            dict(context=context,variant=v,input_tier=tier,budget=b,scope='common_query'),'nH_MAE') for b in budgets)+' |')
    return '\n'.join(rows)


def figures():
    d=read(OUT/'analysis/all_method_metrics.csv');folder=OUT/'figures';folder.mkdir(exist_ok=True)
    allvalues=d[d.scope.eq('common20')].DOI_macro_nH_MAE.to_numpy(float);vmax=max(float(np.nanmax(allvalues)),.05)
    fig,axes=plt.subplots(1,2,figsize=(10.5,12),sharey=True)
    cmap=plt.colormaps['YlGnBu'].copy();cmap.set_bad('#dddddd')
    for ax,scheme in zip(axes,('DOI','cluster')):
        z=np.array([[d[d.scheme.eq(scheme)&d.variant.eq(v)&d.budget.eq(b)&d.scope.eq('common20')].DOI_macro_nH_MAE.iloc[0]
            for b in (0,1,2)] for v in VARIANTS])
        im=ax.imshow(np.ma.masked_invalid(z),cmap=cmap,vmin=0,vmax=vmax,aspect='auto')
        for i in range(len(VARIANTS)):
            for j in range(3):ax.text(j,i,f'{z[i,j]:.3f}' if np.isfinite(z[i,j]) else 'invalid',ha='center',va='center',
                color='white' if z[i,j]>.55*vmax else 'black',fontsize=7)
        ax.axhline(16.5,color='black',linewidth=.6);ax.set_xticks(range(3),['0','1','2']);ax.set_yticks(range(36),VARIANTS,fontsize=7)
        ax.set(xlabel='Hall anchors per DOI',title='DOI exclusion' if scheme=='DOI' else 'Related-group exclusion')
    axes[0].get_yticklabels()[0].set_fontweight('bold');axes[1].tick_params(labelleft=False)
    fig.subplots_adjust(left=.35,right=.84,wspace=.16,top=.96,bottom=.06)
    cax=fig.add_axes([.87,.20,.024,.60]);fig.colorbar(im,cax=cax,label='Common-twenty DOI-macro MAE (dex)')
    for ext in ('png','pdf'):fig.savefig(folder/f'all36_common20.{ext}',dpi=180)
    plt.close(fig)
    fd=read(OUT/'analysis/all_method_per_family.csv');families=sorted(fd.family.unique())
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8))
    methods=(PRIMARY,'FixedPhysical','InventoryProcess_internal','ActivationProcess','MedianActivation','DirectRidge')
    for ax,b in zip(axes,(0,2)):
        for v in methods:
            q=fd[fd.scheme.eq('cluster')&fd.scope.eq('common20')&fd.budget.eq(b)&fd.variant.eq(v)].set_index('family')
            ax.plot(range(len(families)),[q.loc[f].nH_MAE for f in families],'o-',ms=3,label=v)
        ax.set_xticks(range(len(families)),[f.split('/')[-1] for f in families],rotation=35,ha='right',fontsize=7)
        ax.set(title=f'{b} anchors per DOI',ylabel='Common-query MAE (dex)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'fixed_family_diagnostics.{ext}',dpi=180)
    plt.close(fig)
    e.write(folder/'provenance.json',dict(summary_sha256=e.sha(OUT/'analysis/all_method_metrics.csv'),
        family_sha256=e.sha(OUT/'analysis/all_method_per_family.csv'),heatmap_all36=True,
        family_plot_methods=methods,empty_values_retained=True,no_new_predictions=True))


def main():
    audits={}
    for p in AUDITS:
        a=json.loads((OUT/p).read_text());assert a['passed'],p;audits[p]=a
    notes=OUT/'analysis/predictive_conclusions.md';receipt=OUT/'analysis/conclusion_receipt.json'
    assert notes.exists() and receipt.exists(),'Inspect complete actual results and save evidence-specific conclusions first'
    assert json.loads(receipt.read_text())['notes_sha256']==e.sha(notes)
    metrics=read(OUT/'analysis/all_method_metrics.csv');assert set(metrics.variant)==set(VARIANTS) and len(VARIANTS)==36
    figures();train=audits['training_audit.json'];analysis=audits['analysis/audit.json'];scope=audits['physical_scope/audit.json']
    replay=audits['delivery/cohort_replay/audit.json'];export=json.loads((OUT/'delivery/export_complete.json').read_text())
    external=json.loads((OUT/'external_Azizie/execution_complete.json').read_text())
    blocks=['# ML与Sandbox：绝对份额学习与跨来源掺杂性质预测',r'''## 1. 本轮研究问题与证据等级

本轮检验一个明确的预测假设：直接学习孤立掺杂原子份额的低维共享logit，经过真实缺陷、电荷、费米输运求解，是否能改善跨来源Hall载流子响应，同时保留迁移率和派生电导输出。主方法ShareEffectiveProcess_internal已在外层分数产生前指定。

33个条件、6 DOI、5保守关联组、原有锚点角色和common20全部保持不变。它们已经在之前研发中被查看，当前属于回顾性研究。当前阶段的内外层代码隔离不消除跨阶段设计选择可能产生的偏乐观；本报告不将此称为全新盲测、独立实验室确认或稀土验证。

新17变体与旧19强对照同时报告。旧预测整行继承且核对，不重复拟合。主要监督仍为nH，未新增mu或变温标签，也未新增DFT/MD或实体测量。''']
    theory=ROOT/'docs/ML_SANDBOX_POOL_SHARE_THEORY_20261018.md'
    text=theory.read_text();body='\n'.join(text.splitlines()[1:]);body=body.replace('\n## ','\n### ')
    blocks.append('## 2. 实际方法的定义、数学推导与边界\n'+body)
    blocks.append(f'''## 3. 数值进入条件与实际训练

373项状态/导数门槛、121项修复后损失门槛、110项来源门槛、293项来源隔离/回放检查和61项部分接口检查通过。293项包含117点回放和36个目标函数核对，61项包含27点回放；不能将这些计作新材料样本。

首次109项损失检查有4项失败，原代码/日志/JSON全部保留。名义生长参考在近补偿条件下的小步差分被求解噪声掩盖；两组独立宽步Richardson外推保持训练公式及原3e-5门槛，得到121项通过。训练只使用修复后版本，没有覆盖首次失败。

先于标签描述固定的5544状态扫描有3110个符合实现的观测域，132个同状态核对支持新旧来源导数相同。离散极值间包络不等于连续可达性，更不是留出预测精度。

实际训练{train['fit_records']}条记录，{train['actual_nonlinear_fits']}次数值非线性拟合、{train['actual_optimizer_starts']}次优化启动。初始未停止{train['initial_not_stopped']}次，按统一上限细化后，选中未停止{train['selected_not_stopped']}条、训练域无效{train['training_invalid']}条。停止不认证全局最优；完整性审计通过不表示所有候选能服务预测。

来源阶段实际{analysis['source_tasks']}任务、{analysis['actual_source_point_evaluations']}次点调用（包括经验模型调用）、{analysis['actual_source_vectors']}个候选向量；{analysis['point_failure_records']}个点失败和{analysis['invalid_candidate_solutions']}个无效候选保留。合并{analysis['combined_point_rows']}行，数值{analysis['combined_numeric']}、域有效{analysis['combined_valid']}。无效候选及覆盖不足不参与选择性全队列MAE。''')
    blocks.append('## 4. 主要预测结果：固定共同20查询\n\n误差单位dex，为log10绝对误差的来源宏平均。全部36方法保留；“未完整覆盖”不是零误差。带_internal的变体改变来源份额，其零锚点与对应基础方法相同，不重复视作独立证据。\n\n### DOI排除：6 DOI等权\n\n'+table('DOI'))
    blocks.append('### 保守关联组排除：6 DOI等权\n\n'+table('cluster'))
    blocks.append('![全部36方法共同查询误差；横线以上为17新变体，灰色表示未完整覆盖。](figures/all36_common20.png)')
    blocks.append('### 同一组排除预测：5关联组等权\n\n先平均关联组内DOI，再等权平均5组。与上表差异来自汇总权重，未改变任何预测。\n\n'+table(column='group_macro_nH_MAE'))
    blocks.append('### 全部查询：33/26/20条件\n\n预算增加后查询组成变化，应结合固定common20判断锚点效果。DOI方案和完整覆盖数字另见analysis/all_method_metrics.csv。\n\n'+table(scope='all_query'))
    blocks.append('![6个预先具名方法逐来源诊断；其余全部方法保留于CSV和主表。](figures/fixed_family_diagnostics.png)')
    rows=['| 主方法减去控制的5组等权MAE | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for control in ('FixedPhysical','InventoryProcess_internal','ActivationProcess','ShareEffectiveProcess','MedianActivation','UnitDose','DirectRidge','PhysicalAffine'):
        rows.append('| '+control+' | '+' | '.join(value(OUT/'analysis/primary_matched_comparisons.csv',
            dict(scheme='cluster',budget=b,control=control),'delta_group_macro') for b in (0,1,2))+' |')
    blocks.append('## 5. 内部共享学习与来源适应的增量\n\n负差值表示主方法误差较小。完整主方法对其余35控制、两个方案、三个预算的逐来源改善/持平/恶化数、最差恶化和删一组敏感性均见CSV；这些是小来源队列的描述，不是多重比较校正后的显著性检验。\n\n'+'\n'.join(rows))
    shared=json.loads((OUT/'prediction/shared_choices.json').read_text());outer=[r for r in shared if r['held'] is not None and r['method']=='ShareEffectiveProcess']
    fallback=sum(r['reg'] is None for r in outer)
    blocks.append(f'主共享工艺族在11个外层划分中，{fallback}个选择显式参考回退。其实际参数、数值秩、训练资格和来源政策见selected_fit_diagnostics.csv与selected_policies.csv。不同来源和预算不可相互替换；11个划分也不是11项独立材料研究。')
    blocks.append('## 6. 完整性质链\n\n全队列9组mu配对/3 DOI，共同查询4组/3 DOI；所有预算固定该共同配对子集。下面8个方法是报告脚本预先具名的诊断，全部36方法原生mu、共同CT和派生sigma结果均在完整指标CSV。无独立sigma标签，不能将代数派生输出当第三次独立验证。\n\n### 原生或经验mu MAE\n\n'+table(column='native_mu_MAE',variants=CHAIN))
    blocks.append('### 统一CT(predicted nH)的mu MAE\n\n这里使用预测nH，不能混同给定实测nH的条件迁移率任务。\n\n'+table(column='CT_mu_MAE',variants=CHAIN))
    blocks.append('### 原生派生sigma MAE\n\nnH与mu误差可相消，故须与前两项联合解读。\n\n'+table(column='native_derived_sigma_MAE',variants=CHAIN))
    context='cluster::without:related_AFRL_Cornell_Wright'
    blocks.append('## 7. 到既有Azizie条件的冻结转移\n\n共享权重和来源超参数不重选，Hall锚点只作允许的来源适应。A为e/k两条近似同Hall化学输入，B为12条过程化学校准代理；e/k重叠，总共仍是12个Hall条件，A/B也不是纯化学输入消融。A的共同查询为k，B共同10个；实际O3与缺失H/热历史采用声明的有效氧/未知零情景，300K为比较约定。来源关系排除不等于已经验证独立实验室。\n\n### 删除整个关联组：B任务共同10查询，全部36方法\n\n'+external_table(context,'learned_process_proxy_12'))
    blocks.append('### 删除整个关联组：A任务共同高剂量k查询\n\n输入和Hall值均是公开印刷约数；UnitDose恰好相等不能证明真实100%激活。\n\n'+external_table(context,'rounded_same_Hall_2'))
    for context in ('DOI::full','cluster::full'):
        blocks.append('### '+context+'：B任务固定8方法诊断\n\n完整36方法均在external_Azizie/summary.csv，不能只据full上下文得出跨来源结论。\n\n'+external_table(context,'learned_process_proxy_12',CHAIN))
    blocks.append(f"实际转移{external['group_requests']}组/{external['rows']}点，数值{external['numeric']}、有效{external['valid']}。a/j输入相同而标签不同的确定性误差下界保留在identical_input_bound.csv；它不唯一解释全部外推失配。")
    blocks.append(f'''## 8. 保存模型与物理适用域

模型包实际{export['presets']}预设/{export['models']}参数文件，其中{export['unqualified_presets']}个预设未满足服务资格，按协议保留并拒绝输出。旧文件按SHA继承，模型包不含查询标签，不复制第三方能量数据库。

独立公共接口回放{replay['requests']}组/{replay['point_rows']}点；{replay['valid_property_vectors_compared']}个有效三性质向量核对，{replay['unaccepted_numeric_continuations_compared']}个未接受的数值延续核对，最大有效对数差{replay['maximum_log_property_difference']:.3e}。数值/域标志及输入读许可分别检查。它证明复用一致，不是新材料准确性。

完整上下文7个新物理族×2上下文×33已知输入共{scope['states']}槽位。未资格模型{scope['unqualified_model_points_not_evaluated']}点拒绝评估；{scope['numeric_states']}点数值完成。独立512点输运重构最大差{scope['maximum_independent_mu_reconstruction_error']:.3e}。

超过半数电流权重位于k乘平均自由程小于1的状态{scope['majority_current_weight_kl_lt1']}个，位点计数包络超过1的状态{scope['shared_site_claim_exceeds_one']}个，有效Tf高于名义生长温度{scope['effective_Tf_above_growth']}个。以上无过滤诊断不修改主分数，也不是Born、真实位点或热历史证书。''')
    blocks.append('## 9. 实际证据判断与下一项预测改进\n\n'+notes.read_text())
    blocks.append('## 10. 复现与未完成范围\n\n实际依赖顺序：training→training_audit→prediction→prediction_audit→export→cohort_replay→external→external_audit→analysis/physical_scope/delivery_final_check。状态/损失/来源门槛在训练或推断前运行，全部协议SHA需先核对；不要覆盖冻结结果或因观察超时重启活跃任务。CSV用csv_io_v2.read_result_csv读取。\n\n公共CLI为predict_pool_properties_v1.py；显式输入字段、未知情景、最多2个Hall锚点和实际模型资格由接口核对。零锚点可导出固定参数局部IFT，不包含CV或来源重拟合导数。本阶段不验证新稀土、真实微观反应率、寿命或光暗电流比。\n\n数学说明不自动证明首创性，文献阅读范围沿用docs/ML_SANDBOX_CONTEXT_PRIOR_ART_20261018.md。系统goal仍active，以真实掺杂性质预测为研发终点；下一个模型选择必须有具体证据依据，不预设最终精度优势或发表结果。')
    dest=OUT/'ML与Sandbox_绝对份额学习与性质预测.md';report='\n\n'.join(blocks)+'\n'
    assert not any(ord(c)<32 and c!='\n' for c in report);dest.write_text(report)
    jsonpaths=(*AUDITS,'delivery/export_complete.json','external_Azizie/execution_complete.json',
        'prediction/shared_choices.json','analysis/conclusion_receipt.json')
    e.write(OUT/'report_claims.json',dict(report_sha256=e.sha(dest),claims=CLAIMS,primary=PRIMARY,
        source_code_sha256=e.sha(Path(__file__)),theory_path=str(theory.relative_to(ROOT)),theory_sha256=e.sha(theory),
        json_audit_sources={str((OUT/p).relative_to(ROOT)):e.sha(OUT/p) for p in jsonpaths},
        no_new_predictions_or_fits=True,outer_method_reselected=False,goal_complete=False))
    print('Wrote actual phase26 predictive report',len(CLAIMS),'traceable CSV values;goal active',flush=True)


if __name__=='__main__':main()
