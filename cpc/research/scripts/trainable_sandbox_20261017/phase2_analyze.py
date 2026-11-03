"""Join all retained phase2 comparisons, verify contracts and produce figures."""
from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.OUT
def mdtable(d,columns):
    header="| "+" | ".join(columns)+" |\n| "+" | ".join("---" for _ in columns)+" |\n"
    for r in d[columns].itertuples(index=False,name=None):
        vals=[f"{x:.6g}" if isinstance(x,(float,np.floating)) else str(x) for x in r]
        header+="| "+" | ".join(vals)+" |\n"
    return header

def main():
    # Original per-fold outputs preserve exact fit hashes. Older primitive-only
    # aggregations may have parsed a digits-and-e hash as infinity.
    folders=("property","acquired_property","acquired_property_addendum")
    frames=[pd.concat([read_result_csv(f) for f in sorted((OUT/p/"outer").glob("*.csv"))],
                      ignore_index=True) for p in folders]
    metadata_repairs=[]
    for folder,frame in zip(folders,frames):
        old=read_result_csv(OUT/folder/"predictions.csv")
        corrected=dict(zip(zip(frame.method,frame.budget,frame.sample_id),frame.fit_key))
        for row in old.itertuples():
            key=(row.method,row.budget,row.sample_id)
            if str(row.fit_key)!=corrected[key]:
                metadata_repairs.append(dict(folder=folder,method=row.method,sample_id=row.sample_id,
                    budget=row.budget,old_fit_key=str(row.fit_key),correct_fit_key=corrected[key]))
        frame.to_csv(OUT/folder/"predictions_metadata_v2.csv",index=False)
    energy.write(OUT/"metadata_reader_repairs.json",dict(
        error="primitive numeric inference parsed hash-like digits-and-e identifiers as floats",
        corrected_from="original per-fold CSV strings; no model refits and no numeric prediction changes",
        original_aggregations_retained=True,changed_rows=metadata_repairs))
    combined=pd.concat(frames,ignore_index=True)
    keys=["method","budget","sample_id"]
    repeated=combined[combined.duplicated(keys,keep=False)]
    for _,g in repeated.groupby(keys):
        if np.ptp(g.predicted_log_nH.to_numpy(float))>1e-12:
            raise RuntimeError("Copied reference predictions changed")
    combined=combined.drop_duplicates(keys).copy()
    combined.to_csv(OUT/"all_property_predictions.csv",index=False)
    records=[]
    roles=read_result_csv(energy.BASE/"episodes_v1.csv")
    common=set(roles[roles.budget.eq(2)&roles.role.eq("query")].sample_id)
    for (m,b,f),g in combined[combined.role.eq("query")].groupby(["method","budget","family"]):
        for scope in ("all_query","common18"):
            h=g if scope=="all_query" else g[g.sample_id.isin(common)]
            records.append(dict(method=m,budget=int(b),family=f,scope=scope,n=len(h),
                numeric=int(h.predicted_log_nH.notna().sum()),
                physical=int(h.physical_valid.sum()),
                MAE=float(abs(h.predicted_log_nH-h.observed_log_nH).mean())
                    if h.predicted_log_nH.notna().all() else None))
    by=pd.DataFrame(records)
    by.to_csv(OUT/"all_property_by_family.csv",index=False)
    summary=by.groupby(["method","budget","scope"]).agg(families=("family","nunique"),
        query_records=("n","sum"),numeric=("numeric","sum"),physical=("physical","sum"),
        MAE=("MAE","mean"),worst_family=("MAE","max")).reset_index()
    summary.to_csv(OUT/"all_property_summary.csv",index=False)
    # Source exclusion in every fit file; unchanged copied baselines resolve to
    # their original fitting directory.
    method_folder={}
    for name,frame in zip(("property","acquired_property","acquired_property_addendum"),frames):
        for m in frame.method.unique():
            if m not in method_folder:method_folder[m]=name
    invalid=[]
    for row in combined.itertuples():
        f=OUT/method_folder[row.method]/"fits"/(row.fit_key+".json")
        data=json.loads(f.read_text())
        if row.family in data["train_families"]:invalid.append([row.method,row.sample_id])
    energy_pred=read_result_csv(OUT/"energy_predictions.csv")
    energy_leaks=[r.label_id for r in energy_pred.itertuples()
                  if r.element in r.training_elements.split("|")]
    # Exact oracle forward reproduction at the zero-anchor budget. The
    # two-anchor shrinkage grid differs explicitly from phase1.
    old=read_result_csv(energy.BASE/"predictions_v1.csv")
    old=old[old.method.eq("physics_global_process_offset")&old.budget.eq(0)]
    new=combined[combined.method.eq("oracle_full")&combined.budget.eq(0)]
    joined=old.merge(new,on=["sample_id","budget"],suffixes=("_old","_new"))
    oracle_error=float(abs(joined.predicted_log_nH_old-joined.predicted_log_nH_new).max())
    protected=[]
    for file in ("artifact_manifest_phase1.json","protected_inputs_before.json"):
        x=json.loads((energy.BASE/file).read_text())
        entries=x["files"] if isinstance(x,dict) else x
        changed=[r["path"] for r in entries if not (ROOT/r["path"]).exists()
                 or energy.sha(ROOT/r["path"])!=r["sha256"]]
        protected.append(dict(manifest=file,checked=len(entries),changed=changed))
    audit=dict(energy_prediction_rows=len(energy_pred),energy_elements=int(energy_pred.element.nunique()),
        property_methods=int(combined.method.nunique()),property_prediction_rows=len(combined),
        property_query_rows=int(combined.role.eq("query").sum()),
        source_leakage_cases=invalid,energy_element_leakage_cases=energy_leaks,
        oracle_zero_anchor_forward_max_difference_dex=oracle_error,
        oracle_two_anchor_comparison="new shrinkage grid differs from phase1; not an identical adaptation replication",
        protected=protected,
        physical_flag="regular interior root, single-electron Hall convention and nH>=1e13; does not add the historical m_star compensation filter",
        source_count="same five retrospective property families; 15 energy elements are not 15 independent DFT sources",
        new_DFT_jobs=0,new_independent_material_experiments=0,
        repaired_metadata_rows=len(metadata_repairs),model_predictions_changed_by_metadata_repair=False)
    assert not invalid and not energy_leaks and oracle_error<2e-6
    assert all(not r["changed"] for r in protected)
    energy.write(OUT/"contract_audit.json",audit)
    figures(summary)
    report(summary,audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

def figures(summary):
    figdir=OUT/"figures";figdir.mkdir(exist_ok=True)
    e=read_result_csv(OUT/"energy_summary.csv");e=e[e.stratum.eq("all")].set_index("method")
    methods=["global_mean","charge_pair_mean","ridge","rbf","extra_trees"]
    names=["Mean","Charge-pair\nmean","Ridge","RBF","ExtraTrees"]
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].bar(names,[e.loc[m,"element_parent_MAE_eV"] for m in methods])
    axes[0].set_ylabel("Crossing energy MAE (eV)")
    axes[0].set_title("Leave whole element out: 15 elements")
    p=summary[summary.scope.eq("all_query")&summary.budget.eq(0)].set_index("method")
    axes[1].bar(names,[p.loc[m,"MAE"] for m in methods])
    axes[1].axhline(p.loc["oracle_full","MAE"],color="k",ls="--",label="All-energy reference")
    axes[1].set_ylabel("Measured carrier MAE (dex)")
    axes[1].set_title("Energy completion -> 29 records / 5 sources")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(figdir/f"energy_and_property.{ext}",dpi=180)
    plt.close(fig)
    a=read_result_csv(OUT/"acquisition/summary.csv")
    b=read_result_csv(OUT/"acquisition_ablation/summary.csv")
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    floor=1e-6
    for ax,(elem,model) in zip(axes.flat,[("Si","extra_trees"),("Sn","extra_trees"),("Si","rbf"),("Sn","rbf")]):
        for policy,label in (("goal","Learned prior + goal"),("energy_variance","Energy variance"),("random","Random (32 mean)")):
            q=a[a.element.eq(elem)&a.model.eq(model)&a.policy.eq(policy)].sort_values("budget")
            ax.plot(q.budget,np.maximum(q.MAE_mean,floor),marker="o",ms=3,label=label)
            if policy=="random":
                ax.fill_between(q.budget,np.maximum(q.MAE_min,floor),np.maximum(q.MAE_max,floor),alpha=.12)
        for method,cov,label in (("global_mean","diagonal_A","Mean + goal / diagonal A"),
                                 ("ridge","diagonal_xi","Ridge + goal / diagonal xi")):
            q=b[b.element.eq(elem)&b.model.eq(method)&b.covariance.eq(cov)].sort_values("budget")
            ax.plot(q.budget,np.maximum(q.MAE,floor),ls="--",label=label)
        ax.set_yscale("log");ax.set_ylim(floor,3);ax.set_title(elem+" / "+model)
        ax.grid(alpha=.15)
    for ax in axes[-1]:ax.set_xlabel("Additional revealed energy entries")
    for ax in axes[:,0]:ax.set_ylabel("Disagreement with full-energy Sandbox (dex)")
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Virtual published-label replay; values below 1e-6 plotted at the floor",fontsize=11)
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(figdir/f"acquisition_replay_and_controls.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    paths=[
        ("ET_goal_k{}","extra_trees","ET / goal"),
        ("ET_variance_k{}","extra_trees","ET / energy variance"),
        ("mean_goal_k{}","global_mean","Mean / goal"),
        ("Ridge_diagonal_xi_k{}","ridge","Ridge / goal / diagonal xi"),
        ("ET_diagonal_xi_k{}","extra_trees","ET / goal / diagonal xi"),
    ]
    for ax,budget in zip(axes,(0,1,2)):
        q=summary[summary.scope.eq("all_query")&summary.budget.eq(budget)].set_index("method")
        for pattern,start,label in paths:
            values=[q.loc[start,"MAE"]]+[q.loc[pattern.format(k),"MAE"] for k in (2,4)]
            ax.plot([0,2,4],values,marker="o",ms=3,label=label)
        rand=[[q.loc[f"ET_random{seed}_k{k}","MAE"] for seed in (0,1)] for k in (2,4)]
        ax.plot([2,4],[np.mean(v) for v in rand],marker="x",ls=":",label="ET / random (2 mean)")
        ax.axhline(q.loc["oracle_full","MAE"],color="k",ls="--",label="All-energy reference")
        ax.set_title(f"{budget} carrier anchors");ax.set_xlabel("Additional energy entries")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Measured carrier MAE (dex)")
    axes[0].legend(fontsize=6.7)
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(figdir/f"real_carrier_after_acquisition.{ext}",dpi=180)
    plt.close(fig)

def report(summary,audit):
    energy_scores=read_result_csv(OUT/"energy_summary.csv")
    basic=summary[summary.scope.eq("all_query")&summary.method.isin(
        ["oracle_full","global_mean","charge_pair_mean","ridge","rbf","extra_trees"])]
    basic=basic.pivot(index="method",columns="budget",values="MAE").reset_index()
    basic.columns=["method","0 anchors","1 anchor","2 anchors"]
    full=summary[summary.scope.eq("all_query")].pivot(index="method",columns="budget",values="MAE").reset_index()
    full.columns=["method","0 anchors","1 anchor","2 anchors"]
    common=summary[summary.scope.eq("common18")].pivot(index="method",columns="budget",values="MAE").reset_index()
    common.columns=["method","0 anchors","1 anchor","2 anchors"]
    a=read_result_csv(OUT/"acquisition/summary.csv")
    a=a[a.budget.isin([2,4])]
    cols=["element","model","policy","budget","MAE_mean","MAE_min","MAE_max"]
    text="""# 第二阶段：能量学习、Sandbox掺杂性质预测与等预算信息选择

## 1. 这一阶段怎样服务于用户目标

研究方向保持为通过ML＋Sandbox预测掺杂性质。已经实现的链条是：

**元素/位点/电荷特征 → 训练过的缺失能量学习器 → 参考一致的能量重建 → 高温缺陷库存 → 室温电荷再平衡 → Hall载流子预测 → 决定下一条应补充的能量信息。**

本阶段不再仅做可识别性解释，而是实际训练5类能量预测器，接回物理求解，运行跨来源性质验证，并实施同预算的能量揭示闭环。它是研发进展；当前goal继续active，不能因某张虚拟回放图有利就宣布新掺杂材料预测已经验证。

## 2. 真实数据、标签与成本

现有KROGER输入含873个电荷态、259类缺陷、19种元素。本轮选择单个掺杂原子的127类母类/396个状态，每类供应一个参考能量，形成269个非参考交叉能量标签，覆盖15种元素。Hf和Ge没有可供本任务验证的多电荷态标签。15个元素是化学覆盖，不是15个独立DFT来源。

学习对象是两电荷能量直线的交叉位置；它不一定是下包络上的稳定CTL。相同母类的原子参考项相消，但电子VBM参考必须保持一致。参考截距不作为学习特征。完整定义和推导见理论与验证约定.md。

孤立子集115标签，复合子集154标签。粗粒度特征存在结构冲突，所有样本保留，没有根据测试误差删样本。公开输入不是我们新做的HSE，私有派生能量表与第三方原始数值不进入公共代码发布。

对Si性质预测遮蔽29条能量，保留844条；对Sn遮蔽23条，保留850条。补充k条后，已知成本分别为844+k和850+k。还有已知结构目录、位点、电荷列表和参考态成本。节省不能表述成整个873能量库从零计算的大比例节省。

## 3. 能量学习确有信号，但其误差仍较大

外层留整个元素，内层留训练元素选参数，训练元素/母类等权。所有标准化来自训练集。平均绝对误差如下，单位eV：

"""
    text+=mdtable(energy_scores[energy_scores.stratum.eq("all")],
        ["method","elements","labels","element_parent_MAE_eV","worst_element_MAE_eV"])
    text+="""
ExtraTrees降低了平均交叉能量误差，但约1eV的误差不能视为精确微观能级预测。Ridge、RBF和树模型的全部结果与孤立/复合分层均在CSV中保留。

## 4. 更小的能量误差没有自动变成更好的真实性质预测

同一29条/5来源的留来源预测，全部29条都缺失可直接使用的氧分压，统一用未测量的10^-5 atm情景。各能量库只在训练来源上拟合共享有效冻结温度和背景施主；0/1/2个载流子锚点角色完全相同，偏移收缩由训练内部选择。下表为来源等权MAE，单位dex：

"""
    text+=mdtable(basic,list(basic.columns))
    text+="""
能量均值和Ridge在这里优于原始ExtraTrees/RBF。这直接说明：面向平均能量误差的训练目标，与最终载流子误差并不等价。完整873能量参照仍有更好的平均性质误差，但它使用更多信息。

“oracle_full”只表示掌握完整的已发表能量库，不表示真实材料的无误差真值。真实制备条件、有限温度能量、目录完整性和输运近似仍有误差。

![能量误差与性质误差](figures/energy_and_property.png)

## 5. 目标性质导向的信息选择已经实际跑通

RBF或树模型提供缺失能量均值与声明的协方差近似；Sandbox给出能量到log10(nH)的隐式导数。以预计降低最终载流子方差为准则选择下一条能量，再揭示并更新均值，重新求解物理状态。

与最大原始能量方差、32组随机选择对比。每条路径揭示0至8条已有但被遮蔽的公开能量；训练时不使用载流子测量标签。设计输入12个温度/剂量组合，评估输入另8个组合；都是指定的虚拟工况。统计近似、选点公式和局部成立条件见acquisition/选点公式与成立条件.md。

以下误差是相对于完整能量Sandbox的预测差，**不是相对于独立材料实验的精度**。随机min/max是32条路径范围，不是材料置信区间。

"""
    text+=mdtable(a,cols)
    text+="""
性质导向选点能较快恢复该模型的载流子响应。还有重要反例：RBF在Si上从1到4条标签时误差反弹；Sn的RBF在4条预算下不如最大能量方差。因此局部方差目标不保证实际误差逐步降低，也不是普遍最优定理。

被保留的未揭示能量仍可能有较大误差，但它们对当前性质工况不敏感，因而载流子预测可以接近完整模型。这是局部目标的压缩，不是完整能量库重建成功。

## 6. 拆开ML与物理选点的贡献

追加了所有5类学习器与两种简单独立方差的组合：原始截距等方差、交叉能量等方差。无需复杂协方差也可能表现很好；例如某些4标签预算下，Ridge加独立交叉能量方差优于原始树协方差。这些对照已进一步进入真实来源任务，而非只展示有利的原始组合。

这组追加实验是在看过第一组结果后制定的，属于显式的回顾性开发。不能重新称为盲测。完整明细在acquisition_ablation/summary.csv。

![选点策略与学习消融](figures/acquisition_replay_and_controls.png)

## 7. 把补回的能量重新用于真实文献性质预测

所有选点路径均不依赖真实载流子查询值。将其第2、第4条能量预算对应的数据库接回与第4节完全相同的物理拟合、内层收缩和来源留出程序。随机真实任务仅运行事先指定的seed0/1，不能冒充32组真实外部验证。

下面完整列出所有模型，单位dex。k是额外能量条目数，列中的anchor是额外载流子测量条数，二者不能混淆。

"""
    text+=mdtable(full,list(full.columns))
    text+="""
在同一能量库信息预算基础上，ET的性质导向补2条能量将0/1/2载流子锚点任务的误差分别降至0.259946/0.228817/0.219248dex，明显优于其原始缺失能量版本。然而，均值补全加相同类型选点达到0.259410/0.228677/0.212683dex；Ridge加独立交叉能量方差达到0.259106/0.228140/0.212431dex。当前真实性质证据没有确立复杂特征学习超越这些简单控制的额外收益，不能只展示ET对随机的有利比较。

补4条能量使部分方法更接近完整能量模型，却可能使其与实测载流子的误差回升。这提示完整物理模型及工艺情景本身仍有失配；较少能量下较低的实测误差不能直接解释为发现了更真实的微观能量。具体原因尚未唯一识别，也不能根据外层结果事后只保留k=2。

![补回能量后的真实载流子误差](figures/real_carrier_after_acquisition.png)

同样还报告所有锚点预算共享的18个查询点，避免把查询集合变化当成少样本改善：

"""
    text+=mdtable(common,list(common.columns))
    text+="""
第7节的得失必须按同一能量预算、同一载流子锚点预算、同一查询集比较；完整能量模型的信息成本更高。趋近该参照只能证明减少能量信息时保留了模型表现，不能解决参照本身的材料失配。

## 8. 理论、泄漏与旧结果保护核验

能量参考重建/平移代数误差小于2e-15eV；高斯逐次/分块条件与trace选点公式的差异在双精度舍入范围内；8个能量到载流子梯度的中心差分缩放误差小于6e-9。这些是实现核验，不是材料精度。

"""
    text+=f"\n机器检查：{audit['property_methods']}个性质方法，{audit['property_prediction_rows']}条逐样本预测；能量{audit['energy_prediction_rows']}条预测；没有检测到外层来源或能量元素泄漏。第一阶段847个冻结文件、85个原受保护文件均未改变。\n"
    text+="""
本轮oracle零锚点与第一阶段全局过程模型的前向一致；2锚点的收缩网格在本阶段明确改为4个比值，因此不能宣称所有适应结果逐点复现。所有数值/物理覆盖另列；本阶段物理标记为正规内点根、单电子Hall及检测下限，不额外使用历史m_star补偿筛选。

汇总时发现旧CSV推断器会把只含数字和字母e的哈希标识解析为数值，极端情况下变成inf。已从原始逐折文件恢复27行元数据，未重训、未更改任何模型预测；旧汇总与失败日志保留。后续使用csv_io_v2.py及all_property_predictions.csv，修复记录在metadata_reader_repairs.json。

## 9. 下一步应怎样继续优化，而不偏离预测目标

1. 将“平均能量准确”改为同时约束“最终性质准确”：用训练元素上的Sandbox响应建立性质敏感的学习损失，并在留整个元素、留来源上与相同核/特征的普通能量损失比较。
2. 检验局部导数选点对能量均值、协方差及新工况的敏感性。不能将目前已被反复查看的两个元素/有限工况当成普遍保证。
3. 补充有明确同一样品、实际剂量、测量温度和可读Hall数据的新来源。缺失工艺输入必须显式进入情景/分层变量；不能靠自由隐变量伪造真实工艺。
4. 迁移率与电导率需要完整剂量输入到性质的配对验证。旧52组给定实测nH的迁移率任务不等于完整掺杂预测，当前5个合格配对仍不足以宣称可靠材料性能排序。
5. 稀土筛选需要参考一致的能量/电荷态/结构支持；当前68个本地HSE输入的审计并未认证完整CTL，4f冻结芯限制也未消失。本轮没有新DFT/MD，没有新独立材料实验。

## 10. 重现入口

代码在scripts/trainable_sandbox_20261017/phase2_*.py；能量学习与选点数学模块在src/sandbox/energy_learning.py和goal_acquisition.py。各子目录有拟合前协议、模型、逐点预测与来源统计。运行环境使用ga2o3 Python，CPU线程固定1；本轮小数据模型在CPU运行，没有使用GPU0。用户授权的GPU1仍供后续需要的训练/计算使用。

文献与阅读深度见两个理论文件。能量/CTL机器学习、KROGER物理与目标导向实验设计均有既有工作，本轮不能把这些基础组件本身当作首创。可检验的研究贡献是它们在掺杂预测任务中的训练、物理耦合与信息成本闭环，贡献强度仍需后续独立数据和同预算验证支持。
"""
    (OUT/"第二阶段_能量学习与掺杂预测结果.md").write_text(text)

if __name__=="__main__":main()
