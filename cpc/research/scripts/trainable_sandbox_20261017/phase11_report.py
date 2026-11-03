"""Assemble the prediction/theory evidence without selecting favorable tasks."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.BASE/"phase11_parent_allocation"
REF=OUT/"solver_budget_sensitivity"
LABELS={"original_RT":"原目录RT","thermal_RT":"固定有效电离能RT","joint_baseline":"联合无库存学习",
    "parent_constant_joint":"常数库存偏置","parent_linear_joint":"线性库存学习","parent_rbf_joint":"RBF库存学习",
    "tf_rbf_joint":"同容量Tf学习","output_rbf_joint":"同容量联合输出修正",
    "parent_rbf_RT":"纯RT库存学习","output_rbf_RT":"纯RT输出修正",
    "dose_activation_RT":"非单调激活曲线","dose_line_RT":"剂量线",
    "direct_ridge_RT":"直接Ridge RT","direct_ridge_joint":"直接Ridge联合",
    "identity_n_equals_C":"n等于化学剂量","identity_plus_unbounded_offset":"剂量＋锚点常数校准"}
SHORT={"original_RT":"Original RT","thermal_RT":"Fixed E RT","joint_baseline":"Joint baseline",
    "parent_rbf_joint":"Learned inventory","output_rbf_joint":"Output field","tf_rbf_joint":"Tf field",
    "direct_ridge_joint":"Direct Ridge","identity_plus_unbounded_offset":"Dose + anchor offset",
    "output_rbf_RT":"Output field RT"}


def table(frame,scope,methods=None):
    f=frame[frame.scope.eq(scope)]
    pivot=f.pivot(index="method",columns="budget",values="MAE")
    order=list(LABELS) if methods is None else methods
    lines=["| 方法 | 0锚点 | 1锚点 | 2锚点 |","| --- | --- | --- | --- |"]
    for m in order:
        if m in pivot.index:
            lines.append("| "+LABELS[m]+" | "+" | ".join(f"{pivot.loc[m,b]:.6f}" for b in (0,1,2))+" |")
    return "\n".join(lines)


def figures(ps,ts,external):
    folder=OUT/"figures";folder.mkdir(exist_ok=True)
    chosen=("original_RT","thermal_RT","joint_baseline","parent_rbf_joint","tf_rbf_joint","output_rbf_joint","direct_ridge_joint")
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,frame,scope,title in zip(axes,(ps,ps,ts),("all_query","common18","common40_125"),
        ("RT: original queries","RT: fixed common 18","Temperature: common 40-125 K")):
        for m in chosen:
            d=frame[frame.method.eq(m)&frame.scope.eq(scope)].sort_values("budget")
            ax.plot(d.budget,d.MAE,"o-",label=SHORT[m],ms=4,lw=1)
        ax.set(xticks=[0,1,2],xlabel="Allowed anchors",ylabel="Source-equal MAE (dex)",title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("prediction_comparison."+ext),dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.7,4))
    for m in ("original_RT","thermal_RT","joint_baseline","parent_rbf_joint","output_rbf_RT","identity_plus_unbounded_offset"):
        d=external[external.method.eq(m)&external.scope.eq("all_query")&external.primary_growth_scenario].sort_values("budget")
        ax.plot(d.budget,d.MAE,"o-",label=SHORT[m],lw=1)
    ax.set(xticks=[0,1,2],xlabel="New-source specimen anchors",ylabel="MAE (dex)",
        title="PRB 2024: 3 paired SIMS/Hall conditions; unknown process inputs")
    ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("new_source_comparison."+ext),dpi=180)
    plt.close(fig)


def main():
    first=json.loads((OUT/"contract_audit.json").read_text());audit=json.loads((REF/"contract_audit.json").read_text())
    sensitivity=json.loads((REF/"sensitivity_audit.json").read_text())
    correction=json.loads((REF/"retained_stall_refinement/audit.json").read_text())
    assert first["passed"] and audit["passed"] and correction["passed"]
    external=OUT/"external_PRB2024";ea=json.loads((external/"contract_audit.json").read_text());assert ea["passed"]
    ps=read_result_csv(REF/"primary_summary.csv");ts=read_result_csv(REF/"temperature_summary.csv")
    es=read_result_csv(external/"summary.csv");figures(ps,ts,es)
    cost=json.loads((REF/"execution.json").read_text())
    diag=json.loads((REF/"state_diagnostic_audit.json").read_text());assert diag["passed"]
    def val(frame,method,budget,scope):
        return float(frame[frame.method.eq(method)&frame.budget.eq(budget)&frame.scope.eq(scope)].MAE.iloc[0])
    p0=val(ps,"parent_rbf_joint",0,"all_query");p2=val(ps,"parent_rbf_joint",2,"all_query")
    b0=val(ps,"joint_baseline",0,"all_query");b2=val(ps,"joint_baseline",2,"all_query")
    parts=["# ML＋Sandbox：共享库存学习、掺杂预测与新来源检验\n",
        "系统 goal 保持 active。本阶段继续研发从实际掺杂量与可得工艺输入预测材料性质的 ML＋Sandbox；载流子是首个主终点。理论与可识别性用于改进预测，不替代预测方法。GPU 1 授权持续有效，本轮为 CPU 求根与小数据训练，没有新增 DFT、MD 或物理实验。\n",
        "## 1. 本轮实际做了什么\n",
        "新增可训练的内部母类库存分配函数，完整重求高温库存和冻结后的电荷平衡；不是只在性质输出上加一个回归器。60 项目录/规范/守恒/IFT 检查、115 项学习接口检查以及86个零偏置实际状态回放通过。它们是数值实现证据，不能自动证明材料预测准确。\n",
        f"14 类方法的540个缓存拟合、7560次内层候选评分、1218行室温预测及6258行变温预测均独立重建。初版新增660个优化起点，其中100个未收敛，40个拟合的最佳有限解未收敛，1个这样的拟合被外层选择；9次训练中间状态失败全部保留，最终查询没有缺失。\n",
        f"对全部{cost['affected_records']}个曾有失败起点的拟合，两个原起点统一把评估上限60提高至180，新增{cost['new_starts']}个起点。仍有{cost['nonconverged_starts']}个起点、{cost['selected_nonconverged']}个最佳有限解未收敛；复合结果中外层选中未收敛拟合数为{audit['selected_outer_nonconverged_fits']}。因此下表是明确优化协议下的结果，不是全局最优证明。\n",
        "## 2. 原室温主任务：所有模型与预算\n",
        "原29片Si/Sn、5个来源的任务不变。外层排除整个来源；内层在训练来源选模型与锚点收缩。0/1/2是每个新来源允许的样品锚点，分别有29/23/18个查询。指标先在来源内平均，再来源等权，单位dex。下面使用180上限敏感性后的复合V2；V1逐行结果另存，未覆盖。\n",
        table(ps,"all_query"),"\n",
        f"RBF库存学习的0锚点误差{p0:.6f}、2锚点误差{p2:.6f}；相同新联合损失下无库存学习为{b0:.6f}、{b2:.6f}。不能将模型已实现或局部数值响应写成普遍预测增益。原目录RT及全部同容量输出/Tf对照均保留，不能只同直接Ridge比较。\n",
        "## 3. 相同查询集与变温补充\n",
        "为避免预算变化同时改变难例组成，下面在所有预算共有的18片室温查询上比较。它与各预算各自查询是两个同时报告的终点。\n",
        table(ps,"common18"),"\n",
        "变温仍是原有7片Si、2个来源、142个曲线读数，没有新增独立样品。本轮将室温任务选出的配置和锚点收缩直接迁移到温度任务，未实施阶段10专用温度选参/内部适应。不同协议的温度分数不能解释为库存模块的净增量。共同40–125K的80个查询结果如下；所有温区与90个读取包络查询另存CSV。\n",
        table(ts,"common40_125"),"\n",
        "此前阶段10的双内部适应在共同温区两锚点为0.206518dex，同共享输出偏移0.218581，普通两点线0.230520。这里保留这些强对照作为上下文，但不把不同选参协议混成单次结构消融。\n",
        "![室温和补充温度任务，完整来源隔离与固定查询集](figures/prediction_comparison.png)\n",
        "## 4. 收敛复核如何影响结论\n",
        f"初版540个拟合的文件哈希全部保持。统一提高上限改变了{sensitivity['changed_outer_config_choices']}个外层配置选择和{sensitivity['changed_outer_anchor_lambda_choices']}个锚点收缩选择；这些变动属于数值优化敏感性，不是新的物理关系。目标函数比较逐项保留，未按测试误差选择重算对象。数据契约通过与所有优化器收敛是两项不同结论。\n",
        f"统一上限检查的原始passed=false也保留：一个三来源内部拟合的训练目标反而高4.79×10的负8次方；该次第一起点触发电子密度下溢。随后从保存的原最佳参数出发，用同目标/边界的L-BFGS-B作单例修正，目标{correction['original_objective']:.10f}变为{correction['corrected_objective']:.10f}，实际状态梯度差分检查通过。修正后配置、收缩和全部外层预测不变；三来源拟合不参与新来源的四来源部署CV，因此不改720行新来源结果。它不是全局最优证明，原失败未改成success。\n",
        f"该单例L-BFGS-B按目标相对变化停止，投影梯度最大值仍为{correction['optimizer']['projected_gradient_max']:.3g}；未将其说成达到10的负8次方的一阶最优。外层拟合的收敛数使用预设优化器停止标志，也不等价于全局最优或物理参数唯一。\n",
        "初版实际完整预测调用1386718次、高温求解309195次、高温缓存命中1077523次、热态残差评估6244075次；这些量分别记录，不称为相同运行时间。复合审计中的成本包含继承记录；额外收敛复核的实际成本单列于execution.json。\n",
        "## 5. ML是否真正进入了物理内部\n",
        f"对四类库存模型、三种预算共{diag['rows']}个源外室温状态，分别求已训练模型、保持全局参数但令库存学习为零的反事实、独立训练的无库存基线。没有删除拒绝行；本次拒绝数为{diag['rejected_rows']}。库存贡献与全局参数重拟合贡献相加，严格回到总预测变化。\n",
        "常数库存偏置与非零函数库存偏置均能改变母类分配，但有效冻结温度、背景与电离关系会补偿部分载流子响应。全部原始母类数量、电荷、孤立掺杂原子占比、多X复合体占比和带H复合体占比是模型隐状态，不是实测微观真值。该诊断回答内部模块是否被调用，不能代替留出精度比较。\n",
        "## 6. 额外来源：化学掺入量到Hall浓度\n",
        "Li等PRB110,174106(2024)的Table I报告三组同条件配对的SIMS Sn与室温Hall数值：\n",
        r"| 条件 | 化学Sn，\(\mathrm{cm}^{-3}\) | Hall载流子，\(\mathrm{cm}^{-3}\) |"+"\n| --- | --- | --- |\n"+
        r"| 1 | \(1.2\times10^{18}\) | \(4.3\times10^{17}\) |"+"\n"+
        r"| 2 | \(1.9\times10^{18}\) | \(6.7\times10^{17}\) |"+"\n"+
        r"| 3 | \(3.4\times10^{18}\) | \(1.2\times10^{18}\) |"+"\n",
        "Table I的HTML和PDF两路文字提取逐行一致；它们来自同一张原表，不是两次独立测量。网页工具未传回可查看的PDF截图，本地PDF请求为403，未将像素视觉核对记为完成。原文数值没有单独误差条，不能把印刷舍入精度当作仪器噪声或置信区间。\n",
        "只把化学浓度作为剂量输入。样品名称由载流子量级命名，绝不进入特征。共享训练和配置选择只用原五来源；新来源的最低剂量、再中间剂量依次作为允许的锚点，查询保留最高剂量共同点。未知生长特征使用原训练来源等权加权分位数，保留全部五个情景；下面是中位数主情景。氧压仍为条件情景，40% CO2不能当作40% O2。\n",
        table(es[es.primary_growth_scenario],"all_query"),"\n",
        "三个点的n/C约为0.35，近似固定激活比例。因此一条锚点校准的剂量比例本身就是非常强的对照，不能删掉后宣称复杂模型获胜。零锚点热模型如有优势，应归于其对应电离统计和训练，不应归于被选为零的库存函数。只有一个新增来源，缺少生长温度、氧分压和背景杂质，属于条件迁移；不能据此宣称普遍新材料准确度。\n",
        f"720行新来源预测由独立脚本全部回放，最大误差{ea['maximum_replay_error']:.3g}dex，11个全五来源部署拟合的最佳起点均收敛。初版仅在诊断字典合并时发生重复Tf_K键错误；V2修复元数据合并并复用未变的数值拟合，原失败日志和代码保留。\n",
        "![额外来源的全部预算与简单激活比例对照](figures/new_source_comparison.png)\n",
        "## 7. 新来源与收敛敏感性的衔接\n",
        ("复核后，原五来源全局部署的42项配置/锚点选择全部不变，所以上述已收敛部署模型和720行预测继续适用。\n" if not sensitivity['changed_external_deployment_choices'] else
         "复核改变了部分原五来源的部署选择。上述表为保留的V1部署，不冒充复核后的最终选择；变动详情与后续部署另存external_selection_check。\n"),
        "## 8. 交付、限制与继续方向\n",
        "理论全文如下；输入和代码哈希、两版预测、来源分组、全部失败、优化起点、成本、候选选择、根有效性和库存反事实均在同目录。主脚本phase11_prediction.py，复核phase11_solver_budget.py，独立审计phase11_prediction_audit.py与phase11_refinement_audit.py；新来源使用phase11_external_source_v2.py与phase11_external_audit.py。旧预测接口和旧冻结结果未覆盖。\n",
        "下一项具体改进是训练来源条件下的内部库存少样本适应，直接比较内部a_f、输出偏移和有效温度适应。该方法尚未训练；不能把下一项理论方案列为既有优势。原室温主任务、完整迁移率/电导率验证及有依据的新元素排序仍未完成，goal保持active。\n",
        "## 9. 数学、物理条件与下一方案的完整论证\n",
        "## 1. 学习对象及其物理位置\n"+
        (ROOT/"docs/ML_SANDBOX_PARENT_ALLOCATION_THEORY_20261017.md").read_text().split("## 1. 学习对象及其物理位置",1)[1]]
    target=OUT/"ML与Sandbox_共享库存学习与掺杂预测检验.md"
    target.write_text("\n".join(parts))
    energy.write(OUT/"report_claims.json",dict(created_utc=energy.now(),
        source_primary_summary_sha256=energy.sha(REF/"primary_summary.csv"),
        source_temperature_summary_sha256=energy.sha(REF/"temperature_summary.csv"),
        parent_RBF_0anchor=p0,parent_RBF_2anchor=p2,matched_baseline_0anchor=b0,matched_baseline_2anchor=b2,
        external_scope="V1 original-source-only selection; inspect sensitivity_audit for selection consequences",
        goal_complete=False,new_DFT_MD=0,new_physical_experiments=0))
    print(target)


if __name__=="__main__":main()
