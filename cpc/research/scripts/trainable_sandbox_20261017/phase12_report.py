"""Assemble all phase12 outcomes; no retraining or metric-based filtering."""
from pathlib import Path
import json,sys,re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.BASE/"phase12_stock_adaptation"
FINAL=OUT/"physical_scope_v2"
TRANS=OUT/"full_property_chain"
LABELS={
    "original_offset":"原目录物理＋输出适应", "original_stock":"原目录物理＋库存适应",
    "original_Tf":"原目录物理＋Tf适应", "constant_offset":"常数库存先验＋输出适应",
    "constant_stock":"常数库存先验＋库存适应", "shared_RT_offset":"共享RT库存函数＋输出适应",
    "shared_RT_stock":"共享RT库存函数＋库存适应", "matched_output_offset":"同容量输出函数＋输出适应",
    "matched_Tf_offset":"同容量Tf函数＋输出适应", "joint_offset":"联合电离模型＋输出适应",
    "joint_stock":"联合电离模型＋库存适应", "shared_joint_offset":"联合库存函数＋输出适应",
    "shared_joint_stock":"联合库存函数＋库存适应", "direct_RT_offset":"直接Ridge＋输出适应",
    "dose_offset":"剂量直线＋输出适应", "dose_affine":"剂量直线＋截距/斜率适应",
    "activation_prior_offset":"训练源中位激活比＋输出适应", "identity_offset":"n=C＋嵌套收缩偏移",
    "identity_unbounded":"n=C＋无收缩偏移", "identity_H_unbounded":"n=C+已报H＋无收缩偏移"}
FAMILIES=["10.1002/aelm.202300679","10.1007/s10853-015-9693-6","10.1007/s40843-022-2167-x",
    "10.1063/5.0027870","10.1063/5.0101132"]
SHORT_SOURCE=dict(zip(FAMILIES,("TEOS","Baldini Sn","Xiang Si","Mauze Sn","McCandless Si")))


def table(frame,scope):
    p=frame[frame.scope.eq(scope)].pivot(index="model",columns="budget",values="MAE")
    lines=["| 方法 | 0锚点 | 1锚点 | 2锚点 |","| --- | --- | --- | --- |"]
    for method,label in LABELS.items():
        lines.append("| "+label+" | "+" | ".join(f"{p.loc[method,b]:.6f}" for b in (0,1,2))+" |")
    return "\n".join(lines)


def make_figures(summary,source):
    folder=OUT/"figures";folder.mkdir(exist_ok=True)
    choices={"original_offset":"Original + output", "constant_offset":"Constant stock + output",
        "constant_stock":"Constant stock + internal", "shared_RT_stock":"RBF stock + internal",
        "joint_stock":"Ionization model + internal", "dose_affine":"Dose + affine adaptation"}
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.9))
    for ax,scope,title in zip(axs,("all_query","common18"),("Original 29/23/18 queries","Fixed common 18 queries")):
        for model,label in choices.items():
            f=summary[summary.model.eq(model)&summary.scope.eq(scope)].sort_values("budget")
            ax.plot(f.budget,f.MAE,"o-",label=label,lw=1,ms=4)
        ax.set(xticks=[0,1,2],xlabel="Allowed specimen anchors per source",ylabel="Source-equal MAE (dex)",title=title)
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=6.5);fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("primary_prediction."+ext),dpi=190)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.4,3.9));x=np.arange(5);w=.24
    for k,(model,label) in enumerate((("original_offset","Original + output"),
        ("constant_offset","Constant stock + output"),("constant_stock","Constant stock + internal"))):
        s=source[source.model.eq(model)&source.budget.eq(1)&source.scope.eq("all_query")].set_index("family")
        ax.bar(x+(k-1)*w,[s.loc[f,"MAE"] for f in FAMILIES],w,label=label)
    ax.set(xticks=x,xticklabels=[SHORT_SOURCE[f] for f in FAMILIES],ylabel="MAE (dex)",title="One anchor: source-level gain is concentrated in TEOS")
    ax.legend(fontsize=7);ax.grid(axis="y",alpha=.2);fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("one_anchor_by_source."+ext),dpi=190)
    plt.close(fig)
    bounds=read_result_csv(TRANS/"head_capacity_bounds.csv")
    bounds=bounds[bounds.budget.eq(0)].drop_duplicates("sample_id")
    full=read_result_csv(TRANS/"predictions.csv")
    f=full[full.model.eq("original_offset")&full.budget.eq(0)&full["head"].eq("direct_Hall_argument")].set_index("sample_id")
    fig,ax=plt.subplots(figsize=(9.3,3.8))
    for i,r in enumerate(bounds.itertuples()):
        ax.plot([i,i],[r.log_mu_min,r.log_mu_max],lw=7,color=".75",solid_capstyle="round")
        ax.scatter(i,r.observed_log_mu,c="#b33335",marker="x",s=52,label="Published mobility" if i==0 else None,zorder=3)
        ax.scatter(i,f.loc[r.sample_id,"predicted_log_mu"],c="#23609d",s=25,label="Actual full-chain forecast" if i==0 else None,zorder=3)
    ax.plot([],[],lw=7,color=".75",label="Entire fixed-head output range")
    ax.set(xticks=np.arange(len(bounds)),xticklabels=[s.split("::")[-1] for s in bounds.sample_id],
        ylabel=r"$\log_{10}\mu_H$ ($\mu_H$ in cm$^2$ V$^{-1}$ s$^{-1}$)",title="CT capacity limitation on the same 5 paired specimens")
    ax.legend(fontsize=7,loc="lower right");ax.grid(axis="y",alpha=.2);fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("mobility_head_capacity."+ext),dpi=190)
    plt.close(fig)


def source_audit():
    teos=Path("/data/lawrence/Physics/Ga2O3-Net/extra-paper/v56_markdown/ec3fc6d0e0ba6300.md")
    baldini=Path("/data/lawrence/Physics/Ga2O3-Net/extra-paper/doping_extracted/doping/Sn/baldini2015.pdf")
    records=[dict(doi="10.1002/aelm.202300679",local_path=str(teos),sha256=energy.sha(teos),
        actual_read_depth="local full-text extraction: Table1,Hall/SIMS discussion and growth/microstructure sections;not raw instrument records",
        verified="Table1 and body both print mobility.144/.242cm2/Vs forTEOS2/3;do not autochange to144/242",
        reuse="2already paired mobility samples;TEOS1conflicting source mobility excluded,not newdata",
        physical_scope="100nm/sapphire and microstructural discussion are candidate evidence,not unique transport-mechanism proof"),
        dict(doi="10.1007/s10853-015-9693-6",local_path=str(baldini),sha256=energy.sha(baldini),
        actual_read_depth="7page PDF text;Fig5/caption and mobility/scattering paragraphs;rendered pages cached,not recorded as visually inspected",
        verified="8SIMS/Hall points inFig5;specimen mobilities not shown;maximum41atn1.1e18 is one conditional text pair already in52",
        reuse="no defensible chemical-to-mobility mapping for8Fig5points;no nearest joining;no newchemicalpairedmu"),
        dict(id="arxiv:2311.00821",url="https://arxiv.org/abs/2311.00821",
        actual_read_depth="official arXiv abstract/title/author metadata verified;sourceID check,not newcurve extraction",
        verified="Gann implantation/anneal paper;not TEOS paper;citation groups are not automatically independent laboratories")]
    energy.write(OUT/"literature_input_audit.json",dict(created_utc=energy.now(),records=records,
        new_chemical_paired_mobility_labels=0,carbon_energies_in_catalogue=False,
        limits="source text is published/processed evidence,not independent reproduction;no rawHall protocols or additionalDFT"))


def main():
    audit=json.loads((FINAL/"contract_audit.json").read_text());assert audit["passed"]
    runtime=json.loads((OUT/"research_runtime/verification.json").read_text());assert runtime["passed"]
    assert json.loads((OUT/"research_runtime/transport_replay_audit.json").read_text())["passed"]
    summary=read_result_csv(FINAL/"summary.csv");source=read_result_csv(FINAL/"by_source.csv")
    transport=read_result_csv(TRANS/"summary.csv");capacity=read_result_csv(TRANS/"head_capacity_summary.csv")
    make_figures(summary,source);source_audit()
    bylines=["| 1锚点的来源 | 原物理＋输出 | 常数库存＋输出 | 常数库存＋内部 |","| --- | --- | --- | --- |"]
    for family in FAMILIES:
        r=source[source.family.eq(family)&source.budget.eq(1)&source.scope.eq("all_query")].set_index("model")
        bylines.append("| "+SHORT_SOURCE[family]+" | "+" | ".join(f"{r.loc[m,'MAE']:.6f}" for m in ("original_offset","constant_offset","constant_stock"))+" |")
    chains=["| 方法与锚点数 | 载流子MAE | 迁移率MAE | 派生电导率MAE |","| --- | --- | --- | --- |"]
    for m in ("original_offset","constant_stock","shared_RT_stock"):
        for b in (0,1,2):
            r=transport[transport.model.eq(m)&transport["head"].eq("direct_Hall_argument")&transport.budget.eq(b)].iloc[0]
            chains.append("| "+LABELS[m]+f"，{b} | "+" | ".join(f"{r[k]:.6f}" for k in ("nH_MAE","mu_MAE","sigma_MAE"))+" |")
    caps=["| 锚点数 | 固定头误差下界 | 原物理实际误差 | 代入真n的诊断误差 |","| --- | --- | --- | --- |"]
    for r in capacity.itertuples():caps.append(f"| {int(r.budget)} | {r.lower_bound:.6f} | {r.actual_error:.6f} | {r.same_query_perfect_n_error:.6f} |")
    parts=["# ML＋Sandbox：内部库存适应、掺杂预测与输运检验\n",
        "系统goal保持active。本轮主目标仍是从实际掺杂量与可得工艺输入，经训练的共享内部关系和守恒Sandbox预测掺杂性质；载流子浓度是首个验收终点，迁移率、电导率和有依据的新掺杂排序继续保留。理论、可识别性和诊断服务于改进预测。\n",
        "## 1. 本轮进展与关键结论\n",
        "新增四种原RT物理层上的匹配学习模型，共150个基础拟合；另实现相对已保存共享模型的一维来源库存适应，并与相同锚点的输出适应、Tf适应、直接ML和剂量比例对照。20种方法、所有预算与全部失败均保存。独立进程不读取性质表，复现保存模型的实际物理预测，验证训练关系能够在预测时复用。\n",
        "常数库存先验＋内部来源更新在1锚点主任务得到0.229552dex，原物理＋输出适应为0.251591；但收益主要来自一个TEOS来源，0/2锚点不保持。RBF共享库存函数仍未显示稳定增量。不能据此宣布ML＋Sandbox在全部预算、来源或性质上获胜。\n",
        "完整迁移率链的误差仍约1.2—1.35dex。进一步核验表明当前固定CT头自身的输出下限已造成约1.23dex误差下界；只改进载流子预测无法消除此限制。这重申并定量化了早期CT表达范围审计，后续要同时改进电子统计与输运闭合。\n",
        "## 2. 数据、预测输入与严格隔离\n",
        "主任务不变：29片Si/Sn、5个DOI来源；整来源外层留出，每个新来源允许0/1/2个已指定样品锚点，查询总数29/23/18。同时报告共同18片查询，防止锚点预算改变难例组成。MAE先在来源内平均，再来源等权；dex是log10误差，例如0.30dex约对应2倍乘性偏差。五个DOI不能自动当作五个独立实验室。\n",
        "基础模型的标准化、全局参数和共享权重只用相应训练来源；正则化强度和来源适应收缩由训练来源内留一选择。外层查询标签不进入训练或适应，但整套队列已经反复查看，属于回顾性研发；预计算保存的未选候选不能称为盲测或预注册。\n",
        "输入为实际化学掺入浓度、掺杂元素、可得生长温度和已报告H。29条记录均没有已报告氧分压，本轮使用显式1e-5atm情景；部分温度是来源级代理。KROGER当前19元素目录不含C；Xiang样品D的高C还是非均匀表面富集，不编造C能量或当作均匀体相供应。输入不完整限制材料外推。7片既有Si的变温曲线没有增加独立样品，本轮没有新增温度专用验证。\n",
        "## 3. 内部适应如何进入真实物理预测\n",
        "共享关系给出的无量纲库存偏置为b，乘kBTf后才是eV能量量纲；它经非规范母类方向改变高温库存，再冻结母类数量并重求测量温度电荷平衡。全部电荷态同幅偏移在固定库存下会抵消；本轮的学习作用发生在高温库存分配，不将其说成冷态电荷概率的新直接能量信息。\n",
        r"\["+"\n"+r"v=\phi(x)^T\psi,\qquad b(z)=24\tanh\!\left(\frac{v+\ln(10)z}{24}\right),\quad w=k_BT_f b(z)."+"\n"+r"\]"+"\n",
        "x为不含查询性质的样品输入；phi为训练来源标准化的7维RBF/元素/工艺特征，psi为已保存共享权重，常数模型只用1维。z为新来源相对冻结共享关系的一个增量。z=0精确恢复原模型；非零z通过完整守恒求解传播。该量是有效条件校准，不是真实工艺参数或新HSE标签。\n",
        r"\["+"\n"+r"J(z)=\sum_{j\in\mathcal A}\big[\log_{10}n_H(x_j;z)-y_j\big]^2+\lambda z^2."+"\n"+r"\]"+"\n",
        "A为允许的锚点集合，yj是锚点实测log10 Hall浓度；查询标签不在此目标。lambda网格为0、0.001、0.01、0.1、1、10和无穷大；0锚点或无穷大收缩给出精确零更新。内部z在事先声明的[-6,6]数值区间搜索；输出偏移及两坐标经验适应不施加该截断，保留更强的简单对照。\n",
        "求解先扫描25个含0节点，对每个网格局部极小区间精化，保留无效物理状态和多极小值；不是连续全局最优证明。局部I为锚点响应导数平方和，I/(I+lambda)只是局部数据/收缩分辨比例，不能当真实微观参数置信区间。共享截距固定以后再做来源更新，避免同时自由拟合两个可互相平移的截距。完整推导附于第11节。\n",
        "## 4. 原室温主任务：全部20种方法\n",table(summary,"all_query"),"\n",
        "新增无中性态扩展的库存/输出/Tf函数使用同一容量；其无穷大正则化严格退回原目录RT，避免把不同电子统计层的收益误归给学习连接。联合电离模型继承以前室温＋温度训练，其比较同时包含统计层与训练目标变化；相应无学习和共享关系对照均保留。\n",
        "## 5. 固定查询集与逐来源核验\n",table(summary,"common18"),"\n",
        "![仅示意若干预先列出的结构对照；全部20模型见两张完整表](figures/primary_prediction.png)\n",
        "共同18片上，常数库存内部更新的1锚点0.228735优于原物理输出0.250950，但2锚点升至0.274557。这说明当前收益依赖预算与来源，不能以最好单格分数代表全方法。逐来源1锚点结果：\n","\n".join(bylines),"\n",
        "TEOS的1锚点误差从原物理输出0.176035降至0.059525；相同常数库存先验上的输出适应为0.307122。其余四来源相对常数输出一个改善、两个更差、一个持平，差异均约0.01dex以内；尚无多来源普遍获益证据。\n",
        "![内部更新的主要收益集中在TEOS；Mauze难例依然保留](figures/one_anchor_by_source.png)\n",
        "TEOS两种预算使用同一基础拟合。1锚点训练内选lambda=0.1，z=1.730073，局部分辨比例0.398962；2锚点选lambda=1，z=0.371074，比例0.128028，查询误差升至0.263910。内层来源选择了更强收缩，是这一差异的具体原因之一；不能把它归纳为更多数据在物理上必然有害，也不能事后按TEOS查询分数挑lambda。\n",
        "## 6. 数值门槛、失败与修正记录\n",
        "无扩展学习109项检查与435次原RT零函数回放通过，最大梯度相对检查约9.21e-9；来源适应119项检查与870次旧模型零更新回放通过，最大导数检查约2.39e-8。它们验证规范方向、隐式梯度、边界和接口，不证明材料精度。\n",
        f"150个新增基础拟合共300个起点，其中{audit['new_base_nonconverged_starts']}个未按预设标志收敛，{audit['new_base_best_nonconverged']}个记录的最佳有限解未收敛；最终外层选中者为{audit['new_base_outer_selected_nonconverged']}。训练中间物理状态失败{audit['new_base_state_failures']}次，全部保留。每起点评估上限60；本轮没有再按测试表现追加求解预算，也未证明未选候选全局最优。\n",
        "V1的零更新缓存遗漏了单电子Hall适用域标志：物理非零更新入口已拒绝n小于p，但缓存零点漏查。V2统一更正3个episode、25个预测条目的适用域拒绝，未重训；所有外层配置、lambda和预测都未改变。V1代码、结果和失败保留，V2不是原地覆盖。\n",
        f"独立审计重建900个来源episode、78054个剖面候选目标、7494个闭式偏移及12420次内层候选评分；核对78300条谱系与1740行外层保存结果。查询无缺失；300个仿射适应系数另行核对。目标重建最大差{audit['new_base_objective_reconstruction_error']:.3g}。相同候选上限42不等于相同运行时间，基础拟合、适应网格和求根成本单列。\n",
        "## 7. 完整掺杂性质链：载流子、迁移率与派生电导率\n",
        "原29片只有5条明确化学浓度—Hall—迁移率配对，来自TEOS与McCandless两个DOI；0/1/2锚点下有5/4/2个查询。没有为Baldini的8个化学/Hall图点按位置或最近邻补造迁移率。单独的52对/11引用组条件迁移率用于训练排除目标DOI的经验头，不增加5条完整链验证样品。\n",
        "主输运头直接把预测Hall浓度作为CT经验曲线参数，与本轮单位Hall因子载流子约定保持清楚；另一头保留旧启发式Hall反变换，作为独立特征变换对照，不能冒充同一物理状态。两头52/11条件MAE分别0.371934和0.366609dex；不同队列不能直接相减并称为载流子误差贡献。全部20方法、两头共440行完整预测另存CSV，下面列三组核心对照：\n","\n".join(chains),"\n",
        r"\["+"\n"+r"\sigma=e\,n_H\mu_H,\qquad \Delta\log_{10}\sigma=\Delta\log_{10}n_H+\Delta\log_{10}\mu_H."+"\n"+r"\]"+"\n",
        "e为正元电荷，nH单位cm的负3次方、muH为cm平方每伏秒，sigma为S/cm。本轮sigma由同一对Hall量在预测端和观测端派生，没有新增独立电导率测量。两个对数误差可以抵消，较小sigma误差不能单独证明两个组成预测均正确。\n",
        "## 8. 输运头容量限制：为何不能只优化载流子\n",
        r"\["+"\n"+r"\mu_H(n)=10^{\Delta_f}\left[82+\frac{205-82}{1+(n/(1.18\times10^{18}))^{1.25}}\right]."+"\n"+r"\]"+"\n",
        "n为这一经验头的Hall参数；82和205的单位均为cm平方每伏秒，Deltaf为排除目标DOI训练的对数中位偏移。对所有正n，该头的logmu必在L=log82+Deltaf和U=log205+Deltaf之间。因此对于观测y，绝对log误差至少为max(L−y,0,y−U)。下界只针对当前冻结头，允许目标源mu锚点或换输运结构会改变它。\n","\n".join(caps),"\n",
        "TEOS当前训练头的最低输出约55cm平方每伏秒，而公开Table1及正文均确认两个标签为0.144和0.242；不是应自动乘1000的单位修正。0锚点整体固定头下界1.234605dex，实际1.332676，约93%的该误差已受输出范围限制。表中“真n”只作事后诊断，未进入正式方法。这个结果是先前表达范围问题的完整链量化，不是新散射机制发现。\n",
        "![灰线为任意正载流子输入可达到的全部迁移率范围；部分观测远在范围之外](figures/mobility_head_capacity.png)\n",
        "TEOS原文提供100nm薄层、蓝宝石衬底及晶粒/孪晶/堆垛缺陷的讨论，只构成额外输运通道的候选依据。Baldini正文说明逐样品迁移率未画出，仅有最大值41对应n=1.1e18的既有条件配对；无新的化学配对mu可加入。不同DOI也不是全部独立实验室或批次。\n",
        "## 9. 保存模型与独立预测入口\n",
        "SourceStockModel加载显式材料参数、氧压情景、能量库及保存的共享权重；预测函数本身不接收测量标签。ScalarProfile只接收明确允许的锚点。九组代表模型、5外层来源、3预算导出135个研究模型包；不是使用全部五来源重训的最终部署模型。\n",
        f"新进程开启文件访问审计，禁止读取性质表、拟合记录和episode文件。运行任务只含物理输入和授权锚点logn，没有查询目标。{runtime['prediction_rows']}行全点载流子复现最大差{runtime['maximum_log_nH_error']:.3g}dex，99行明确配对查询的mu/sigma复现最大差分别2.43e-11和7.42e-11dex。能运行和复现是实现证据，不是新增独立材料精度。\n",
        "接口分别返回经过经验输出修正的值、原物理状态值以及same_physical_state_for_output标志。非零输出偏移不会重求Nabs/Nq平方；必须保留这一差别，不能把修正n与未变库存混称新的电荷中性物理状态。入口和执行命令见research_runtime/README.md。\n",
        "## 10. 按预测目标继续推进的下一项理论\n",
        "下一轮同时保留原29/5主终点与完整性质链，先处理永久局域中性态不能自然恢复高浓度全电离的结构限制，再审计可用电荷预算的输运闭合。新候选是给定输入下固定的局域/延展施主子群比例ell(C)，从同一个自由能推导库存、电荷和梯度。它能把ell=0的原目录模型和ell=1的永久中性态模型作为精确端点。\n",
        "候选目前仅有解析推导，尚未实现、训练或得到材料增益；不能称为已经证明的Mott/局域化机制。更改自由电子统计后必须重验原nH任务；更改Hall因子后必须同步修正nH与muH读出。输运将先核对现有Brooks–Herring/POP函数的单位、统计、适用域及Hall约定，再与化学剂量/电荷中性代理和预测n经验头公平比较。详细候选条件与推导见ML_SANDBOX_TWO_CLOSURES_NEXT_20261017.md。\n",
        "这是一项继续改进可训练ML＋Sandbox预测方法的明确路线，不以机制解释结束研发。本阶段没有新增DFT、MD、真实样品或仪器实验，也未使用GPU训练；GPU1授权持续有效。goal仍为active，主任务稳定增量、实用输运准确度及新掺杂排序均未完成。\n",
        "## 11. 训练前写定的完整理论与计算约定\n",
        "以下重录本阶段训练前保存的理论和协议说明；该原文件代码哈希已绑定，未根据外层结果改写。有关温度任务是条件计划，本轮只执行室温与完整性质链。\n",
        re.sub(r"(?m)^## ([1-6])\.",r"### 11.\1","## 1."+(ROOT/"docs/ML_SANDBOX_SOURCE_STOCK_ADAPTATION_20261017.md").read_text().split("## 1.",1)[1])]
    report=OUT/"ML与Sandbox_内部适应与完整性质链检验.md";report.write_text("\n".join(parts))
    claims=dict(created_utc=energy.now(),summary_sha256=energy.sha(FINAL/"summary.csv"),
        models=20,new_base_fits=150,primary_specimens=29,primary_sources=5,common_queries=18,
        one_anchor_internal_constant_MAE=.22955233180682472,
        one_anchor_original_offset_MAE=.2515914701173037,
        one_anchor_gain_scope="mainlyTEOS;not maintained0/2anchors;no stableRBFincrement",
        full_chain_mobility_specimens=5,full_chain_mobility_DOIs=2,
        fixed_mobility_head_0anchor_lowerbound=1.23460505680328,
        independent_sigma_labels=0,goal_complete=False,next_model_implemented=False)
    energy.write(OUT/"report_claims.json",claims)
    print(report)


if __name__=="__main__":main()
