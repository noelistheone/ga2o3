"""Complete prediction results, attribution and numerical limits for phase10."""
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

OUT=energy.BASE/"phase10_internal_adaptation"
LABELS={"original_RT_offset":"原目录RT＋偏移","original_RT_Tf":"原目录RT＋Tf适应",
    "fixed_joint_offset":"固定能级＋偏移","fixed_joint_Tf_E":"固定能级＋双内部适应",
    "shared_joint_offset":"共享E(C)＋偏移","shared_joint_Tf":"共享E(C)＋Tf适应",
    "shared_joint_E":"共享E(C)＋E适应","shared_joint_Tf_E":"共享E(C)＋双内部适应",
    "direct_joint_offset":"直接Ridge＋偏移","direct_joint_affine":"直接Ridge＋截距斜率",
    "dose_line_offset":"剂量线＋偏移"}
ORDER=list(LABELS)
SHORT={"original_RT_offset":"Original RT + offset","fixed_joint_Tf_E":"Fixed E + internal",
    "shared_joint_offset":"Learned E(C) + offset","shared_joint_E":"Learned E(C) + E",
    "shared_joint_Tf_E":"Learned E(C) + internal","direct_joint_affine":"Direct Ridge + affine"}


def table(summary,task,scope):
    q=summary[summary.task.eq(task)&summary.scope.eq(scope)].set_index(["method","budget"])
    rows=["|模型|0锚点|1锚点|2锚点|","|---|---:|---:|---:|"]
    for m in ORDER:rows.append("|"+LABELS[m]+"|"+"|".join(f"{q.loc[(m,b),'MAE']:.6f}" for b in (0,1,2))+"|")
    return "\n".join(rows)


def summarize_resolved(pred):
    roles=read_result_csv(energy.BASE/"episodes_v1.csv")
    common=set(roles[roles.budget.eq(2)&roles.role.eq("query")].sample_id)
    rows=[]
    for (task,method,budget,family),g in pred[pred.role.eq("query")].groupby(["task","method","budget","family"]):
        scopes=("all_query","common18") if task=="primary" else ("all_query","common40_125")
        for scope in scopes:
            h=g if scope=="all_query" else (g[g.common40_125] if task=="temperature" else g[g.sample_id.isin(common)])
            if not len(h):continue
            h=h.copy();h["error"]=abs(h.predicted-h.observed)
            mae=float(h.error.mean()) if task=="primary" else float(h.groupby("sample_id").error.mean().mean())
            if not h.error.notna().all():mae=np.inf
            rows.append(dict(task=task,method=method,budget=int(budget),family=family,scope=scope,MAE=mae,reads=len(h)))
    by=pd.DataFrame(rows);by.to_csv(OUT/"resolved_by_source.csv",index=False)
    s=by.groupby(["task","method","budget","scope"]).agg(MAE=("MAE","mean"),worst_source=("MAE","max"),reads=("reads","sum")).reset_index()
    s.to_csv(OUT/"resolved_summary.csv",index=False);return s


def figures(s,pred):
    folder=OUT/"figures";folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1));colors={m:plt.get_cmap("tab10")(i) for i,m in enumerate(SHORT)}
    for ax,task,scope in zip(axes,("primary","temperature"),("all_query","common40_125")):
        for m in SHORT:
            d=s[s.task.eq(task)&s.scope.eq(scope)&s.method.eq(m)].sort_values("budget")
            ax.plot(d.budget,d.MAE,"o-",ms=4,label=SHORT[m],color=colors[m])
        ax.set(xticks=[0,1,2],ylabel="Source-equal Hall MAE (dex)");ax.grid(alpha=.2)
    axes[0].set(title="Original RT task: 29 specimens / 5 sources",xlabel="Specimen anchors per held-out source")
    axes[1].plot([2],[.230520],"kD",ms=6,label="Two-anchor Arrhenius line")
    axes[1].set(title="Common 40-125 K: 7 specimens / 2 sources",xlabel="Temperature anchors per held-out specimen")
    axes[1].legend(fontsize=6,loc="upper right");fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("prediction_comparison."+ext),dpi=180)
    plt.close(fig)
    line=read_result_csv(energy.BASE/"phase8_process_temperature/simple_anchor_predictions.csv")
    fig,axes=plt.subplots(2,4,figsize=(12,6.1))
    d=pred[pred.task.eq("temperature")&pred.budget.eq(2)]
    for ax,sample in zip(axes.flat,sorted(d.sample_id.unique())):
        g=d[d.sample_id.eq(sample)]
        base=g[g.method.eq("shared_joint_Tf_E")].sort_values("Tobs_K")
        measured=base[~base.obs_id.str.startswith("RT::")]
        ax.plot(measured.Tobs_K,measured.observed,"k.",ms=3,label="Published trace")
        a=base[base.role.eq("anchor")]
        ax.plot(a.Tobs_K,a.observed,"ko",mfc="none",ms=6,label="Allowed anchors")
        for m in ("shared_joint_offset","shared_joint_Tf_E","direct_joint_affine"):
            p=g[g.method.eq(m)].sort_values("Tobs_K")
            ax.plot(p.Tobs_K,p.predicted,label=SHORT[m],color=colors[m],lw=1.2)
        p=line[line.sample_id.eq(sample)&line.method.eq("two_anchor_line_p0.0")&line.budget.eq(2)].sort_values("T_K")
        ax.plot(p.T_K,p.predicted,"k--",lw=1,label="Two-anchor line")
        ax.set(title=("Xiang " if "2167" in sample else "McCandless ")+sample.split("::")[-1],
            xlabel="T (K)",ylabel="log10 Hall density (cm^-3)");ax.grid(alpha=.2)
    axes.flat[-1].axis("off");axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc="center",fontsize=7)
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("two_anchor_heldsource_curves."+ext),dpi=180)
    plt.close(fig)


def main():
    audit=json.loads((OUT/"contract_audit.json").read_text());assert audit["passed"]
    resolved=json.loads((OUT/"solver_budget_sensitivity/stalled_refinement_audit.json").read_text());assert resolved["passed"]
    summary=read_result_csv(OUT/"summary.csv")
    pred=read_result_csv(OUT/"solver_budget_sensitivity/predictions_resolved.csv")
    rs=summarize_resolved(pred);figures(summary,pred)
    c=rs[rs.task.eq("temperature")&rs.scope.eq("common40_125")&rs.method.eq("fixed_joint_Tf_E")&rs.budget.eq(2)].MAE.iloc[0]
    text=r"""# ML＋Sandbox：内部参数适应与掺杂预测完整验证

## 1. 研究方向和本轮实际推进

系统goal保持active，方向仍是ML＋物理Sandbox预测掺杂后性质。载流子是当前第一主终点，迁移率、电导率与有依据的候选排序仍待推进。本轮实现并验证内部物理参数的少样本适应，保留原29片/5来源室温主任务及7片旧Si样品的温度补充任务；没有新增物理样品、DFT/MD或GPU训练。

上一轮共享E(C)已经通过真实标签与完整冷热隐式梯度训练，变温优于固定能级，但只允许锚点修正一个输出偏移。本轮检验锚点能否用于修正Sandbox内部状态，从而改善未见点的预测。实际学到并跨条件复用的仍是有效关系和过程参照，不把低维适应称作新神经算法或唯一微观机制。

本轮11模型、92个新增基模型拟合记录、11277个局部校准记录全部保留。数值门槛19项通过，129个真实状态零更新复现最大误差4.49×10的负12次方，内部坐标差分最大缩放误差7.44×10的负9次方。独立重构10881次内层候选评估、16755项来源/样品/输出核对和5874行外层预测。旧阶段全部冻结文件未改。

## 2. 同一个Sandbox里，究竟适应了什么

物理层仍为实际Si/Sn掺入量及已知H→高温形成统计/化学计数与电荷中性→冻结母类库存→指定测量温度的FD电荷再平衡→单位Hall因子的nH。形成能目录、材料设置、氧压情景及未报告H/C的处理与阶段9相同。新增中性施主分支同时影响冷热统计，不是新HSE标签。

设训练好的内部坐标为theta0=(log10(Tf/K),log10(E0/eV))。Tf是有效平衡参照，E0是共享E(C)的能量尺度。背景B和浓度系数s固定为基模型训练值。新条件的theta=theta0+D z，D给定坐标尺度，z为无量纲局部修正。只改Tf、只改E0及二者联合分别消融。

给定允许锚点A，最小化

\[
\mathcal L_A(z)=\sum_{a\in A}[F(x_a;\theta_0+Dz)-y_a]^2+\lambda z^Tz,
\qquad l\leq\theta_0+Dz\leq u.
\]

F为物理模型输出的log10 Hall密度，x为化学及观测条件，y为锚点实测log密度。lambda是训练内选择的预测正则，不是测得的噪声；D的温度尺度0.05dex，能量尺度用预设六值或单能量模式0.3dex。边界沿用原温度logit范围和E0=0.003–0.2eV。无锚点或无限收缩时保持基模型。

所有导数沿完整冷热求解链计算：dstate/dtheta=−Fu的逆乘Ftheta，再通过输出链式法则得到dF/dtheta。库存随参数改变的路径被保留，不是只给最后预测加一个系数。新预测参数保存并复用，同一来源/样品的全部查询复用同一个更新。

## 3. 为什么两条观测也不等于两个真实微观参数

在当前点线性化F≈f0+Jz，其中J=(dF/dtheta)D。令r=f0−y，对平方损失求导并置零得

\[
\widehat z_{\mathrm{lin}}=-(J^TJ+\lambda I)^{-1}J^Tr.
\]

正lambda保证这个线性子问题严格凸。实际非线性目标不因此全局凸，必须记录多起点、边界、根残差及未收敛。代码从零修正和裁剪后的线性解开始，用完整IFT训练局部参数。

若J的奇异值为sigma，局部Gauss–Newton数据分辨率为sigma²/(sigma²+lambda)。单锚点二维J必有零空间，正则让优化答案唯一并不意味着两参数由数据唯一决定。本轮双参数、双温度锚点的7片解虽然数值秩都是2，但有效数据维数仅约0.056–0.981，全部低于1。部分弱方向主要由先验限制；这些分辨率值不是经过校准的后验置信区间，也不是真实微观识别证明。

## 4. 对照、划分和计算预算

11模型包含原目录RT＋偏移/Tf适应，固定电离能＋偏移/双适应，共享E(C)＋偏移/Tf/E/双适应，直接Ridge＋偏移/截距斜率及剂量直线＋偏移。经验双坐标可用两个锚点修正截距及剂量或逆温度斜率，防止只给物理模型曲线形状适应能力。两温度锚点普通logn=a+b/T直线仍是强对照。

原室温外留整DOI，内层留训练DOI，原0/1/2样品锚点、guard、29/23/18查询及固定共同18片不变。温度外留整DOI；内层在唯一可用训练温度DOI留一样品，并移除该片全部RT/曲线标签。原RT来源的其余样品仍可训练。温度内层是同来源样品验证，不冒称跨来源内层验证。

温度每片0/1/2是RT和约150K锚点，查询142/138/131；共同40–125K80点、图读区间相交90点及全查询分别报告。模型及正则仅用相应训练内分数选择。本轮温度任务使用温度专用样品内层，区别于阶段9直接转用RTlambda；改选参协议本身的收益必须与内部参数更新的收益分开。

lambda固定网格0.001/0.01/0.1/1/10/无穷大。双内部模型六能量尺度×六lambda，直接Ridge六alpha×六lambda，都不超过36候选；简单单参数控制不重复无意义配置。基模型两起点每起点最多60评估，主局部拟合两起点每起点最多30评估。候选上限相同不等于秒数相同：新增基模型累计88.61秒，局部校准累计704.55秒，均不含数据整理/进程启动/报告。保存45265次成功锚点向量回调；逐冷热根迭代未单独插桩，不能将回调数冒充根迭代数。

## 5. 原室温主任务结果：仍未取得一致增益

以下为5来源等权MAE，单位dex，主协议30评估上限。全部数值收敛诊断后，原室温预测表未改变。

__PRIMARY__

共同18片进一步排除查询变化影响：

__COMMON__

共享E(C)双适应两样品锚点0.287401，比同关系输出偏移0.314286改善；逐来源2个更好、3个相同，没有更差。但原目录RT＋Tf适应为0.255590，仍更好。一个样品锚点时双适应恶化为0.362362，TEOS来源的误差由同关系偏移约0.295升至约0.725。增加自由度会过拟合锚点；训练内选择也不能保证跨来源成功。

原目录RT＋Tf两锚点只比原目录RT＋偏移0.257810改善约0.00222，来自1个来源，其余4个相同。不能将这点差距提炼为稳健的统一物理优势。全部方法在高浓度Sn来源仍有较大误差，最高剂量低Hall点完整保留。

## 6. 变温结果：有条件增量及其实际大小

共同80查询，先样品内平均、再来源内样品等权、最后两来源等权。下表主协议原始结果保留；固定能级双适应的数值敏感性修正见第8节，共享模型数字不变。

__TEMPERATURE__

两温度锚点的共享双适应0.206518，优于普通两点线0.230520、当前直接Ridge截距斜率0.250945和同信息固定能级双适应约0.349。相对当前直接双坐标，两温度来源均改善；相对本轮共享关系＋偏移0.218581则仅1来源改善、1来源更差，平均下降0.012062dex，约5.5%。只有两个Si温度来源，不能声称统计显著、普适或新Sn验证。

单温度锚点时，共享输出偏移0.254444优于双内部适应0.305170。仅适应电离能的两锚点0.215219接近双适应，说明当前标签下第二内部坐标的额外价值有限。不得只报告两锚点最佳一列。

全部温度查询如下，保留高温和低温所有查询：

__ALLTEMP__

在全部查询中，共享双适应0.195072与共享输出偏移0.195513几乎持平，差值仅0.000441dex；普通两点线0.207130。共同温区增量不能扩写成全温区明显优势。读数包络90点的同模型两锚点0.218185，输出偏移0.226208；像素区间仍不是实验置信区间。

![原室温与变温两种任务。横轴锚点单位不同；所有模型和强对照见完整CSV。](figures/prediction_comparison.png)

## 7. 收益归因：大部分改善来自正确使用训练内选参

阶段9共享关系＋偏移在温度预算2为0.290988。本轮同输出偏移为0.218581，再加双内部适应为0.206518。前一步改变了温度内层划分及正则候选范围，后一步才是相同本轮协议下内部适应的增量。不能把0.291→0.207的全部下降都说成新物理或新ML算法贡献。

直接Ridge零温度锚点也从阶段9约0.639变为本轮0.323，来自温度训练内的alpha选择，而非模型结构新增。这一控制表明验证/选参契约本身会显著改变结论。共享关系仍有价值：本轮相同适应框架下，学E(C)相对固定能级的改进较大；但其数据范围、低维有效关系和原室温局限都要保留。

![7片既有样品，两个允许温度锚点用空心圆标出；黑点含用于展示的guard点，误差计算排除guard。所有来源/样品均展示。](figures/two_anchor_heldsource_curves.png)

## 8. 数值严谨性：未收敛控制也必须认真处理

主实验8276个局部优化起点中39个未收敛，涉及24个校准记录；其中15个记录的最低有限目标来自未收敛起点，1个属于被选中的外层固定能级控制。没有求根失败、查询缺失或静默样品删除。所有物理更新的带惩罚锚点目标均未高于基模型目标；这只是当前运行核对，不等于留出误差保证。

为消除比较依赖迭代上限的风险，对全部24个含未收敛起点的记录统一把上限提高至120，其余在30以内已终止的轨迹复用。48个新起点后仍有2个最低目标未收敛，原convergence_audit.json的全收敛检查为false，保留。其余内层候选重新评分，模型配置和lambda选择没有改变。

两个停滞点都来自固定能级的Xiang D双适应。直接在这两个实际状态检查IFT与中心差分，最大细步长缩放误差约7.32×10的负9次方，未发现梯度错误。随后用同一目标/边界的L-BFGS-B从当前解及零修正出发，投影梯度分别1.77×10的负9次方和3.89×10的负10次方，满足本轮1e-6的一阶条件。再次核对内层选择没有改变。

数值修正后，共同温区固定能级双适应预算2为__RESOLVED_FIXED__dex，其他主要比较和共享模型数字不变。从120评估到最后边界解，单个外层预测最大变化约0.002564dex。原30/120及停滞记录、差分与修正结果分开保存；不将全局收敛、所有初值成功或真实微观正确性作为已证明。

## 9. 物理响应诊断怎样指导下一轮模型

对相同来源留出共享基模型，另检查29个RT和7个最低温状态，不重新训练或选参。RT的绝对能量响应dlogn/dlogE0中位数约0.01682；在7片配对中，6片低温对E0的敏感度比RT约高8.8–29.1倍。这支持当前电离关系更擅长约束低温曲线，而不足以修复所有室温库存/激活误差；它是模型内诊断，不是唯一真实失效机制。

Xiang A是重要例外：最低温已由固定背景主导，能量响应仅约3.02×10的负5次方；冷/RT响应比约5.08×10的负5次方。不能笼统声称低温总是更有信息。本轮固定B且原拟合下界为1e14，在这个平台附近仅改E难恢复实测冻结斜率。背景下界与其物理含义需单独检查。

Mauze最高两剂量间化学浓度仅增约1.49倍，图读Hall下降约3.25dex，而该基模型变化约0.042dex。这是跨样品差分，不是严格控制其他条件的剂量导数；逐样品温度/历史等未完整报告。原作者正文p5也将高浓度下降讨论为未完全电离或显著补偿的候选，未唯一识别原因。因此下一轮应检验掺杂母类库存/激活分配及缺失工艺影响，不能用再调低温E关系掩盖室温失配，也不能删除该困难点。

## 10. 下一步仍围绕ML＋Sandbox预测

优先回到原室温主终点，审计现有目录是否提供能够改变有效激活/补偿的母类分配方向。若采用共享库存分配学习，必须保留总化学掺入量、同母类全电荷态共同权重和冷热统计一致性，区分实际形成能与有效过程偏置；先排除可被化学势吸收的规范方向，再训练少量跨条件共享关系。现有目录不支持的中性库存只能作为明确的新有效状态假设，不编造DFT能量。

需要同时保留无学习、常数分配、同复杂度经验激活曲线、当前共享电离关系和直接ML对照；继续用原29/5及温度补充验证，不以温度小增益宣布主目标完成。背景下界及并联Hall只作针对性候选检验，未知H/C/氧压不能用拟合值冒充实测输入。完整迁移率、电导率、新元素与稀土排序仍未得到本轮认证，原v4不替换为未验证输运系统。

## 11. 代码、证据和相关文献

parameter_adaptation.py提供不依赖数据文件的有界正则适应接口；phase10_internal_adaptation.py实现11组相同标签预算验证，phase10_adaptation_gates.py、phase10_adaptation_audit.py、phase10_solver_budget.py、phase10_stalled_gradient.py与phase10_stalled_refinement.py分别记录实现、来源/评分、迭代敏感性、真实停滞梯度和有界解。完整方程与选参约定见docs/ML_SANDBOX_PARAMETER_ADAPTATION_20261017.md。

另交付thermal_predictor.py与research_predictor_v2：保存五个来源排除训练模型的共享关系/物理状态、显式材料与氧压情景及能量输入校验。独立接口加载JSON后即可计算指定掺入量的变温载流子/带电预算，并用允许锚点更新内部坐标；不导入实验训练表。447行保存温度预测复现最大误差约6.51×10的负13次方，文件访问审计未读取训练标签表。模型JSON保存适应设置与锚点哈希，不能把这个接口认证成已验证的新材料或完整输运预测。初版隐式默认压力导出及对应代码/验证也保留，v2将压力显式写入模型。

protocol.json绑定原始代码/输入，methods/、base_fits/、calibrations/保留所有记录；summary.csv与by_source.csv是原主协议，resolved_summary.csv及solver_budget_sensitivity/predictions_resolved.csv保存数值诊断后版本。canonical_prediction_annotations.csv明确原数值/Hall字段不等于全物理有效；输出偏移本身不生成新的电荷中性状态，内部适应则重新求解完整状态。contract_audit.json、各独立检查及目标审计均保留。结果及代码随后冻结到阶段10清单，下一执行入口独立保存。

方法背景：[CAVIA原论文](https://proceedings.mlr.press/v97/zintgraf19a.html)研究共享参数与少量任务参数的适应；本轮没有其高阶元梯度训练。[Kennedy与O'Hagan2001](https://doi.org/10.1111/1467-9868.00294)研究计算模型校准，[Brynjarsdottir与O'Hagan2014](https://doi.org/10.1088/0266-5611/30/11/114007)讨论模型失配与参数混淆；本轮正则拟合不等于完整贝叶斯失配模型。高浓度Sn现象及工艺证据来自[Mauze等2020](https://doi.org/10.1063/5.0027870)，候选原因不是本研究已证实的缺陷身份。
"""
    for token,task,scope in (("__PRIMARY__","primary","all_query"),("__COMMON__","primary","common18"),
        ("__TEMPERATURE__","temperature","common40_125"),("__ALLTEMP__","temperature","all_query")):
        text=text.replace(token,table(summary,task,scope))
    text=text.replace("__RESOLVED_FIXED__",f"{c:.6f}")
    target=OUT/"ML与Sandbox_内部参数适应与掺杂预测验证.md";target.write_text(text,encoding="utf-8")
    energy.write(OUT/"report_provenance.json",dict(created_utc=energy.now(),generator_sha256=energy.sha(Path(__file__)),
        hashes={n:energy.sha(OUT/n) for n in ("summary.csv","contract_audit.json",
            "solver_budget_sensitivity/stalled_refinement_audit.json","solver_budget_sensitivity/predictions_resolved.csv")},
        direction="ML+Sandbox predictive improvement,originalRTprimaryretained",goal_complete=False))
    print(target)


if __name__=="__main__":main()
