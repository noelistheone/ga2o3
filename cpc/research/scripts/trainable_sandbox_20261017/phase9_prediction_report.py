"""Render phase9 evidence from saved predictions; no additional fitting."""
from pathlib import Path
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.BASE/"phase9_complete_temperature"
LABELS={
    "no_extension_RT":"原目录，仅RT训练",
    "no_extension_joint":"原目录，联合训练",
    "fixed_binding_joint":"固定电离能，联合训练",
    "learned_binding_joint":"学习共享电离能",
    "learned_screen_joint":"学习共享浓度关系",
    "dose_line_RT":"实际剂量直线",
    "direct_ridge_RT":"直接Ridge，仅RT训练",
    "direct_ridge_joint":"直接Ridge，联合训练"}
ORDER=list(LABELS)
SHORT={"no_extension_RT":"Original states / RT", "no_extension_joint":"Original states / joint",
    "fixed_binding_joint":"Fixed binding / joint", "learned_binding_joint":"Learned binding / joint",
    "learned_screen_joint":"Learned E(C) / joint", "dose_line_RT":"Dose line",
    "direct_ridge_RT":"Direct Ridge / RT", "direct_ridge_joint":"Direct Ridge / joint"}


def table(frame,scope):
    t=frame[frame.scope.eq(scope)].set_index(["method","budget"])
    lines=["|模型|0锚点|1锚点|2锚点|", "|---|---:|---:|---:|"]
    for m in ORDER:
        values=[float(t.loc[(m,b),"MAE"]) for b in (0,1,2)]
        lines.append("|"+LABELS[m]+"|"+"|".join(f"{v:.6f}" for v in values)+"|")
    return "\n".join(lines)


def figures(primary,temp,pred):
    folder=OUT/"figures";folder.mkdir(exist_ok=True)
    colors={m:plt.get_cmap("tab10")(i) for i,m in enumerate(ORDER)}
    shown=["no_extension_RT","no_extension_joint","fixed_binding_joint",
           "learned_binding_joint","learned_screen_joint","direct_ridge_joint"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,frame,scope in zip(axes,(primary,temp),("all_query","common40_125")):
        for m in shown:
            q=frame[frame.method.eq(m)&frame.scope.eq(scope)].sort_values("budget")
            ax.plot(q.budget,q.MAE,"o-",label=SHORT[m],color=colors[m],ms=4)
        ax.set(xticks=[0,1,2],ylabel="Source-equal MAE (dex)");ax.grid(alpha=.25)
    axes[0].set(title="Original RT prediction: 29 specimens / 5 sources",
                xlabel="Specimen anchors per held-out source")
    axes[1].plot([2],[.230520],"kD",label="Two-temperature-anchor line",ms=7)
    axes[1].set(title="40-125 K: 7 existing specimens / 2 sources",
                xlabel="Temperature anchors per held-out specimen")
    axes[1].legend(fontsize=6,loc="upper right")
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("prediction_comparison."+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(12,6.1))
    for ax,sample in zip(axes.flat,sorted(pred.sample_id.unique())):
        d=pred[pred.sample_id.eq(sample)&pred.budget.eq(0)&~pred.obs_id.str.startswith("RT::")]
        reference=d[d.method.eq("learned_screen_joint")].sort_values("Tobs_K")
        ax.plot(reference.Tobs_K,reference.observed,"k.",ms=4,label="Published measured trace")
        for m in ("no_extension_joint","fixed_binding_joint","learned_screen_joint","direct_ridge_joint"):
            q=d[d.method.eq(m)].sort_values("Tobs_K")
            ax.plot(q.Tobs_K,q.predicted,color=colors[m],lw=1.3,label=SHORT[m])
        ax.set(title=("Xiang " if "2167" in sample else "McCandless ")+sample.split("::")[-1],
            xlabel="Measurement T (K)",ylabel="log10 Hall density (cm^-3)")
        ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc="center",fontsize=7)
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("zero_anchor_source_heldout_curves."+ext),dpi=180)
    plt.close(fig)


def main():
    audit=json.loads((OUT/"contract_audit.json").read_text())
    assert audit["passed"]
    primary=read_result_csv(OUT/"primary_summary.csv")
    temp=read_result_csv(OUT/"temperature_summary.csv")
    pred=read_result_csv(OUT/"temperature_predictions.csv")
    pars=read_result_csv(OUT/"fitted_parameters.csv")
    pars=pars[pars.training_sources.eq(4)]
    figures(primary,temp,pred)
    text=r"""# ML＋Sandbox：完整变温层与共享内部关系预测检验

## 1. 方向核对与本轮回答的问题

系统goal保持active。研究方向是由机器学习和物理Sandbox共同预测掺杂后材料性质。当前先验证实际掺入量到Hall载流子浓度，迁移率、电导率和有依据的候选排序属于后续性质链。理论与可识别性分析用于改进预测模型；仅有解释报告不能完成该目标。

本轮已把显式材料参数、完整供应能量目录、高温库存、变温电荷再平衡和共享内部关系统一成一个可训练计算链。学习到的参数进入形成统计及电荷平衡，通过隐式导数训练，随后在整篇文献来源留出时复用。它区别于最后在性质输出上加一个自由偏移；同时，本轮学习器仅是低维有效关系，不能将其称为已经发明了新ML算法。

本轮要回答两个问题：同一个内部关系能否同时改善原室温主任务与变温预测？新增自由度的收益能否超过固定物理、直接ML和简单锚点控制？全部比较都保留，不根据答案更换主终点。

## 2. 数据是什么，哪些信息仍然缺失

原主任务仍为29片Si/Sn样品、5个DOI来源。0/1/2锚点是每个留出DOI中的样品标签；对应查询29/23/18片，并另保留跨预算完全相同的18片查询。主分数先在同来源查询中平均，再在5来源等权平均，以免11片的来源支配3片来源。

温度信息来自其中7片既有Si样品，分属Xiang与McCandless两个原有DOI，共142个载流子曲线读数。它们不是142个独立样品，也没有新增物理样品。54个迁移率图读数保留在阶段8，本轮不把它们当作已经验证过的变温输运训练。全部曲线读数均来自公开处理后图，不是仪器原始数据；像素上下界也不是实验置信区间。

联合训练使用原RT表值和温度不高于275K的载流子图读数。先在同片样品内平均，再在同来源的样品内平均，最后来源等权，以避免同片20多个温度点相当于20多片样品。外留一个DOI时，同时去掉其全部室温及温度标签。所有数据来源此前已在研发中接触，因此这是回顾性来源留出，不能称盲测。

Xiang C/D有报告H浓度，分别作为额外固定掺入元素供应给新物理模型；其他样品H未知，本轮采用明确的H=0条件情景。未检出或未报告不能当实测零。D的碳堆积已知，但KROGER目录没有C，故保留未建模输入而不编造形成能。直接ML也获得已知H/C浓度及缺失标记。原29片均缺乏可直接使用的氧分压，仍固定为公开声明的情景输入。Mauze逐样品生长温度缺失，700–900摄氏度区间不能伪装成逐片实测温度。

本轮物理模型的有效Tf及背景正电荷是跨训练条件共享校准量，尚未学成由每片生长条件决定的过程模型。直接ML获得生长温度代理等可用信息；“相同信息可用”不等于所有模型使用同一特征函数。新模型还采用单位Hall因子，旧阶段的密度经验Hall因子与H供应不同，所以不能把新旧全部差异归因于内部学习。

## 3. 先把材料输入变成显式、不可变的对象

MaterialProfile保存参考态密度NC、NV、带隙Eg0、带边分配f、参考温度、Varshni参数和局域空穴参数及来源。训练、推断、缓存均使用同一对象及其SHA256。新层不修改旧thermo模块的全局材料常数，支持同一进程交错调用不同材料情景。

本轮正式比较固定NC=3.33×10的18次方、NV=1.71×10的20次方每立方厘米，Eg0=5.1eV，f=0.4。仓库默认的1.0×10的19次方、5.0×10的20次方、5.0eV只用于兼容复现。四个材料偏移坐标进入导数检查，但预测训练时固定为零；没有根据外层答案额外调整材料常数。

两套参数下共58个真实样品300K结果已重放。这证明指定实现的版本一致性，不证明所有形成能参考、真实材料常数或冷却过程已经准确。发表值与仓库默认不一致的事实及原45次重拟合均留在阶段8。

## 4. 完整Sandbox如何从化学掺入量算到载流子

高温未知量为费米能级EF及每个固定掺入元素的化学势。对母类i的电荷态q，计算

\[
N_{iq}=a_{iq}\exp\left[-\frac{H_{iq}(T)+qE_F
-3k_BT s_{\mathrm{vib}}(T)\Delta N_i
-\Delta\mathbf N_i\cdot\boldsymbol\mu}{k_BT}\right].
\]

N为每立方厘米电荷态密度，a为目录前因子，H为参考形成能，q为相对正电荷整数，EF与H用eV；kB为eV/K，T为K。DeltaN为原子数变化，mu为元素化学势；振动项沿用指定目录近似。每种实际掺入量都由对应原子计数约束，另加电荷中性方程。

给定高温根后，母类原子库存冻结为Ci=sumq Niq。同母类的组成相同，所以低温条件电荷概率中的化学势项消去：

\[
\pi_{iq}(T_o,E_F)=\frac{g_{iq}\exp[-\beta_o U_{iq}(T_o,E_F)]}
{\sum_r g_{ir}\exp[-\beta_o U_{ir}(T_o,E_F)]},
\quad N_{iq}^{\mathrm{cold}}=C_i\pi_{iq},\quad \beta_o=(k_BT_o)^{-1}.
\]

g为同母类相对统计权重，U包括H、qEF及保留的振动项，To为测量温度。库存不变并不意味着电荷占据不变。每个温度重新解

\[
Q(E_F)=B+p+p_{\mathrm{STH},1}+p_{\mathrm{STH},2}-n
+\sum_i C_i\langle q\rangle_i=0.
\]

B是本轮假定全电离且温度不变的背景正电荷，n/p为FD电子/空穴，两个STH项为所选模型的局域空穴。B是有效闭合量，不能当独立测得的杂质浓度。nH=n/rH，本轮rH=1且假定单电子Hall；即使n大于p，也不能据此排除真实样品的并联电子通道或异常Hall因子。

固定Ci时，微分归一化指数概率得到

\[
\frac{\partial\pi_{iq}}{\partial E_F}
=-\beta_o\pi_{iq}(q-\langle q\rangle_i),\qquad
\frac{\partial\langle q\rangle_i}{\partial E_F}
=-\beta_o\operatorname{Var}_{\pi_i}(q)\leq0.
\]

第一步是对指数求导；第二步用分母导数减去平均电荷；第三步乘q求和得到二阶矩减均值平方。结合单调FD电子和空穴，可得冷态Q对EF严格递减。这给出指定区间根的唯一性与局部导数依据；不等于已证明高温多变量根全局唯一。

输出另含总绝对电荷预算B+sum Ci〈abs(q)〉和平方电荷预算B+sum Ci〈q²〉。二者是状态统计量，后一项在特定散射近似下有用途；仅有它们还不能完整预测迁移率。原300K经验CT只开放为兼容接口，不外推成低温迁移率结论。

## 5. ML实际学了什么，为什么同时影响冷热两阶段

原目录中的相应孤立Si/Sn施主只有+1统计分支，不能仅靠固定带电态在冷却时描述常规施主中性占据。本轮对经组成及位点核对的Si母类169/170、Sn母类151/152，测试同组成、同母类的有效中性分支：

\[
H_0(T)=H_+(T)+E_C(T)-E_D(C_X),\qquad
E_D(C_X)=E_0\exp[-s x],\quad
x=\left(\frac{C_X}{10^{18}\,\mathrm{cm}^{-3}}\right)^{1/3},\quad s\geq0.
\]

EC为当前材料设置下导带边，ED为有效束缚能，CX为实际掺入元素浓度。中性态相对统计权重固定为2，保证给定母类库存时

\[
f_+=\frac{1}{1+2\exp[(E_F-E_C+E_D)/(k_BT)]}.
\]

指数来自中性与正电态的能量差，相对权重2是模型假设。同一中性分支同时进入高温形成统计和冷态概率，不能在低温临时添态却保持旧的高温库存。该扩展不增加掺杂原子总数，也不是获得了新的HSE中性态计算标签。

三种嵌套变体分别固定E0=0.0366eV且s=0、学习共享E0且s=0、同时学习共享E0和s。同一训练折内，所有浓度及样品复用相同参数，不给每条曲线一个自由电离能。跨来源预测时复用训练好的关系。它是有效局域统计的低维学习，尚不能解释为真实屏蔽定律。

学习路径的导数包括

\[
\frac{\partial E_D}{\partial\log_{10} E_0}=(\ln10)E_D,\quad
\frac{\partial E_D}{\partial s}=-xE_D,\quad
\frac{\partial E_D}{\partial\log_{10} C_X}
=-\frac{s\ln10}{3}xE_D.
\]

这些导数进一步影响高温库存及低温电荷，而非只修正最后的nH。令u包含冷热根、w为共享参数，求导来自

\[
\mathcal F(u,w)=0,\quad
\frac{du}{dw}=-\mathcal F_u^{-1}\mathcal F_w,\quad
\frac{dy}{dw}=y_w+y_u\frac{du}{dw}.
\]

第一式对w求导后得到Fu du/dw+Fw=0，只有Fu可逆且相应分支光滑时才能解第二式。第三式是输出链式法则。联合矩阵是高温根与冷态根组成的下三角块结构，必须保留库存随w变化的通路。得到dy/dw后，才用真实训练标签的加权平方损失更新E0、s、Tf、B。根的数值收敛不等价于参数唯一，也不保证模型真实。

## 6. 训练和对照怎样避免把增加自由度当成改进

八个模型都保留完整逐点预测。前五个为原目录RT、原目录联合、固定电离能联合、共享电离能联合、共享浓度关系联合；后三个为剂量直线、RT直接Ridge和联合直接Ridge。RT-only是忽略允许温度标签的诊断，不是与联合训练具有完全相同训练数据的因果对照；判断关系学习应首先比较联合训练组。

外层留整DOI，内层在另外4个DOI中留一来源选Ridge惩罚及锚点偏移收缩强度。Ridge有6个惩罚候选，物理模型每种一个配置、两个优化起点、每起点最多60次函数评估。Tf范围600–1950K，B为10的14–18次方每立方厘米，E0范围0.003–0.2eV，s为0–5。相同候选上限或相同优化评估上限不等于相同运行秒数；本轮分别记录实际成本。

锚点只校准一个输出偏移。其正则解为delta=sum(ya−fa)/(k+lambda)，k为可用锚点数，lambda从内层选择。lambda无穷大等于拒绝适应，不自动保证多给一个锚点就能改善。

温度补充任务的0/1/2是每片的温度锚点：RT表值及约150K读数；对邻近点使用原guard规则。查询142/138/131，40–125K共同中心温区80点，图读区间与该范围相交的90点另作敏感性。温度适应直接沿用原RT任务内层选出的lambda，不再用外层曲线答案选温度超参数。这与阶段8专门温度内层调参不同，故不能把跨阶段全部差异当作完整Sandbox结构的净贡献。

审计独立重构了1080个嵌套训练上下文、3240次候选内层评分及4272行保存预测的来源排除与锚点偏移。共有227个缓存拟合记录、150个物理优化起点；1个起点未收敛，但没有被选为任何拟合记录的最终参数，原记录保留。没有求根失败；累计拟合计时473.61秒，不包含进程启动、数据整理和报告。数值/模型假设分开注释，未删任何困难样品。

## 7. 原室温主任务：没有取得一致提升

以下是5来源等权MAE，单位dex。0/1/2对应查询29/23/18片，不能直接把横向变化全归为更多锚点。

__PRIMARY__

共享关系相对固定电离能联合模型改善，但相对原目录RT控制并未一致改善：零锚点为0.271299对0.266289；两锚点为0.314286对0.257810。共享常能级零锚点0.267756也没有超过原目录RT控制。它们均超过本轮直接Ridge，并不意味着超过全部合适基线。

再在完全相同18片查询上比较：

__COMMON__

在共同18片中，原目录RT控制零锚点0.250950、两锚点0.257810；共享浓度关系为0.280842和0.314286。对原主终点，本轮不能宣称增量已建立。所有模型在Mauze高浓度/平台下降来源仍有较大误差；不能靠删去最高剂量低载流子点改善结论。

## 8. 变温预测：内部共享关系有条件地提供了增量

以下对同一80个中心温区查询，先样品内平均，再来源内样品等权，再两个来源等权。

__TEMPERATURE__

共享E(C)零锚点0.297549，低于固定电离能0.857479、共享常能级0.519508和本轮联合直接Ridge0.638579。两来源分别为0.377626和0.217473，均低于相应直接Ridge及固定电离能。全部查询结果也保留：共享E(C)零锚点0.264915，两锚点0.267847。这里的增量是本轮同协议中的有效关系学习信号，只有两个Si温度来源，不是普适性或Sn变温验证。

固定的普通两温度锚点直线logn=a+b/T在两锚点预算仍更好：共同温区0.230520、全部查询0.207130；共享E(C)对应0.290988和0.267847。RT/最后温度常数、固定T的0.75和1.5次幂修正的两条线全部保留在对照CSV中，没有挑选后删除。固定层、常数能级等多组分数跨温度锚点不变，是内层选择lambda无穷大的结果，不是新增锚点没有被记录。

![左：原室温主任务；右：同片变温任务。横轴的锚点单位不同，图中的两点直线仅在预算2可用。](figures/prediction_comparison.png)

![七片样品的零锚点整来源留出预测。黑点是公开实测迹线；所有样品均展示，偏差没有隐藏。](figures/zero_anchor_source_heldout_curves.png)

## 9. 数学正确、数值正确与物理解释的边界

165项数值检查全部通过，最大缩放中心差分误差1.32244×10的负7次方，包括两套材料设置、Si/Sn、加/不加中性分支、冷热不同温度、剂量/能量/材料/观测温度的隐式导数、原子/库存/电荷守恒、概率归一化及正性、同冷热温度无淬火恒等、58个实际样品300K重放，以及不同profile的交错调用。它们验证的是指定方程和FD数值近似的实现，不是165次材料实验。

GPU1桥已单独核验，一台可见物理GPU1，13维参数的一阶Torch值与梯度均与CPU实现一致。非线性状态根仍在CPU求解，梯度返回GPU；这不是已将完整求根改成GPU内核，也不支持高阶导数结论。GPU0未使用。

新注释不再将raw physical_valid误解为完整物理有效。原字段只检查根/梯度、稀释/位点数值规则及电子Hall条件。所有物理查询均通过这些规则，但固定电离能模型有3个不同温度观测、跨3种预算共9条预测低于1e13参考读取水平，全部保留。该参考水平不是各文献已报道的检测限。

孤立局域施主的两态统计存在适用域：高浓度时杂质带、离域和并联通道不能由一个ED(C)自动表示。即使ED趋近零，模型仍保留可占据局域态，因此它不自动满足金属态完全离化极限。McCandless的2.4×10的18次方每立方厘米仅用作Si Mott参考注释，不是Sn的已验证阈值，也不按它删点。

所有五个外层共享浓度关系拟合的B都靠近本轮下界。E0随留出来源约0.086–0.199eV、s约2.109–3.054；留出Xiang时E0接近0.2eV上界。它们是受当前关系、库存、Hall与数据约束的有效参数，不能解释为发现了真实孤立施主能级或唯一屏蔽强度。两个温度来源均为Si，Si/Sn共享关系仍缺乏独立Sn温度数据支持。

## 10. 本轮之后怎样继续改进预测方法

本轮没有把目标改成机制解释，也没有宣布“大创新已经证明”。已实现可训练、可复用的内部关系并出现变温预测增量，但原主任务未改进、简单两锚点线仍更强，完整变温迁移率尚未实现。下一轮应解决明确的模型失配，而非只增加网络或再做无目标扫描。

第一步，固定本轮输出，按样品核查低温冻结斜率、平台及室温残差是否要求新的局域/离域环境、额外补偿或不同Hall读出。至少比较能够产生平台的物理替代与原两态极限；不能直接在输出n上加一个无物理来源的自由地板。原B下界与E0接近上界的现象是失配/补偿诊断线索，不是唯一失效机制的证明。

第二步，在训练前明确新状态的组成、能量参考、简并、冷热作用及对应守恒，再对新共享关系做导数与极限检查。若采用局域与离域环境混合，必须说明它代表结构环境/有效谱权重何者，避免把电子占据概率当成独立原子库存；同组成分支也必须说明其简并前因子如何分配及是否双计数。

第三步，Hall读出必须与状态扩展相容。对同号、单位Hall因子、指定并联权重的简单通道，nH=(sum wnmu)²/(sum wnmu²)；因此相同表观nH平台可能由不同真实密度/迁移率组合产生。没有输运或几何证据时，不能将学到的环境混合唯一解释成Mott金属比例。这个歧义应限制模型和结论，而不成为终止性质预测研发的理由。

另一个可直接检验的限制是适应方式。本轮即使获得两个温度锚点，也只允许改一个输出偏移，不能调整激活斜率；普通两点线却能调整截距与斜率。因此下一项方法实验应比较内部能级/库存参数的受约束少样本适应与现有偏移，使用训练来源学习共享关系和适应先验，再仅用允许锚点更新；须先核对实际锚点雅可比与噪声条件，而非把两条观测当作两个微观参数的真值。它仍是ML＋Sandbox性质预测，不把低维参数拟合包装成新算法。

第四步，冻结同样29/5室温主任务与7曲线补充任务，以固定极限、相同信息的直接ML、常数/低维闭合、普通两点线作对照。只用训练内证据选择参数和新关系；由于这些外层已开发接触，仍报告为回顾性。新关系必须说明在哪些预算、来源与性质上获得增量；若主任务仍无提升，继续具体修正而不将温度子任务替代最终目标。

## 11. 交付、来源和复现入口

核心代码为src/sandbox/material_profile.py、temperature_defects.py；理论为docs/ML_SANDBOX_FULL_TEMPERATURE_20261017.md。phase9_temperature_gates.py执行数值核验，phase9_joint_prediction.py保存训练与来源留出预测；phase9_prediction_audit.py独立重构来源、训练权重、候选分数、锚点偏移并注释适用域。训练脚本按独立进程运行不同适配器，避免旧模块全局设置相互污染。

结果目录保存prediction_protocol.json、numerical_protocol.json、numerical_gates.json、gpu1_bridge_verification.json、227个fits JSON、各方法逐点预测、汇总/来源/参数/优化/规范注释CSV及contract_audit.json。输入/代码哈希绑定运行版本，所有不利结果和一个未收敛起点保留。既有85、847、1157、1091、13192、206、387项保护清单均核验无变化；阶段8以前预测器v4不替换，不将本轮变温载流子结果认证为完整性质链。

复现时使用ga2o3环境、PYTHONPATH=src与单线程BLAS；先运行phase9_temperature_gates.py，再独立运行phase9_joint_prediction.py，最后运行phase9_prediction_audit.py和本报告生成脚本。GPU桥只用CUDA_VISIBLE_DEVICES=1；根/小数据训练本轮使用CPU。新增代码与完整结果随后冻结到独立阶段9清单，继续入口另存。

材料设置来源：[KROGER正式PCCP论文](https://doi.org/10.1039/D4CP04817B)。公开同片变温与化学输入：[Xiang等2022](https://doi.org/10.1007/s40843-022-2167-x)、[McCandless等2022](https://doi.org/10.1063/5.0101132)。本文新增中性统计分支与E(C)是本研究有效模型假设，不归称上述文献已经推导或验证。原论文修订尚未启动，没有新增DFT/MD或物理实验，没有稀土排序结论。
"""
    text=text.replace("__PRIMARY__",table(primary,"all_query"))
    text=text.replace("__COMMON__",table(primary,"common18"))
    text=text.replace("__TEMPERATURE__",table(temp,"common40_125"))
    target=OUT/"ML与Sandbox_完整变温与共享关系预测检验.md"
    target.write_text(text,encoding="utf-8")
    energy.write(OUT/"report_provenance.json",dict(created_utc=energy.now(),
        generator_sha256=energy.sha(Path(__file__)),
        source_hashes={n:energy.sha(OUT/n) for n in ("contract_audit.json","primary_summary.csv",
            "temperature_summary.csv","fitted_parameters.csv","temperature_predictions.csv")},
        goal_status="active",report_scope="fulltemperature real-label shared-relation learning,not new physical experiments"))
    print(target)


if __name__=="__main__":main()
