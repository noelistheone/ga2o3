"""Prediction research synthesis: all retained comparisons and new theory."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.BASE/"phase56_validation"
REPORT=OUT/"ML与Sandbox_参考一致学习与预测方法总报告.md"

def table(headers,rows):
    text="| "+" | ".join(headers)+" |\n| "+" | ".join("---" for _ in headers)+" |\n"
    for row in rows:
        text+="| "+" | ".join(f"{v:.6f}" if isinstance(v,(float,np.floating)) else str(v) for v in row)+" |\n"
    return text

def main():
    canonical=read_result_csv(OUT/"canonical_comparison.csv")
    labels={"Dose line":"经验剂量线＋分层偏移","Direct Ridge":"直接Ridge","Direct trees":"直接树模型",
        "Full energies":"全部能量物理参照","Mean energies":"均值补能",
        "Kernel base":"原输入RBF","Projected reference":"守恒投影RBF：能量选参",
        "Projected property-selected":"守恒投影RBF：性质选参","Raw reference":"原始参考能量RBF",
        "Native reference":"本征公共参考RBF","Tuned Ridge":"统一网格普通Ridge",
        "Native Ridge":"本征公共参考Ridge","Nonlinear loss":"完整非线性损失",
        "Ridge +2 energies":"Ridge＋2条已知能量","Mean +2 energies":"均值＋2条已知能量",
        "ET +2 energies":"ET目标选点＋2能量"}
    selected=["Dose line","Direct Ridge","Direct trees","Full energies","Mean energies","Kernel base",
              "Nonlinear loss","Tuned Ridge","Projected reference","Projected property-selected",
              "Raw reference","Native reference","Native Ridge","Ridge +2 energies","Mean +2 energies","ET +2 energies"]
    rows=[]
    for name in selected:
        d=canonical[canonical.label.eq(name)&canonical.scope.eq("all_query")].set_index("budget")
        rows.append([labels[name],*[float(d.loc[k,"MAE"]) for k in (0,1,2)]])
    main_table=table(["模型","0锚点dex","1锚点dex","2锚点dex"],rows)
    er=read_result_csv(energy.BASE/"phase5_reference_features/summary.csv").set_index("method")
    energy_table=table(["同网格表示","能量宏MAE eV","适用数/47"],[
        [n,float(er.loc[m,"energy_macro_MAE_eV"]),int(er.loc[m,"physical"])]
        for m,n in (("rbf_base_energy","RBF原18维"),("rbf_atomic_energy","RBF原子守恒投影"),
                    ("rbf_closed_energy","RBF原子＋电荷投影"),("rbf_raw_energy","RBF原始已知参考能量"),
                    ("ridge_base_energy","统一网格Ridge原18维"),("ridge_closed_energy","统一网格Ridge守恒投影"))])
    cf=read_result_csv(OUT/"counterfactual/summary.csv")
    cf_table=table(["替换模型","能量替换量dex","过程重拟合量dex","最终MAE dex"],
        [[r.method,float(r.energy_change),float(r.process_change),float(r.final_MAE)] for r in cf.itertuples()])
    cs=read_result_csv(OUT/"all_full_chain_summary.csv")
    chain_table=table(["5配对子集模型","锚点","nH dex","muH dex","派生sigma dex"],[
        [name,b,*[float(cs[cs.origin.eq(origin)&cs.method.eq(method)&cs.budget.eq(b)].iloc[0][q+"_MAE"]) for q in ("nH","mu","sigma")]]
        for name,origin,method,b in (
            ("统一网格Ridge","phase5","ridge_base_energy",0),
            ("Ridge＋2能量","phase2","Ridge_diagonal_xi_k2",0),
            ("均值＋2能量","phase2","mean_goal_k2",0),
            ("全部能量参照","phase2","oracle_full",0),
            ("Ridge＋2能量","phase2","Ridge_diagonal_xi_k2",2))])
    src=read_result_csv(energy.BASE/"phase5_reference_features/new_property_source/summary.csv")
    source_table=table(["相关研究测试","0锚点dex","1锚点dex"],[
        [name,*[float(src[src.method.eq(method)&src.budget.eq(k)].iloc[0].nH_MAE) for k in (0,1)]]
        for name,method in (("nH等于实际剂量","identity_dose"),("统一网格Ridge","ridge_base_energy"),
                            ("守恒投影：性质选参","rbf_closed_property"),("Ridge＋2已知能量","P2_Ridge_goal_k2"),
                            ("全部能量参照","oracle_full"))])
    audit=json.loads((OUT/"contract_audit.json").read_text())
    text=r"""# ML＋Sandbox：参考一致学习、完整预测验证与方法总报告

本文继续用户指定的研发方向：通过可训练ML＋Sandbox预测掺杂性质。包含本轮的参考输入设计、两组固定协议计算、强基线核对、完整推断接口，以及能量与过程参数的交叉诊断。当前goal仍active；方法的数值实现已打通，稳定的复杂ML增量与可靠跨研究迁移尚未建立。

## 1. 当前方法具体怎样工作

输入不是只有一个元素符号。当前需要：Si或Sn、实际掺入浓度、已知缺陷结构/电荷目录、每母类参考能量及其他已供应能量。共享学习器补全被屏蔽的相对电荷能量，恢复形成能后交给Sandbox；Sandbox计算高温库存、冻结母类数量、室温电荷再平衡，最后输出Hall载流子。迁移率目前通过经数据校准的CT经验头，电导率由q*nH*muH派生。

现有可追溯工艺记录保留在数据契约中，但29条载流子记录都缺少可直接使用的氧分压；当前闭合使用有效冻结温度和背景施主。有效Tf不等同于实际生长温度。因此现在是有明确假设的条件预测原型，不能称为已准确掌握真实生长动力学。

ML确实参与内部物理输入：保存的学习器被不同浓度/条件复用，预测能量进入守恒求解器；完整非线性变体还把解析物理梯度反传到共享学习器。其训练响应来自已知能量产生的模型输出，不冒称新实验标签。原子参考、变量单位、隐式梯度、局部目标与非线性目标的逐步推导放在本文附录。

## 2. 数据契约和三种信息预算

| 预算 | Si已知/待补状态 | Sn已知/待补状态 |
| --- | --- | --- |
| 无额外能量观察 | 844/29 | 850/23 |
| 另供应2条能量 | 846/27 | 852/21 |
| 完整能量物理参照 | 873/0 | 873/0 |

另供应结构目录和母类参考态；缺失标签均来自同一个已发表KROGER数据库。新增2条只是已有公开标签揭示，不是新DFT。完整873状态参照属于不同信息预算，不作为同预算对手。直接经验模型允许忽略这些物理输入，实际使用的能量数为零；比较收益包含物理先验与模型结构，不能解释为纯网络结构的等信息因果对决。

能量学习有269条非参考标签、15种有多电荷标签的元素。材料主任务仍是29条、5个DOI来源组，仅Si/Sn。条件迁移率52配对/11来源；完整剂量到迁移率仅5个明确身份配对/2来源。每个算法重复使用同样数据，不增加独立样品数或来源数。

0/1/2锚点沿用最低独特剂量、再选中间独特剂量。锚点附近0.005dex的其他读数作guard，故查询29/23/18，另比较固定18个共同查询点。所有选参均在训练来源或训练元素内部；全部历史来源已反复查看，属于回顾性研发。

## 3. 理论如何被实验逐步修正

第一阶段打通可微状态求解、守恒与少锚点适应；单独增加能量偏移或温度元数据没有稳定收益。第二阶段发现能量拟合排名不能直接代表材料预测排名，并完成目标导向能量揭示回放及强简单先验对照。

第三阶段据此构造J^T W J局部性质损失。真实能量偏差较大时，完整响应Taylor余项很大，预测还可能离开电子Hall适用域。因此第四阶段直接优化完整非线性Sandbox响应，用IFT梯度与完整损失回溯。固定配置的改善在公平调参后没有变成稳定材料收益。

本轮第五阶段检查表示是否遗漏已知参考态。第六阶段再检验：逐元素投影是否删除了有用化学信息，改用已知本征缺陷确定公共参考。这两轮都有新的代码、门槛、嵌套训练、逐样本预测和不利测试，不以解释性文档代替实验。

## 4. 第五阶段：参考一致表示不是免费精度定理

每个元素的母类参考能量组成a。使用B=[1,dmGa,dmO]或[1,dmGa,dmO,qref]，计算rho=(I-BB^+)a。由(I-BB^+)B=0，元素参考变换被相消；后者还消掉电荷参考列。它是已知能量的守恒线性组合，不是新几何坐标、基元反应或捕获势垒。

四种输入全部保留：原18维、原子守恒投影、原子＋电荷投影和固定参考框架下的原始参考能量。每种RBF/Ridge分别使用相同六候选；两种选参规则复用相同内层拟合，不把16个方法事后混成一个有利结果。主比较预先固定为按能量选参的原输入RBF与电荷守恒投影RBF。

投影核验约1e-14eV；269条未知非参考能量全部置NaN后特征不变。35个原特征碰撞组中33个可被参考能量区分，但能区分原训练描述，不代表能够跨元素学习正确关系。

@ENERGY@

原始参考能量降低了相同RBF的能量误差，但适用点未增加；守恒投影没有稳定提高能量精度。原始参考版本只是固定输入能量框架下的对照，没有相同的不变性保证。所有表示用的参考能量原本已经供应，不能称新增独立量子信息或从零预测新元素。

![能量误差改善与真实材料误差可以不一致](figures/representation_vs_property.png)

这些虚拟工况的47个参照适用点来自9元素，而非15元素全部可作电子Hall预测。其余数值log(n/rH)可成为数学延拓，不能用巨大误差的相对下降百分比宣称材料准确度进步。

## 5. 第六阶段：以本征缺陷建立公共参考

预算内已知243个本征电荷态、45个本征母类。以D=[dmGa,dmO,q]做母类等权最小二乘，确定三个能量坐标beta；选取元数据预定的掺杂参考锚点，使用psi_i=a_i-a_anchor-(b_i-b_anchor)beta。参考变化时beta等变、psi不变。beta不是实际化学势或真实费米能级。

该设计保留更多相对化学变化，仍只是一项可检验假设。对实际队列的Si/Sn做外层留元素，每次训练内再留14元素，RBF/Ridge网格保持不变。结果没有稳定胜过相同网格的原输入版本，不能把它包装成已经验证的物理表示优势。

本轮同时发现正则化归因问题：统一网格普通Ridge达到0.287952/0.261249/0.225493dex。守恒特征Ridge约0.287781/0.261046/0.225291dex，差异极小；本征参考Ridge反而约0.289928/0.263584/0.227806dex。相较旧Ridge的主要改善来自归一化和正则化网格，不能归功于新物理特征。

## 6. 真实载流子主结果：所有强对照保留

@MAIN@

每列先在来源内平均绝对log10误差，再对5来源等权。完整数值/物理覆盖、最差来源和固定18查询另存canonical_comparison.csv。MAE不是逐点最大误差、置信区间或独立验证。

直接Ridge与64棵树模型重新采用同样的4维材料/过程输入和同一0/1/2锚点收缩网格；每模型最多六候选、训练内选择。实际耗时不同，不宣称等运行时间。额外能量2与0的差异必须分开计数。

![有证据的选项：变化查询集与固定共同查询集](figures/supported_options.png)

统一网格Ridge是当前较稳健的零额外能量选项；有两条显式能量观察时，Ridge＋2与均值＋2都值得保留。它们在旧队列上优于直接回归，但复杂ML相对简单先验的独立增量仍未成立。

完整结果还保留了ET独立方差＋2能量在0锚点的0.256122dex，略优于Ridge＋2的0.259106；1/2锚点又由其他方法更好。不能称一个模型在所有预算中绝对最好，也不应把事后最小值作为无偏性能估计。接口提供跨预算较稳定的选项，全部24个第二阶段方法和新方法原始结果仍可复现。

守恒投影的性质选参版本在0锚点较好、2锚点却更差。新增自由度与更多已知特征，并不保证少样本适应和跨来源外推同时改善。

## 7. 交叉诊断：能量替换与过程重拟合有多少补偿

固定每个来源排除后的两套参数，计算四格：00为完整能量/原过程；10为ML能量/原过程；01为完整能量/ML过程；11为ML能量/ML过程。没有重训，没有用诊断选择新超参数；完整能量在此只作诊断参照，不能算进原低预算预测。

对每个点，定义Delta_E=y10-y00、Delta_P=y11-y10，有严格恒等式e11=e00+Delta_E+Delta_P。绝对误差均值不能按此相加。

@COUNTERFACTUAL@

四个方法各29查询的四格都通过数值与当前电子Hall域检查；回放原预测差约1e-11dex。能量替换与重拟合的变化在24/29点方向相反，投影性质模型为27/29。由此可量化当前模型中的部分补偿；过程变化幅度约0.015—0.030dex，未把能量替换的0.060—0.134dex变化完全吸收。

所以不能再笼统声称所有失败都是新参数落进零空间。完整能量参照仍残留约0.260dex误差；这部分可能含过程闭合、DFT输入、未报告实验变量及数据读数误差，不能称为已证明的不可约下界，更没有被唯一归因到表面氧交换。

## 8. 完整迁移率链与派生电导率仍有明显限制

@CHAIN@

该表只对应5个事先按样品身份匹配的子集，2锚点后仅2个查询；其nH误差不能冒充29条主队列的精度。完整链muH误差约1.25—1.30dex，仍不可靠。52组条件muH的0.366609dex是给定实测nH的另一任务。

完整链只把预测nH送进来源排除的CT＋加权中位数输运头；目标实测nH和muH不参与该链的输入。sigma是q*nH*muH的派生量，不是第三个独立测量。缺陷q^2预算尚未成为具有充分配对验证的微观散射模型。

## 9. 相关研究的迁移失败，不能用旧队列成绩掩盖

Azizie2023最终版TableII有12组Hall配对，仅e/k两组明确与SIMS同样品匹配。化学输入来自约2e18/3e19cm^-3的圆整正文值。旧相关来源整体排除后，训练载流子26条/4组、输运49条/10组。该文已进入历史语料，与旧来源团队相关，是纠正配对后的相关研究测试，不是独立实验室盲测。[作者公开最终版](https://schlom.mse.cornell.edu/sites/default/files/research%20pdfs/5.0139622.pdf)

@SOURCE@

![跨研究失配：圆整数值范围不是置信区间](figures/adverse_transfer.png)

全部新增方法的预测或拒绝状态均保留。本轮16个参考表示方法与4个本征参考方法没有数值拒绝，预测有效不等于预测准确。冻结的9组化学剂量分辨率情景不能解释掉整个误差，也不是统计置信区间。

同研究的条件muH误差为0.050414dex，而剂量到nH仍失败。这提示当前任务应优先改进过程到缺陷库存的闭合；不能把条件输运的好分数当作完整链成功。低剂量锚点不能修正高剂量形状错误；有些1锚点结果比0锚点更差，完整保留。

## 10. 已交付的训练与预测入口

当前推荐入口为scripts/trainable_sandbox_20261017/predict_properties_v4.py，本地模型包predictor_bundle_v4。默认ridge_energy_tuned不需要额外能量观察；goal_ridge_2与goal_mean_2显式携带两条已知能量。另保留旧Ridge、普通核、完整非线性损失和全部能量物理参照。

最终过程参数用全部29条旧队列训练，经验输运头用全部52组旧配对训练；它们是部署用参数，不能作为新的留出精度。推断返回信息预算、供给能量来源、原始守恒状态、锚点偏移、有效过程参数、假设压力和训练剂量范围；不返回未经校准的区间。

v3验证12类保存/擦除输入回放与CLI；v4又验证12次继承模式不变及2次调参Ridge回放。被屏蔽能量可置NaN，额外两条能量从明示的输入观察读取；不会用当前数据库未知值补齐。先前v1完整数据库校验的问题及v2修复均保留。

稀土、绝对寿命与器件PDR仍无对应可靠输入/验证，接口明确拒绝。所有模型包为第三方数据的私有派生物，不直接公开发布。重现脚本应各自在独立Python进程中运行，以避免研究适配器的模块配置互相影响；实际预测类不依赖这些适配器。

## 11. 理论、文献与验收边界

ML能级预测、任务导向损失、目标导向计算选择和隐式求导都有既有工作。本轮核对Polak2022与Kumagai2026作者预印本的官方摘要，并重读KROGER本地作者文本的形成能/库存方法段；没有把摘要下载或搜索结果称为全文复现。[Polak等](https://arxiv.org/abs/2203.10349)、[Kumagai与Kiyohara](https://arxiv.org/abs/2605.16611)、[KROGER](https://doi.org/10.1039/D4CP04817B)

本轮10416个能量内层拟合、320种直接模型嵌套来源成员关系已核对；参考/本征方法的来源排除、未知能量擦除和原核回放也有机器记录。旧85条保护项、第一阶段847文件、第二阶段1157文件、第三四阶段1091文件未改变。没有新增DFT、MD或物理样品。

泛用直接模型评分器最初把physical_valid写成True，同时另标not_applicable_empirical；现将正式比较字段修正为NA，保留原文件，数值预测不变。经验模型没有微观状态，不能把这个占位字段算成物理验证。全部原比较、失败日志和方法清单保留。

当前交付的是可以训练、调用、比较和追溯的预测方法及其证据边界。尚未支持“新的复杂ML机制全面优越”或“大创新已被证明”；不能通过保留有利点、增加模型名称或改写论文来弥补。

## 12. 下一步仍应推进预测方法

新增相对电荷能量表示的两轮假设已做完受控检验。下一项具体研发对象应转向处理输入到受约束缺陷库存的共享关系：固定已知能量和观测定义，比较平衡库存、快速掺入/冻结极限及由ML学习的受限库存闭合，再用完全相同的来源、锚点、简单对照检验。

这不是把缺失工艺参数拟合成真实微观速率。必须先证明库存非负、被测掺杂总量守恒、冷态电荷解和导数适用性，并说明宿主交换是开放边界；若缺少过程时序，只能学习有效闭合，不能声称辨识真实反应势垒。若固定自由度的简单闭合更强，也应如实采用。

新研究标签已看过，任何据此设计的模型都属于开发，之后须再找具有相应输入的新来源检验。当前goal继续active，主线保持ML＋Sandbox掺杂性质预测，本文不替代这个尚未完成的研发终点。

## 附录A：状态求解、共享学习和隐式梯度

@BASE_THEORY@

## 附录B：参考态守恒投影的完整推导

@REFERENCE_THEORY@

## 附录C：本征公共参考的完整推导

@NATIVE_THEORY@
"""
    for key,value in {"@MAIN@":main_table,"@ENERGY@":energy_table,"@COUNTERFACTUAL@":cf_table,
        "@CHAIN@":chain_table,"@SOURCE@":source_table,
        "@BASE_THEORY@":(ROOT/"docs/ML_SANDBOX_TRAINING_THEORY_20261017.md").read_text().split("\n",1)[1],
        "@REFERENCE_THEORY@":(ROOT/"docs/ML_SANDBOX_REFERENCE_FEATURES_20261017.md").read_text().split("\n",1)[1],
        "@NATIVE_THEORY@":(ROOT/"docs/ML_SANDBOX_NATIVE_REFERENCE_20261017.md").read_text().split("\n",1)[1]}.items():
        text=text.replace(key,value)
    REPORT.write_text(text.strip()+"\n");print(REPORT)

if __name__=="__main__":main()
