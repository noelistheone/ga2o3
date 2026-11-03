"""Write the prediction-focused phase3/4 report from audited result tables."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.BASE/"phase34_validation"
REPORT=OUT/"ML与Sandbox_性质目标训练与完整预测验证.md"
NAMES={"global_mean":"能量均值","ridge":"Ridge能量学习","kernel_energy":"普通能量核",
    "kernel_energy_select_energy":"普通核：按能量选参","kernel_diagonal_l1":"局部对角性质损失",
    "kernel_full_l1":"局部完整性质损失","kernel_nonlinear_l1":"完整非线性性质训练",
    "kernel_nonlinear_or_energy":"12候选门控（辅助）","oracle_full":"完整能量参照"}

def table(headers,rows):
    lines=["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]
    for row in rows:
        lines.append("| "+" | ".join(f"{v:.6f}" if isinstance(v,(float,np.floating)) else str(v) for v in row)+" |")
    return "\n".join(lines)

def main():
    p3=energy.BASE/"phase3_target_loss";p4=energy.BASE/"phase4_nonlinear"
    audit=json.loads((OUT/"contract_audit.json").read_text())
    data=read_result_csv(OUT/"real_property_summary.csv")
    rows=[]
    for method in NAMES:
        d=data[data.method.eq(method)&data.scope.eq("all_query")].set_index("budget")
        rows.append([NAMES[method],*[float(d.loc[b,"family_equal_MAE"]) for b in (0,1,2)]])
    property_table=table(["模型","0锚点","1锚点","2锚点"],rows)
    rows=[]
    for e in ("Si","Sn"):
        old=json.loads((p3/"outer"/f"{e}.json").read_text())
        base=next(v for v in old["selections"] if v["method"]=="kernel_energy")
        new=next(v for v in audit["outer_selections"] if v["element"]==e and v["method"]=="kernel_nonlinear_l1")
        rows.append([e,base["property_MAE"],new["virtual_MAE_dex"],base["energy_MAE"],new["energy_MAE_eV"]])
    nonlinear_table=table(["留出元素","普通核响应dex","非线性响应dex","普通核能量eV","非线性能量eV"],rows)
    coverage=read_result_csv(p3/"virtual_coverage_summary.csv").set_index("method")
    rows=[[NAMES[m],float(coverage.loc[m,"mathematical_macro_MAE"]),
           f"{int(coverage.loc[m,'physical'])}/47"] for m in ("kernel_energy","kernel_diagonal_l1","kernel_full_l1")]
    coverage_table=table(["模型","数值延拓响应MAE dex","物理适用点"],rows)
    source=read_result_csv(energy.BASE/"new_property_sources/azizie2023_full_chain_summary.csv")
    newsource=read_result_csv(p4/"new_property_source/azizie2023_full_chain_summary.csv")
    import pandas as pd
    source=pd.concat([source,newsource],ignore_index=True)
    smethods=["identity_dose","global_mean","kernel_energy","kernel_nonlinear_l1","P2_Ridge_goal_k2","oracle_full"]
    snames=["nH等于化学剂量","能量均值","普通能量核","完整非线性训练","Ridge＋2条能量","完整能量参照"]
    source_table=table(["模型","0锚点nH dex","1锚点nH dex"],[
        [n,*[float(source[source.method.eq(m)&source.budget.eq(b)].iloc[0].nH_MAE) for b in (0,1)]]
        for m,n in zip(smethods,snames)])
    t=read_result_csv(p4/"transport/summary.csv")
    transport_table=table(["模型（0锚点）","nH dex","muH dex","派生sigma dex"],[
        [NAMES[m],*[float(t[t.method.eq(m)&t.budget.eq(0)].iloc[0][q+"_MAE"]) for q in ("nH","mu","sigma")]]
        for m in ("kernel_energy","kernel_nonlinear_l1","oracle_full")])
    theory=(ROOT/"docs/ML_SANDBOX_TRAINING_THEORY_20261017.md").read_text()
    text=r"""# ML＋Sandbox：性质目标训练、完整预测验证与继续研发边界

这是可训练掺杂性质预测项目的第三、四阶段记录，不是论文改写。方向仍是“通过 ML＋Sandbox 预测掺杂性质”；可识别性与数学分析服务于模型训练、验证和下一条输入选择。当前 goal 保持 active，不能用本报告替代尚未建立的预测优势。

## 1. 本轮真正完成了什么

原方案先让 ML 尽量降低缺陷能量误差，再把能量送入 Sandbox。第二阶段发现：能量分数较好的模型，真实载流子预测可能更差。因此，本轮先让损失关注 Sandbox 对哪些能量敏感；再根据局部近似失效的证据，改为直接计算完整非线性响应，并把隐式梯度传回共享能量学习器。

三种共享学习已经实际实现、保存、嵌套选参并接回真实性质任务：普通能量核、局部性质敏感核、完整非线性性质训练。最终还组装了可调用的 ML→能量→Sandbox→载流子→经验迁移率→派生电导率接口。没有把 ML 变成仅作图的标签，也没有把方法改写成纯机制解释。

关键结果：固定配置试验中 Si 虚拟响应得到明显改善，但训练内选参后的普通能量核已能达到相同量级；完整非线性训练没有建立真实材料的增量优势。完整链的迁移率验证仍很弱。补充相关研究的两组剂量配对也暴露了不利的载流子迁移结果。所有结果保留。

## 2. 数据、输入与成本各是什么

能量监督：269条相对电荷交叉能量标签、15种具有多电荷标签的元素。目录中每母类提供一条参考能量；全部数据库共873个状态。Si屏蔽29条、保留844条，Sn屏蔽23条、保留850条。此任务是有大量已知物理输入的条件能量补全，不是从零生成未知掺杂元素的结构与能量。

真实载流子：29条实际剂量—Hall记录，5个DOI来源组，仅Si/Sn。来源等权MAE以dex报告：先在每来源内求绝对对数误差均值，再对来源求均值。1dex是10倍比值；这些MAE不是最大误差保证或置信区间。

0/1/2锚点是每来源使用0/1/2个Hall读数。近重复剂量另作guard、不计查询，查询数为29/23/18；固定18个共同查询点单独汇报。所有29条均缺少可直接使用的氧分压，统一1e-5atm只是未测有效氧情景；11条700°C为文献700—900°C范围下界代理。当前有效冻结温度不是实测生长温度。

条件迁移率：52组配对、11来源。完整剂量→迁移率则只有明确身份配对的5个样品、2来源；不能把52组条件任务冒充完整链验证。

新核对研究：Azizie2023有12组Hall配对，但仅e/k两组有明确同样品SIMS与Hall匹配，化学输入来自正文约2e18/3e19cm^-3的圆整数值。它与旧Cornell/AFRL来源相关，且原论文已进入历史原始语料，本轮是纠正配对后的相关研究留出测试，不是未接触的新文献或独立实验室盲测。

## 3. 第三阶段：局部性质敏感训练及其失败信号

从可微求解器得到训练元素的J，构造完整或仅对角的M=J^T W J，加入能量监督和RBF复杂度惩罚。所有特征标准化、核、物理度量与内层选参都重新排除相应元素。对缺席杂质直接删除其状态，避免有限负化学势下残留的未知能量影响训练。

数值门槛：与独立加权核岭回归预测差1.58e-14eV；与增广最小二乘系数差3.28e-11；四个高灵敏度能量导数的最大缩放差分误差1.01e-6。它们说明这组实现与推导一致，不是材料准确性的证据。

虚拟设计：每元素12个训练设计工况，另6个温度/浓度组合验证。15元素的90个验证工况中，完整能量参照只有9元素、47点满足电子Hall与检测域；Ca/Co/Fe/Mg/Pt/Zn在此设计下零点适用，不给它们虚构n型Hall准确度。

@COVERAGE@

表中的巨大误差包括模型离开单电子/检测域后的数学log(n/rH)延拓，不能作为可用Hall预测的改善百分比。普通核、对角、完整度量分别只有37、34、31点物理适用，不能只挑共同成功子集后宣称普遍更优。

事后Taylor诊断发现：普通核平均绝对余项5.53dex，局部对角3.76dex，局部完整6.94dex。它支持“当前大能量误差下，一阶局部目标不足”的判断；不证明全部问题都由Taylor误差引起。该诊断在拟合后进行，未用于改动已冻结选择。

## 4. 第四阶段：完整非线性损失与公平嵌套选择

保留相同18维特征与能量预算，以能量核为初值，在每一步重新求完整Sandbox响应，利用解析IFT梯度作阻尼Gauss–Newton提议，只有通过完整损失Armijo下降检查才接受。最多6步、每步最多8次回溯。完整理论及每个符号的说明见后附理论说明。

固定配置的诊断试验：Si响应MAE从0.988520降至0.120989dex，Sn从0.039143升至0.042716dex。这不是嵌套选参结果，不能作为主要优势或只报告Si。

![固定配置诊断试验：不能替代嵌套选择](figures/nonlinear_pilot.png)

随后对Si和Sn分别排除整个目标元素，剩余14元素内留一验证；正则强度3档、核宽2档，共6候选。普通能量核也使用同6候选和相同训练内性质选择准则。另有“非线性或普通核”12候选门控，单列为辅助，不冒充相同6候选比较。

@NONLINEAR@

在规范选参后，Si/Sn非线性响应并未优于普通核；两者均选alpha=1、length=2。非线性模型能量误差还更大。固定配置的大幅改善说明该配置能够降低响应失配，不足以证明方法在公平调参后更好。

168个内层拟合共接受904步，没有接受后的完整损失上升；保留163次失败数值试点。145个拟合达到六步预算，23个因有限回溯停止。因此不能称为全部优化充分收敛。内层累计282000次请求状态求解、1648.93CPU拟合秒；这是各进程耗时之和，不是并行墙钟时间。

相同的是能量标签预算、6候选网格及允许的训练上限。非线性方法实际使用更多根求解，不能声称等运行成本。无证据支持继续花GPU时间必然改善结果；本轮小核和根求解使用CPU。

## 5. 最终需要看的仍是真实掺杂性质误差

@PROPERTY@

本表全部查询点通过数值与已声明电子Hall域检查；对应0/1/2锚点共29/23/18查询。完整能量参照拥有全部873状态能量，是不同信息预算的参照，不是同预算ML对手。

非线性与普通核的来源配对比较中，各锚点预算均只有1/5来源略好、4/5来源更差，差异多数很小。模型复杂度增加没有建立稳定增量。没有独立依据把这些微小差异写成显著性结论。

![真实载流子任务：可变查询集与固定18查询集](figures/real_property.png)

第二阶段已完成的目标导向能量揭示仍须一起看：额外提供2条已发表能量时，Ridge＋独立xi方差选点的0/1/2锚点误差为0.259106/0.228140/0.212431dex；均值＋目标选点为0.259410/0.228677/0.212683dex。复杂学习器的独立优势仍未成立。此处补充的是公开数据库中被屏蔽的标签，不是新DFT计算。

这些结果支持继续把计算预算用在对目标性质重要的输入上，但不能用“部分错误能量恰好比完整参照更贴实测”证明其微观物理更准确。过程闭合失配与误差补偿尚未排除。

## 6. 迁移率与电导率：完整链没有被条件任务替代

仍使用CT加来源等权中位数偏移作经验输运头。每个目标来源的迁移率标签排除；完整链只把预测nH送入输运头，不读取目标实测nH。5个身份核对样品的0锚点结果如下。

@TRANSPORT@

这些误差说明现有完整链迁移率尚不可靠。条件任务52组/11来源的0.366609dex不能用于声称完整链也达到此精度。派生sigma同时继承nH与muH的误差，且可能有误差相消；不是独立导电率验证。

预测接口保留原始Sandbox状态与观测校准后的结果。锚点偏移是Hall读出修正，不把它乘回状态后冒称得到了另一套守恒缺陷库存。

## 7. 新核对相关研究：保留不利迁移证据

主来源为Azizie等APL Materials2023，DOI10.1063/5.0139622。选读最终版方法、TableII和Fig2相关段落，检查图像；没有声称完整复现SI。本文的亚氧化物MBE使用含臭氧环境，压力不能直接等同于模型的平衡O2化学势。[作者公开最终版](https://schlom.mse.cornell.edu/sites/default/files/research%20pdfs/5.0139622.pdf)

训练排除旧相关来源10.1063/5.0101132，载流子训练26条/4组，输运49条/10组；新研究全部标签未加入训练。使用e的低剂量Hall作1锚点，k为查询；0锚点两者查询。2锚点没有剩余查询，不记为零误差。

化学输入固定9种圆整数值分辨率情景：e=1.5/2.0/2.5e18，k=2.5/3.0/3.5e19cm^-3。它们是输入敏感性设置，不是置信区间。不能用nH或激活率反推化学剂量，也不把其他SIMS台阶的生长率擅自套到这两个Hall样品。

@SOURCE@

![相关研究留出：固定剂量情景范围不代表置信区间](figures/related_study.png)

非线性学习仍未修复跨研究的nH误差。低剂量锚点不能矫正高剂量轨迹形状；主要模型1锚点甚至更差。最简单的nH等于化学剂量对照在这两个主输入上更强。只有2个配对且输入为圆整数值，不能由此断言所有来源都应采用单位激活模型。

同研究12组条件迁移率MAE为0.050414dex，明显优于完整链nH表现。这意味着输运头在本研究条件下较相容，不能替底层剂量到载流子映射背书。压力、生长非平衡、冻结初态和遗漏缺陷都是候选失配来源，现有数据没有唯一确定原因。

## 8. 可调用交付与适用范围

本地模型包predictor_bundle_v2实际包含：Si/Sn留元素能量学习器、对应推断目录、全部29条旧队列训练的最终过程校准、全部52组旧队列训练的经验输运头，以及训练内选择的0/1/2锚点规则。

最终全数据训练包用于推断，不是新的留出精度证据。它支持nonlinear_kernel、ordinary_kernel、ridge和full_library四种模式；后者明确为完整信息对照。模型包是第三方数据的私有派生物，不直接公开发布。

入口为scripts/trainable_sandbox_20261017/predict_properties_v2.py。例如指定--dopant Si --incorporated-cm3 1e18 --method nonlinear_kernel即可运行；可附带0/1/2个同条件、不同化学剂量的Hall锚点JSON。接口输出所有单位、氧情景、信息预算、训练范围、原始守恒状态、观测偏移、有效过程参数与适用限制。

已核验8次保存拟合回放、掺入量守恒、电导率恒等式、锚点公式、CLI JSON输出及不支持元素/非法剂量/超预算输入拒绝。初版还校验了本应未知的能量，虽未把它们送进预测，仍是不必要的输入要求；保留初版，以v2改为只校验已知能量。将Si的29条、Sn的23条隐藏能量全部置NaN后，六次ML完整链结果不变；完整信息参照则拒绝缺失能量。此处不以集成检查次数充当材料样品数。稀土元素、绝对寿命和器件PDR没有相应依据，不返回编造的排序或置信区间。

## 9. 理论怎样根据失败继续优化

本轮已经完成一条具体的理论修正闭环：普通能量分数不能保证性质 → 引入局部性质度量 → 检验完整响应及物理覆盖 → 发现Taylor余项过大 → 实现完整非线性IFT训练 → 在公平调参和真实材料上发现增量仍不成立。

现在不能再把“优化器没有接到物理梯度”当作全部原因。下一项值得固定协议检验的输入问题是：18维表示把不同局部环境压成同样特征，而已供应的母类参考态信息尚未进入学习。参考能量只能以经过原子参考一致性证明的表示加入，不能直接添加未经对齐的绝对形成能或借用被留出电荷标签。

并行的物理问题是：共享平衡冻结闭合不能表征每种真实生长。新研究的高剂量偏离已构成具体检验对象，但看过这些标签后设计的改进只能称开发，不能再把同两个点叫盲测。后续需把真正的输入改进与能量学习增益分开消融，且保留单位激活、常数校准、均值补能和全能量参照。

这两项是继续研发的方向，不是本报告已经完成的模型能力。当前目标不会因为“零空间解释了失败”而被改成纯解释性课题，也不会在证据不足时承诺ML增益、大创新或录用。

## 10. 审计与重现

全部11760个第三阶段内层成员记录和168个第四阶段拟合经过排除核对；没有记录到外层元素、内层留出元素或目标材料来源进入对应训练。904个接受更新没有损失上升。保护清单85条、第一阶段847文件、第二阶段1157文件哈希均未变。

第三阶段报告所有11个能量训练变体和12个真实性质变体；第四阶段保留两种选择规则及七个相同控制。没有新增DFT、MD或物理样品。分析绘图的首轮列名错误日志保留；修复只影响图生成，不重训模型或改动预测值。

复现顺序：phase3_metric.py → phase3_property.py → phase3_diagnose.py；phase4_nonlinear_pilot.py的gates/pilot → phase4_nested.py → phase4_property.py → phase4_transport.py → phase4_new_source_predict.py；build_predictor.py → verify_predictor.py → build_predictor_v2.py → verify_predictor_v2.py；phase34_analyze.py → phase34_report.py。固定ga2o3环境、PYTHONPATH=src和CPU线程1。所有具体参数以各目录protocol.json为准，已有检查点可复用，不必重复训练。

## 附录：已实现方法的完整理论与符号

@THEORY@
"""
    replacements={"@COVERAGE@":coverage_table,"@NONLINEAR@":nonlinear_table,"@PROPERTY@":property_table,
        "@TRANSPORT@":transport_table,"@SOURCE@":source_table,
        "@THEORY@":theory.split("\n",1)[1]}
    for key,value in replacements.items():text=text.replace(key,value)
    REPORT.write_text(text.strip()+"\n")
    print(REPORT)

if __name__=="__main__":main()
