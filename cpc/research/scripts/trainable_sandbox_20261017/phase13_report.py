"""Full prediction evidence for the shared donor-mixture checkpoint."""
from pathlib import Path
import sys,json,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase13_refinement as ref

OUT=ref.ORIGINAL;FINAL=ref.OUT/"validation"
LABELS={"original_RT":"原目录RT物理","original_joint":"原目录＋温度补充训练",
    "localized_fixed_joint":"永久局域＋固定绑定能","binding_screen_joint":"原共享绑定能E(C)",
    "fraction_constant_RT":"常数局域比例，RT","fraction_dose_RT":"共享局域比例，RT",
    "fraction_constant_joint":"常数局域比例，联合","fraction_dose_joint":"共享局域比例，联合",
    "fraction_binding_joint":"共享比例＋学习常数绑定能","output_linear_RT":"同容量输出关系，RT",
    "output_linear_joint":"同容量输出关系，联合","direct_ridge_RT":"直接Ridge，RT",
    "direct_ridge_joint":"直接Ridge，联合","dose_line_RT":"剂量直线",
    "activation_prior_RT":"训练来源中位激活比","identity_unbounded":"n=C＋无收缩锚点偏移",
    "identity_H_unbounded":"n=C+已报H＋无收缩偏移"}


def table(frame,scope=None,value="MAE"):
    d=frame if scope is None else frame[frame.scope.eq(scope)]
    p=d.pivot(index="method",columns="budget",values=value)
    lines=["| 方法 | 0锚点 | 1锚点 | 2锚点 |","| --- | --- | --- | --- |"]
    for m,label in LABELS.items():lines.append("| "+label+" | "+" | ".join(f"{p.loc[m,b]:.6f}" for b in (0,1,2))+" |")
    return "\n".join(lines)


def figures(primary,temperature):
    folder=OUT/"figures";folder.mkdir(exist_ok=True)
    methods={"original_RT":"Original RT", "original_joint":"Original joint", "fraction_dose_joint":"Learned fraction",
        "fraction_binding_joint":"Fraction + binding", "binding_screen_joint":"Learned E(C)","output_linear_joint":"Matched output"}
    fig,axs=plt.subplots(1,3,figsize=(13.2,3.8))
    for ax,frame,scope,title in zip(axs,(primary,primary,temperature),("all_query","common18","common40_125"),
        ("RT: 29/23/18 queries","RT: fixed common18","Temperature: same 7 specimens")):
        for m,label in methods.items():
            d=frame[frame.method.eq(m)&frame.scope.eq(scope)].sort_values("budget")
            ax.plot(d.budget,d.MAE,"o-",label=label,lw=1,ms=4)
        ax.set(xlabel="Allowed anchors",ylabel="Source-equal MAE (dex)",xticks=[0,1,2],title=title);ax.grid(alpha=.2)
    axs[0].legend(fontsize=6);fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("prediction_summary."+ext),dpi=190)
    plt.close(fig)
    b=read_result_csv(FINAL/"primary_by_source.csv");x=np.arange(5)
    selected=("original_RT","fraction_binding_joint")
    families=sorted(b.family.unique());fig,ax=plt.subplots(figsize=(9.1,3.6))
    for i,m in enumerate(selected):
        d=b[b.method.eq(m)&b.budget.eq(0)&b.scope.eq("all_query")].set_index("family")
        ax.bar(x+(i-.5)*.32,[d.loc[f,"MAE"] for f in families],.32,label=methods[m])
    ax.set(xticks=x,xticklabels=["TEOS","Baldini Sn","Xiang Si","Mauze Sn","McCandless Si"],ylabel="MAE (dex)",
        title="Zero anchors: 4 sources improve slightly; Xiang worsens")
    ax.grid(axis="y",alpha=.2);ax.legend(fontsize=7);fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("zero_anchor_sources."+ext),dpi=190)
    plt.close(fig)
    bank=ref.load_composite();choices=json.loads((ref.OUT/"methods/fraction_binding_joint/selections.json").read_text())
    fig,ax=plt.subplots(figsize=(8.6,3.6));c=np.linspace(15.8,20.2,250)
    for choice in choices:
        if choice["budget"]:continue
        f=bank[choice["fit_key"]];names=f["parameter_names"];w=f["parameters"]
        a=w[names.index("localized_logit")];b=w[names.index("localized_slope")]
        ax.plot(c,expit(a-b*(c-18)),label=choice["held"],lw=1.1)
    ax.set(xlabel=r"$\log_{10}(C/\mathrm{cm}^{-3})$",ylabel=r"Learned conditional fraction $\ell(C)$",
        title="Five source-excluded fits: curves are model states, not measured fractions",ylim=[-.02,1.02])
    ax.grid(alpha=.2);ax.legend(fontsize=6.5);fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(folder/("learned_fraction."+ext),dpi=190)
    plt.close(fig)


def main():
    a=json.loads((FINAL/"contract_audit.json").read_text());assert a["passed"]
    runtime=json.loads((OUT/"research_runtime/verification.json").read_text());assert runtime["passed"]
    primary=read_result_csv(FINAL/"primary_summary.csv");temperature=read_result_csv(FINAL/"temperature_summary.csv")
    chains=read_result_csv(OUT/"research_runtime/full_chain_summary.csv");figures(primary,temperature)
    changes=[];choice_changes=0
    for method in LABELS:
        old=json.loads((OUT/"methods"/method/"selections.json").read_text())
        new=json.loads((ref.OUT/"methods"/method/"selections.json").read_text())
        for x,y in zip(old,new):
            choice_changes+=x["selected"]["lam"]!=y["selected"]["lam"] or x["selected"]["config"]!=y["selected"]["config"]
        x=read_result_csv(OUT/"methods"/method/"primary_predictions.csv")
        y=read_result_csv(ref.OUT/"methods"/method/"primary_predictions.csv")
        changes.extend(abs(x.predicted_log_nH-y.predicted_log_nH))
    energy.write(OUT/"selection_sensitivity.json",dict(passed=True,changed_outer_choices=int(choice_changes),
        changed_prediction_rows=int(np.sum(np.array(changes)>1e-10)),maximum_prediction_change=float(max(changes)),
        criterion="uniform all-failed-start refinement,not testerrorselection"))
    theory=(ROOT/"docs/ML_SANDBOX_DONOR_MIXTURE_THEORY_20261017.md").read_text().split("## 1.",1)[1]
    parts=["# ML＋Sandbox：共享施主统计、掺杂预测与训练验证\n",
        "本阶段推进可训练ML＋Sandbox预测掺杂性质，goal保持active。载流子是首个主终点；完整迁移率、电导率及有依据的新掺杂排序继续保留。理论、数值门槛和机制分析用于改进预测，不能单独作为目标完成证据。GPU1授权有效，本轮使用CPU4，没有新增DFT、MD、样品或仪器实验。\n",
        "## 1. 新方法实际增加了什么\n",
        "实现了共享局域/延展施主子群模型，从同一条件自由能产生高温原子库存、冷态电荷占据和完整隐式梯度。它精确包含两个旧端点：局域比例ell=0恢复原仅+1态目录，ell=1恢复永久+1/0局域施主。学习关系进入物理内部并随每个新化学浓度调用，不是只在性质输出上加一个回归器。\n",
        r"\["+"\n"+r"\ell(C)=\operatorname{sigmoid}\!\left[a-b\log_{10}\!\left(\frac{C}{10^{18}\,\mathrm{cm}^{-3}}\right)\right],\quad b\ge0."+"\n"+r"\]"+"\n",
        "C是给定实际化学浓度；a为参考浓度处logit，b为每dex的logit斜率；它们是跨条件训练和保存的共享权重。比例受固定输入条件约束，不称作已测谱权重、真实杂质带人口或已验证Mott转变。沿用原19元素能量输入目录，求解按实际供应排除缺失元素，保留Si/Sn复合体；C没有目录能量，未编造C缺陷。\n",
        "403项自由能/端点/守恒/同温/雅可比/完整IFT检查通过，最大原始导数检查6.24e-7；实际训练映射及来源坐标另有339项检查通过，最大1.15e-8。它们证明所声明模型的数学和实现一致性，不自动证明材料预测增益。\n",
        "## 2. 原预测任务、输入与公平对照\n",
        "原29片Si/Sn、5DOI来源，0/1/2每来源样品锚点，29/23/18查询和共同18查询保持。外层排除整个DOI；内层在其余来源选择Ridge强度与锚点输出收缩。MAE先在来源内平均再来源等权，单位dex。五DOI不等于五个独立实验室。整队列已被反复查看，是回顾性研发，不是盲测或预注册。\n",
        "所有17方法从相同已审计输入出发。物理模型使用实际Si/Sn、已报H、显式材料参数、条件氧压1e-5atm；29条没有实测氧压，部分生长温度为来源代理。物理比例只用浓度；直接Ridge可用相同队列中已有工艺/缺失标志，不能将未使用的特征说成物理层已学习。\n",
        "比较包含原物理、永久局域、原浓度相关绑定能E(C)、常数比例、浓度相关比例、额外学习常数绑定能及同容量线性输出关系。RT方法只用室温；联合方法以来源等权RT平方误差加0.1倍温度曲线损失训练。原E(C)有E0/s两个共享参数；新比例模型固定绑定能或额外学习一个常数E0。两者包含不同统计关系，不把全部差异都归于同一网络结构。\n",
        "15个3/4来源训练上下文，共405个基础记录，其中165个为物理拟合、330个优化起点；其余240个为直接或简单经验记录。所有新增物理模型均两个声明起点、相同120评估上限及相同停止阈值，直接模型记录其较小成本。候选最多42并不表示运行时间相同。完整成本、训练权重、逐点输出和所有起点保存。\n",
        "本阶段的来源适应为未截断输出偏移，收缩网格0、.001、.01、.1、1、10、无穷大，训练来源内选择；剂量恒等式强对照固定无收缩。第12阶段常数库存内部适应的1锚点0.229552、常数库存输出的2锚点0.250297继续保留，是已知更强的对应预算对照。新统计层的内部库存适应尚未重跑；不同基础模型与优化上限不能混称单次结构消融。\n",
        "## 3. 室温主任务：全部方法和预算\n",
        "下表列出全部17方法，指标为各预算查询上的来源等权MAE，单位dex；不存在只报告最好预算的筛选。\n",table(primary,"all_query"),"\n",
        "新共享比例＋常数绑定能零锚点为0.263339dex，原目录0.266289；绝对差仅0.002951dex，约1.1%。同联合训练原目录为0.268902。1/2锚点新模型为0.264685/0.284972，均劣于原目录0.251591/0.257810。纯RT比例学习也没有超过原目录。因此不能宣称跨预算稳定优势或普遍ML收益。\n",
        "## 4. 固定查询与逐来源表现\n",table(primary,"common18"),"\n",
        "![原室温、固定查询和温度补充的结构对照；完整17方法见表格](figures/prediction_summary.png)\n",
        "零锚点新比例＋绑定能在TEOS、Baldini、Mauze和McCandless四来源比原目录略好，但Xiang由0.072680恶化至0.159667。主要来源误差仍在Mauze。四个来源计数不能掩盖一个来源明显变差以及总体平均增量很小，也不能据此声称统计显著。\n",
        "![零锚点逐来源核验；改善和恶化同时保留](figures/zero_anchor_sources.png)\n",
        "## 5. 温度补充揭示的不足\n",
        "温度仍是原7片Si/2DOI、142个n读数，没有新增独立样品。RT选出的收缩直接迁移到每片曲线的RT/近150K锚点，guard和共同40—125K查询保持；不同于此前阶段10温度专门选参。下表共同80查询先按样品再来源等权，不能与其他协议的分数混成结构净增量。\n",table(temperature,"common40_125"),"\n",
        "已有E(C)模型共同温区0锚点为0.308611，新比例＋常数绑定能为0.635130，固定比例/固定能量为0.853638。新模型修复嵌套端点，并未保留浓度相关绑定能所捕获的全部温度信号。不能因新统计层更完整就宣称真实数据更好。\n",
        "![五个来源排除拟合的共享比例；它们是模型隐状态，不是测得的局域比例或置信带](figures/learned_fraction.png)\n",
        "一个直接后续假设是保留已有E(C)关系，再检验它与新比例的联合版本，并用同容量输出修正作对照；区分能量随浓度变化与局域状态适用比例的作用。该组合本轮未训练。若两者只产生参数补偿而不增加源外精度，也须完整报告，不能继续凭复杂度判断创新。\n",
        "## 6. 收敛复核与错误记录\n",
        "初版330起点有2个达到上限，来自同一四来源fraction_dose_RT拟合；该记录被三个预算选中。按预先声明的数值风险标准，所有有失败起点的拟合统一由120提高至360；本次只有这一个。重跑两个原始起点及原最佳点，共新增3个起点，均按停止标志收敛。训练目标0.04360972617降至0.04360937441，不按测试分数挑解。\n",
        f"复合库保留405条上下文，替换一个拟合后当前记录包含331个起点；实际计算累计330+3=333。最终外层选中未收敛者0，全部来源配置/收缩选择改变{choice_changes}个，结果变化很小。优化器停止标志不等于全局最优或微观参数唯一。V1全部保留。\n",
        "初版复核驱动在写完有效数值拟合后，因元数据字典重复key而终止；原代码、协议和失败日志保留。独立finish包装只补全执行元数据并重做全部选择，没有改动或重拟合已保存数值。不要把这一程序错误混成物理失败，也不能把原失败过程记成success。\n",
        "独立审计重建来源排除、权重、所有目标与7620次内层评分，核验1479行室温和7599行温度保存记录。最大训练目标差3.92e-14，经验模型重建差3.44e-12；当前主查询无缺失。经验输出偏移不重求电荷状态，validation表明确区分原物理状态和校准后的n，V1数值保持。\n",
        "## 7. 独立预测复用与完整性质链\n",
        f"11类物理/共享关系模型、5排除来源、3预算导出165个研究模型包。新进程禁止读取性质表、拟合记录和验证CSV，只输入物理变量、保存权重与授权锚点。{runtime['carrier_replay_rows']}行载流子复现最大{runtime['maximum_carrier_error']:.3g}dex，电荷预算最大{runtime['maximum_charge_budget_error']:.3g}；121行配对mu/sigma复现最大{runtime['maximum_transport_error']:.3g}。这是学习关系可复用的证据，不是新材料验证。\n",
        "同一源外经验CT头还给全部17方法产生187行完整链预测，仍只涉及5条明确化学/Hall/mu配对、2DOI；各预算5/4/2查询。头参数复用已核对的52条件对/11引用组训练、排除目标DOI的结果。给定实测n训练mu的条件任务和从化学输入预测mu的完整任务继续分开；没有补造Baldini逐样品mu。全部方法完整链mu MAE如下：\n",table(chains,value="mu_MAE"),"\n",
        "sigma=e nH muH由相同Hall量派生，没有独立sigma标签；分量误差可抵消。当前固定CT头的0锚点误差下界仍为1.234605dex，只改进n不能突破其输出范围。这里的经验头不验证Nq平方到微观散射，更不验证新元素。\n",
        "## 8. 理论与文献边界及继续方向\n",
        "[Neal2018](https://doi.org/10.1063/1.5034474)的电荷/含Hall因子输运方法和浓度讨论已读相关正文；[Rajapitamahuni2020](https://arxiv.org/abs/2010.00193)的高场双通道方法/讨论也已核对。两者支持检查统计与观测约定，不证明本模型正确或唯一识别缺陷。后者是另一个Si薄膜，不是本轮新增化学配对训练数据。\n",
        "n大于p仅排除空穴主导，不证明一个电子导电通道；当前模型没有杂质带导电。a/b、冻结温度、背景及绑定能均为条件有效参数，不能从拟合解命名唯一微观机制。下一步先检验保留E(C)的联合统计关系，输运分支再核对SI系数、FD屏蔽、能量平均及Hall定义。具体推导与待验证要求见ML_SANDBOX_TRANSPORT_UNIT_DERIVATION_20261017.md，尚无新散射精度结果。\n",
        "原29/5主终点不变；机制/数值检查和阶段报告都不足以结束目标。实用完整输运、独立来源验证和有依据新掺杂排序仍未完成。代码、输入/模型哈希、V1/V2、所有失败及独立预测包均保存，goal保持active。\n",
        "## 9. 训练前写定的完整物理和数学论证\n",
        re.sub(r"(?m)^## ([1-7])\.",r"### 9.\1","## 1."+theory)]
    report=OUT/"ML与Sandbox_共享施主统计与预测验证.md";report.write_text("\n".join(parts))
    energy.write(OUT/"report_claims.json",dict(created_utc=energy.now(),primary_summary_sha256=energy.sha(FINAL/"primary_summary.csv"),
        methods=17,physical_models=11,fit_records=405,numerical_checks=742,actual_optimizer_starts=333,
        main0anchor_old=.26628941486615126,new_fraction_binding=.263339,
        stable_crossbudget_increment_established=False,new_combined_EC_fraction_trained=False,
        new_transport_model_trained=False,new_physical_experiments=0,goal_complete=False))
    print(report)


if __name__=="__main__":main()
