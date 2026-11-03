"""Report all fixed comparisons and their implications for predictive redesign."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import phase21_prediction as s
import phase21_affine_controls as affine
from csv_io_v2 import read_result_csv as read
OUT=s.OUT;CLAIMS=[];METHODS=s.VARIANTS+list(affine.CONFIG)


def val(rel,selector,column):
    path=OUT/rel;d=read(path)
    for k,v in selector.items():d=d[d[k].eq(v)]
    assert len(d)==1,(rel,selector)
    value=float(d[column].iloc[0]);CLAIMS.append(dict(path=str(path.relative_to(ROOT)),sha256=s.energy.sha(path),selector=selector,column=column,value=value));return value


def table(rel,methods,base,column='MAE'):
    lines=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for m in methods:
        values=[val(rel,dict(**base,method=m,budget=b),column) for b in (0,1,2)]
        lines.append('| '+m+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    return '\n'.join(lines)


def figures():
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.9));d=read(OUT/'audit/summary.csv');d=d[d.scope.eq('common18')]
    for m in ('OriginalHall','FieldFirstSafe','FieldCurveSafe','MedianActivation','DoseAffine'):
        g=d[d.method.eq(m)].sort_values('budget');axes[0].plot(g.budget,g.MAE,'o-',label=m)
    axes[0].set(xlabel='RT anchor budget',ylabel='Equal-source log MAE (dex)',title='Same18 queries;allmethods keepfullcoverage',xticks=[0,1,2]);axes[0].grid(alpha=.2);axes[0].legend(fontsize=7)
    z=read(OUT/'variants/FieldCurveSafe/predictions.csv');z=z[z.family.eq('10.1002/aelm.202300679')]
    r=s.RT[s.RT.family.eq('10.1002/aelm.202300679')].set_index('sample_id')
    for budget,label in ((0,'No source update'),(1,'One-anchor learned policy')):
        g=z[z.budget.eq(budget)].copy();g['dose']=g.sample_id.map(r.incorporated_cm3);g=g.sort_values('dose');axes[1].plot(g.dose,10**g.predicted,'s--',label=label)
    axes[1].scatter(r.incorporated_cm3,10**r.target,c='black',label='Published nH',zorder=5)
    axes[1].set(xscale='log',yscale='log',xlabel='Actual Si (cm^-3)',ylabel='Hall density (cm^-3)',title='Retained TEOS prediction failure');axes[1].legend(fontsize=7);axes[1].grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'source_results.{ext}',dpi=160)
    plt.close(fig)


def main():
    a=json.loads((OUT/'audit/contract_audit.json').read_text());assert a['passed'];figures()
    primary=table('audit/summary.csv',METHODS,dict(scope='all_query'));common=table('audit/summary.csv',
        ['OriginalHall','FieldFirstSafe','FieldAnchorSafe','FieldCurveSafe','MedianActivation','DoseAffine'],dict(scope='common18'))
    ext=table('external_leedy/summary.csv',METHODS+['goal_ridge_2','goal_mean_2','nonlinear_kernel'],
        dict(scenario='primary_graph',scope='common3',policy='trained',pressure_atm=1e-5),'nH_MAE')
    mt=[]
    for kind,label in (('RT','室温5配对/2来源'),('curve','变温54旧读数/3片Si/1来源')):
        lines=['| 方法：'+label+' | 原输运头 | 化学qA1 | 化学qA3 |','| --- | --- | --- | --- |']
        for m in ('OriginalHall','FieldCurveSafe','MedianActivation','DoseAffine'):
            v=[val('cohort_properties/summary.csv',dict(method=m,budget=0,scope='query',endpoint='muH',kind=kind),'MAE')]
            v += [val('chemical_transport_controls/old_summary.csv',dict(method=m,budget=0,scope='query',kind=kind,qA=q),'MAE') for q in (1,3)]
            lines.append('| '+m+' | '+' | '.join(f'{x:.6f}' for x in v)+' |')
        mt.append('\n'.join(lines))
    content=body(primary,common,ext,'\n\n'.join(mt),a)
    report=OUT/'ML与Sandbox_响应约束与强对照预测检验.md';report.write_text(content)
    s.energy.write(OUT/'report_claims.json',dict(numeric_table_claims=len(CLAIMS),claims=CLAIMS,report_sha256=s.energy.sha(report)))
    print('report',report,'table claims',len(CLAIMS),flush=True)


def body(primary,common,ext,mu,a):
    return r'''# ML与Sandbox：响应约束、强对照与掺杂性质预测检验

## 1. 目标与这次实际完成的研究

研究主线保持为：掺杂元素、实际掺入量和可追溯条件，经训练的ML关系及物理Sandbox，预测载流子浓度、迁移率及相容电导率。理论诊断用于决定怎样改进预测。系统goal继续active；没有新材料样品、DFT或MD；本阶段使用CPU，GPU1授权保留，GPU0不使用。

前阶段发现最低剂量锚点的响应很弱时，按该锚点定标的正则化可能无法控制高剂量查询。本阶段固定所有共享权重和global gamma，检验响应约束与状态约束，随后补齐同锚点预算的斜率/仿射直接控制，以及同预测载流子下的化学补偿输运控制。全来源模型没有新增共享优化启动。

本轮5个内部消融与9个原有控制共14变体，执行518个去重来源推断组；3个补充直接控制独立存档。全部17个新策略在原29/5与0/1/2锚点/common18上比较。17个新策略加7个旧能量预设，共24种方法可通过统一CLI调用。方法命名和接口交付不等于预测精度提升。

## 2. 物理模型和学习对象保持了什么

共享物理关系继承前阶段：在参考高温库存中，将纯X的复合体池与孤立X_Ga池的原子份额作KL倾斜，保持外来原子及O预算，Ga开放。随后固定母类库存，在观测温度重新求电荷占据，并用同一状态的自由载流子与带电散射密度计算Hall读出。有效Tf、背景及共享h都沿用已训练参数。

新来源坐标u改变内部库存分配。这里的KL是指定的统计闭合，不是新证明的真实混合熵、动力学自由能或唯一微观状态。全来源共享h仍选择为0；不能将本轮来源策略变化写成新的共享关系训练增益。旧能量学习器的额外标签预算、全部873条能量物理参照与直接模型的零能量输入分别披露。

## 3. 响应正则化的方程和局部推导

X_k是明确给出的化学/观测输入，g_k(u)=log10 nH(X_k;theta,u)，j_k=dg_k/du。theta为固定共享权重，u无量纲；g和观测y均以dex计。A为至多2个带标签锚点；P为罚所采用的输入节点集合。定义：

\[
 Q_P(u)=\frac1K\sum_{k\in P}[g_k(u)-g_k(0)]^2,\qquad
 L(u)=\sum_{a\in A}[g_a(u)-y_a]^2+\lambda Q_P(u).
\]

K为P的节点数，Q以dex平方计，lambda非负。P分别取第一锚点、全部锚点或全部声明输入。查询性质标签不在Q或锚点拟合项中。另一罚型为lambda ln(1+Q)，尺度固定为1dex。

链式法则给Q'=2 mean[(g−g0)j]，L'=2 sum_anchor[(g−y)j]+lambda Q'。隐式总导数继承高温根、库存分配、冷态电荷根和Hall矩。新求解器直接在原始u上用该导数括驻点，不再对RMS响应作嵌套反求。

局部令g_k(u)约等于g_k(0)+j_k u，r_a=y_a−g_a(0)，v=mean_P j_k²。二次罚的线性化最优解为：

\[
 u_*\simeq\frac{\sum_a j_ar_a}{\sum_a j_a^2+\lambda v}.
\]

单锚点且P只含该点时，v=j_a²，退化为r_a/[(1+lambda)j_a]，故弱锚点仍可产生大内部变化。若P覆盖查询q且lambda>0、v>0，由算术几何均值与j_q²不超过K v：

\[
 |\Delta g_q|\lesssim\frac{|j_q|}{2\sqrt{\lambda v}}|r_a|
 \le\frac{\sqrt K}{2\sqrt\lambda}|r_a|.
\]

该式只限制局部、节点内的预测响应，不限制对未知真实标签的误差，也不证明非线性全区间安全。它不能作为某个lambda一定提高MAE的定理。只用全部锚点RMS也不能改变单锚点的退化关系。

## 4. 五个内部消融与明确输入契约

| 版本 | 罚的节点 | 有限更新所检查的状态 |
| --- | --- | --- |
| FieldFirst | 第一锚点 | 所有锚点 |
| FieldFirstSafe | 第一锚点 | 所有声明节点 |
| FieldAnchorSafe | 所有锚点 | 所有声明节点 |
| FieldCurve | 所有声明节点 | 所有锚点 |
| FieldCurveSafe | 所有声明节点 | 所有声明节点 |

“Safe”仅指指定节点上的数值/电子主导/Hall下限检查，不是精度保证。所有主MAE仍要求查询完整覆盖；不更新始终保留为固定物理参照。一般数值搜索不保证全局最优。

声明设计为该来源已知的非guard实际室温输入，按完整X去重，不插值未知H/碳历史。它使用查询输入X而不使用查询性质y，属于已知输入集合条件下的适应。不能称完全未看查询输入的归纳验证，也不能由有限节点检查推论连续剂量或其他温度全域有效。

u限制−6至6，Q限制36dex平方，均是声明的数值/模型范围；没有将其包装为第一性原理物理边界。有限lambda扩为0、0.001、0.01、0.1、1、10、1e2、1e4、1e6、1e8、1e10，并保留无穷。主14版本使用相同网格、二次/log尾罚和32768名义点预算。实际最大每组每版本2673点，未靠增加不受限算力获得结果。

## 5. 补充强直接控制：允许调整剂量斜率

只比较内部形状变化与输出平移不足以证明物理独有优势。因此在主结果汇总文件尚不存在时，单独冻结RidgeSlope、RidgeAffine、DoseAffine补充协议；主14版本未改。斜率控制为1个来源参数；仿射控制为2个，容量更大但锚点标签预算相同。它们只用二次罚，不消耗缺陷能量标签。

设x=log10 C−18，C为实际掺入量；V的列为x或[1,x]，a为来源系数，校准量为V a。G=V_D^T V_D/K为声明输入上的响应度量。在可识别子空间白化后，Q=norm(z)²，损失为：

\[
 \|W_Az-r\|^2+\lambda\|z\|^2,\qquad
 W_Dz\ge13-g_D(0),\quad\|z\|\le6.
\]

W是白化后的输入矩阵，r是锚点残差。这个1/2维凸问题通过圆盘及活动边枚举求解；零罚欠定时用最小响应范数约定，不假装系数在所有输入处都可识别。独立SLSQP参考有5次停止失败，V1全部保留；用第二个独立trust-constr复核后65项通过，源算法与接受容差未改变。新增接口另过15项独立闭式/缓存检查。

## 6. 原29/5来源预测：没有稳定的新增益

外层完整排除待预测DOI；共享gamma沿用原训练内选择，来源罚只按当前外层训练来源的内部预测选。0/1/2预算主查询分别29/23/18点。表中为来源等权log10绝对误差，单位dex，全部17个策略均完整覆盖：

'''+primary+r'''

固定共同18点后再比较预算：

'''+common+r'''

所有零预算输出逐点复现旧共享模型。第一锚点与全输入响应罚都没有稳定优于原物理；新增状态检查对最终所选解几乎不产生差别，不能把它记作独立精度贡献。源内仿射控制也不普遍优于固定物理，容量更大不能保证跨来源更好。

![固定共同18点及保留的TEOS失败案例](figures/source_results.png)

FieldCurveSafe单锚点误差0.499353dex，主要不利案例为TEOS来源：训练内选择二次罚lambda=0.01，u约−0.013406，该来源查询MAE为1.369483dex。所有状态满足当前门槛，但预测比不更新更差。这是信息/模型与选择策略问题，不是靠删去无效点或数值失败造成的假象。完整候选、覆盖及逐来源结果保留。

## 7. Leedy已知开发来源：简单曲线也能达到相近成绩

该来源仍为5个公开条件，实际SIMS作输入；0.05wt非均匀、0.025wt图文冲突、3点超旧Si范围、300K比较约定及2017关联mu来源均保留。没有用新标签训练共享关系或选择超参数；该来源已用于发现问题，因此不称盲测。

24方法、五标签/输入情景共1800请求，固定不更新/无惩罚诊断340请求，氧输入敏感性360请求，合计2500全部有数值。下面固定共同3个查询，采用冻结训练策略：

'''+ext+r'''

FieldCurveSafe两锚点约1.370333dex，虽相对不更新部分改善，仍远输给简单比例。DoseAffine两锚点约0.049152dex，与前阶段固定无惩罚物理适应0.046069dex很接近；只有3个相关查询及读图误差，不能据此宣称显著差异或物理独有优势。全部24方法和两个固定诊断政策都保存在CSV，未选取事后最佳lambda作为正式算法。

## 8. 输运对照：复杂缺陷目录的mu增量也未建立

保留每个模型原本预测的nH，用相同FD/Hall输运模块加入两个固定的简单化学补偿头。D=C+H+1e14cm^-3，n为反求自由电子浓度，qA分别固定为1或3：

\[
 N_D=\max(D,n),\quad
 N_{q^2}=N_D+q_A\max(D-n,0),\quad n_H=n/r_H.
\]

N_D是该启发式规则的施主量，Nq²为电荷平方加权散射量。若n>D，明确记录假设的额外施主N_D−D，不能称真实原子分辨缺陷库存。Hall反求后使用同一205经验附加散射，两个q值都报告；未用mu标签选头、改nH或重新拟合来源。该规则也不证明Born/RTA在强散射条件成立。

同预测nH下的mu误差如下，均为零锚点、来源等权：

'''+mu+r'''

简单化学头在室温可优于完整目录头，在变温任务也可达到接近或略好的误差。因此不能将此前物理头相对原CT的差别全部归因于微观缺陷组成。旧CT代码将205上限随POP温度项缩放，低温外推会急剧增大；本次约20K的部分预测达到约10^12cm²/Vs，不能把击败这一外推当成复杂物理的独有价值。新增化学头补上了这一强对照缺口。

11018个化学头输出全部数值化；额外施主假设、原始头及所有输入/政策均保留。sigma仍由匹配Hall对派生，不是独立证据；相反符号的n和mu误差可以在sigma中抵消。低温平均误差不能替代对具体温区和输运机制的检查。

## 9. 复现、计算与证据强度

核心响应模块394项、主接口43项、仿射数值65项及接口15项、化学输运21项通过。它们不是新增材料样本或独立精度数据。初始SLSQP参考失败以及初版审计中辅助拟合哈希缺少外部参照的问题均保留；正式审计用父冻结清单的SHA校验模型，不用自我比较代替证据。

85个来源排除包在独立进程执行11696个自身温度请求；1445个室温预测与保存结果回放最大差4.34e-13dex。34个guard请求因输入不在声明设计而在构建时明确排除，原query/common目标不变；不是按误差删样本。变温nH仍为旧142读数/7片Si/2DOI，室温mu仍5配对/2DOI，变温mu仍54旧读数/3片Si/1DOI，无新变温或mu监督。

17001项协议核对通过，区别于物理门槛。主推断记录实际323930个前向点、名义422408点，包含物理与经验模型；不把这些调用数当新样本数。精确缓存仅节省实际计算，不转移名义预算。保存模型文件与原权重字节一致，原阶段1—20冻结结果保留。直接模型无能量输入，物理模型供应873条；旧能量方法的844/850加0或2条预算单独说明，不能混作相同能量标签成本。

## 10. 本轮怎样修正理论和下一步研发

本轮否定的是“更合理的输入响应罚足以自动带来稳健来源适应”这一具体假设。局部响应界仍是条件成立时的数学结论，不能被扩展为预测误差保证；节点状态有效也不能补足遗漏的工艺或输运信息。单纯继续增加模型自由度、换一个罚名或挑选有利来源不能解决这些问题。

接下来先核查TEOS负面案例的原始测量定义、材料状态与输运适用域，并用完整数据评估误差是否集中于特定未建模过程。根据这些证据再固定一个实际预测改进：处理来源/模型失配的统计推断，或有证据支持的物理读出修正。它们都必须回到实际掺入量到nH/mu/sigma的公平预测；机制诊断本身不完成目标。

函数响应正则化与学习正则化已有文献，本轮没有实现随机过程变分ELBO或搬用线性逆问题泛化定理。[Sun等原论文摘要](https://arxiv.org/abs/1903.05779)，[Alberti等原论文摘要](https://arxiv.org/abs/2106.06513)。当前贡献边界由真实预测结果、相同信息对照和公开代码界定，不以SciML名称替代实际增量。

## 11. 使用与文件入口

统一CLI为scripts/trainable_sandbox_20261017/predict_properties_v8.py，提供--list-methods或--input-json/--output，要求显式method。主21预设在delivery/predictor_bundle_v8，3个直接控制在affine_controls/predictor_bundle。少样本预测需显式source_inputs及input_index锚点；查询不提供目标性质。示例、方法范围和命令见本目录README.md。

audit/summary.csv与by_family.csv给主指标；candidate文件保留全部候选和不完整覆盖。cohort_properties、external_leedy与chemical_transport_controls保存逐点、子集及完整失败；report_claims.json追溯所有表格数字。两API与公开/私有输入的用途分别声明。该工具仅验证Si/Sn范围，稀土、新元素和绝对器件性能尚无本轮可靠验证。
'''


if __name__=='__main__':main()
