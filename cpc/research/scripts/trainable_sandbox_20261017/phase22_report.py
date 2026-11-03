"""Create a complete finite-state prediction report with traceable table values."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase22_prediction as p
s=p.s;OUT=p.OUT
CLAIMS=[];CACHE={}


def value(path,selector,column='MAE'):
    if path not in CACHE:CACHE[path]=read(path)
    d=CACHE[path]
    for k,v in selector.items():d=d[d[k].eq(v)]
    assert len(d)==1,(path,selector,len(d));x=float(d[column].iloc[0])
    CLAIMS.append(dict(path=str(path.relative_to(ROOT)),sha256=s.energy.sha(path),selector=selector,column=column,value=x))
    return f'{x:.6f}'


def table(path,methods,*,namekey='variant',extra=None,column='MAE'):
    lines=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for m in methods:
        cells=[value(path,dict({namekey:m,'budget':b},**(extra or {})),column) for b in (0,1,2)]
        lines.append('| '+m+' | '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def main():
    audit=json.loads((OUT/'audit/contract_audit.json').read_text());co=json.loads((OUT/'cohort_properties/audit.json').read_text());ext=json.loads((OUT/'external_leedy/audit.json').read_text())
    assert audit['passed'] and co['passed'] and ext['passed'] and co['numeric_requests']==6880 and ext['numeric_requests']==900
    blocks=['# ML与Sandbox：有限状态来源推断与完整性质预测检验',
'''## 1. 目标和本轮结论

主方向保持为可训练ML＋Sandbox由实际掺入量和可追溯条件预测掺杂性质。理论、概率建模和数值门槛用于改进预测，不能单独完成研究goal。当前goal继续active。

本轮冻结第20阶段所有共享theta与global gamma，只检验来源适应。新增归一化的有限状态先验、工作似然后验与可选零更新原子；同一概率处理也给物理输出和直接经验模型。主实验406去重组/10方法全部执行，来源超参数只在训练来源内选择。没有新增共享优化起点、温度/迁移率训练标签、DFT、MD或材料实验；当前使用CPU，GPU1授权保留。

结果不支持内部库存集合推断的稳定新增益。概率归一化并未解决来源分布偏移和模型失配；部分有限网格预测还具有明显表示敏感性。因此不能将本轮包装为更强预测算法或连续后验收敛证明。全部强对照和失败均保留。''',
'''## 2. 原始物理证据怎样约束改进方向

TEOS原PDF Table1及正文均给出0.144和0.242cm²/Vs；不能自动乘1000。其100nm蓝宝石异质薄层有表面Si增强、晶粒/孪晶/堆垛缺陷讨论，但不能从低迁移率唯一证明晶界或局域化。原低温McCandless数据是同质外延、1T Hall和300到20K，明确报告冻结；原TableI与正文“样品2/3>100”的编号/数值不一致，保留Table标签。[TEOS论文](https://doi.org/10.1002/aelm.202300679)，[McCandless作者PDF](https://djena.engineering.cornell.edu/papers-new/2022/mccandless2022controlled.pdf)。

按固定温区拆解940条原预测/对照与472条化学输运记录，不重拟合。原模型在60K以下普遍高估载流子，部分样品超过2dex。这与当前完全延展施主近似不能描述冻结相容，不能由此唯一确定缺陷身份。既有阶段9、13、16已测试E(C)、局域/延展混合及联合n/mu，不重复命名为新方法。本轮首先检验少量锚点下点选择不稳是否可由集合推断改善。''',
r'''## 3. 概率模型及其数学边界

u为无量纲来源坐标，theta为冻结共享参数，g_k(u)=log10 nH(X_k;theta,u)。已声明输入集合D不含查询性质标签。129节点由t在[-1,1]等距、u=6t³构造；b_i为相邻原始u中点划分的单元宽度/12，正且和为1。i为状态、k为输入，i0为零更新状态。

Q_i为所有声明输入处[g_k(u_i)−g_k(0)]²的均值，单位dex平方。V要求所有节点数值/电子主导/载流子下限有效，且Q_i≤36；无效节点及其原因保留。tau为响应尺度、rho为零原子概率：

\[
 p_i(\tau)=\frac{b_i e^{-Q_i/(2\tau^2)}1_{i\in V}}{\sum_j b_j e^{-Q_j/(2\tau^2)}1_{j\in V}},\qquad
 P_i=\rho1_{i=i_0}+(1-\rho)p_i.
\]

锚点a属于A，y_a为允许提供的300K Hall读数的log10。工作模型假设独立Gaussian残差epsilon_a，标准差sigma以dex计；它不是实测仪器误差或已验证的残差独立性：

\[
 y_a=g_a(u_i)+\epsilon_a,\quad
 L_i=(2\pi\sigma^2)^{-|A|/2}e^{-\sum_a[g_a(u_i)-y_a]^2/(2\sigma^2)},\quad
 w_i=\frac{P_i L_i}{\sum_jP_jL_j}.
\]

归一化分母严格保留，以logsumexp计算。零原子和slab通过证据相加，未比较不同测度单位的概率质量和密度高度。本轮是明确有限测度上的工作后验，不是连续u积分、真实微观参数后验或完整theta贝叶斯推断。

主点动作取潜在logn分布的加权中位数，返回实现该中位数的实际状态及其mu、sigma电导。中位数两侧概率各≤1/2时，绝对值损失次梯度包含0，故最小化该潜在分布的期望绝对误差。它不等于对加噪声测量分布的最优中位数。逐查询摘要可对应不同u，不能声称一个来源点参数同时实现所有摘要。每个查询仍满足电导率=e*nH*muH；e是正元电荷。''',
'''## 4. 固定协议与独立实现核验

原29样品/5引用来源、0/1/2样品锚点、guard及共同18查询保持。所有已知非guard输入参与响应设计，查询标签不参与来源状态更新；属于输入集合条件预测，不称完全归纳盲测。全来源部署的来源分布参数也只由旧5来源内部留出选择。

tau=.05/.15/.5/1.5dex，sigma=.05/.15/.5dex，rho=0/.5/.9/1；rho=1的tau重复合并，每非零预算实际39候选。纯slab消融固定rho=0；零锚点主输出精确参考，不用非对称先验均值改变基线。内部与匹配输出控制使用相同统计处理，所有版本保留原32768点预算上限；物理目录供应873能量输入、直接模型0，不能声称能量信息成本相同。

85项独立概率/截断Gaussian/证据门槛和50项独立部署概率/缓存门槛通过。完整审计逐候选重建后验、同状态读出与训练内选择，共341469项协议检查；最大权重差4.41e-14，主性质决策差0。它们不是新增材料样本。主实际218534、名义260006前向点包含物理与经验模型。17540个网格节点读出违反观测域被显式先验条件排除，点异常0；未按标签误差删样本。''',
'## 5. 载流子主终点：全部新方法\n\n来源等权MAE，单位dex；各预算查询29/23/18，全部700个主查询覆盖。另20个预定guard行不在声明设计中；850个非guard行均有数值。\n\n'+table(OUT/'summary.csv',p.VARIANTS,extra=dict(scope='all_query')),
'### 第21阶段全部17个冻结强对照\n\n以下保留既有同队列结果；不同来源先验形式和容量已在前报告说明。新方法不能只与较弱直接ML比较。\n\n'+table(s.OUT/'audit/summary.csv',list(s.VARIANTS)+['RidgeSlope','RidgeAffine','DoseAffine'],namekey='method',extra=dict(scope='all_query')),
'## 6. 固定共同18：预算比较\n\n'+table(OUT/'summary.csv',p.VARIANTS,extra=dict(scope='common18')),
'''内部集合/零原子在两个非零预算均未优于原物理，且两锚点明显恶化。同共享库存的输出集合没有相同幅度恶化，但也未建立稳定增量。Ridge等部分经验模型在某预算改善，不能归为内部物理关系的独有收益。既有阶段12的单锚点0.229552及两锚点0.250297等强历史结果继续保留，不因新版本删除。''',
'''## 7. 有限网格的表示敏感性

129为固定主定义；33/65是同一超参数的敏感性检查，不按查询误差择优。65与129之间18个所选来源/预算情景的最大查询变化超过.02dex，全部方法中的最大为.112633dex。内部两版各有1个情景超过门槛、最大.042971；同库存输出两版最大.009087、剂量直线.004967。不能把输入有限网格上的归一化称作连续后验已收敛。

当前内部主结果已有明显负面增量，不继续为追求有利分数任意加密或扩展超参数。若以后要将该表示用于精度主张，必须先按统一规则处理网格敏感性，并与线性输出的解析连续参考核对。失败不证明所有贝叶斯/集合方法无效，只限制当前模型、先验、有限设计和数据。''',
'''## 8. 自身温度与完整性质链

50来源排除包在两个独立进程执行6880请求，全部有数值；850室温回放最大1.42e-14dex。20guard请求明确排除，主查询不变。全部正后验支持均需在查询温度观测域内，否则保留拒绝、不重新归一化；本次没有请求拒绝。旧数据仍仅142个nT读数/7片Si/2DOI、5个muRT配对/2DOI、54个muT读数/3片Si/1DOI；没有新增标签。sigma仅由Hall对派生。

下表均固定共同查询，避免样本组成变化被误读为适应收益。温度读数并不是独立样品。''']
    for title,endpoint,kind,count in [('变温nH','nH','curve','66读数/3旧片/2DOI'),('室温muH','muH','RT','2配对/2DOI'),
        ('变温muH','muH','curve','24读数/1旧片/1DOI'),('派生室温电导率','sigma_derived','RT','2配对/2DOI，无独立标签')]:
        blocks.append('### '+title+'：'+count+'\n\n'+table(OUT/'cohort_properties/summary.csv',p.VARIANTS,namekey='method',extra=dict(endpoint=endpoint,kind=kind,scope='common_query')))
    blocks.extend(['''内部muT的小幅变化来自一个旧样品；不能宣称新散射模型或跨研究泛化。经验模型仍继承低温会强烈上升的旧CT头；完整目录相对该头的优势不能全部归因于微观缺陷组成。第21阶段同nH的化学补偿输运强对照继续保留。本轮没有通过后验重新拟合mu头或选择电荷假设。''',
        '## 9. Leedy已知开发来源\n\n五标签/输入情景750请求，加报告氧输入敏感性150，共900全部有数值。0.025图文冲突、0.05非均匀、三高剂量超范围、300K比较约定和关联2017来源均保留。下表固定共同3查询，来源策略来自旧训练来源，未用Leedy查询选择。\n\n'+table(OUT/'external_leedy/summary.csv',p.VARIANTS,namekey='method',extra=dict(scope='common3',scenario='primary_graph',pressure_atm=1e-5),column='nH_MAE'),
'''内部两锚点约1.3865dex，仍远输给简单比例/单位剂量校准。原输出零原子两锚点部分改善，也未达到同来源简单比例约.05dex水平。不能将同一已知来源的输入情景或已看过的读数称盲测。新方法没有解决该来源的跨分布失配。''',
'''## 10. 下一步实际研发方向

本轮结果不支持“只将MAP点换成状态集合就能修复当前预测”。固定少量来源下，表示、共享物理近似与未报告工艺仍会限制泛化。下一步先补审计可追溯的实际化学/工艺—Hall配对和受控条件变化，再决定共享内部学习关系；不继续盲目堆来源适应复杂度。理论诊断本身不完成原预测目标。

已从作者链接取得Azizie2023的12页原PDF，并解析其原数据DOI元数据/公开仓库，取得39MB生长日志ZIP及24条目录项，尚未读取日志内容或建立样品配对。主文局部只确认两个明确同样品SIMS/Hall条件；不能将整幅迁移率图自动当化学配对或独立实验室。Ge2021目前仅有正式元数据及作者发表目录，出版社PDF403，未读全文、无新增标签。[Azizie作者论文](https://djena.engineering.cornell.edu/papers-new/2023/azizie2023silicon.pdf)，[原数据仓库](https://data.paradim.org/doi/zsda-pa72/)，[Ge论文DOI](https://doi.org/10.1063/5.0059657)。

这些新来源必须先核查phase、实际化学浓度与Hall配对、工艺校准是否使用目标Hall造成循环、跨文献样品复用，以及原KROGER所含元素/状态。旧29/5查询和全部失败保持；两点来源不能虚构第三个独立查询。输入审计后再冻结新的共享关系学习/外推实验。''',
'''## 11. 复现和文献边界

source_ensemble.py实现有限概率模型；ensemble_predictor.py是v9保存权重入口，predict_properties_v9.py为显式method CLI。10预设/6参数文件全部从旧模型字节复制；示例锚点仅演示。fresh worker禁止读取性质表、fit、episode、查询参考表；API、源外回放和科学预测结果分别报告。完整目录见README，所有表格数字有report_claims.json来源/筛选/SHA。

[Yao等原论文](https://sites.stat.columbia.edu/gelman/research/published/stacking.pdf)第1—3节说明错设模型中的组合限制；本轮未实现其stacking或PSIS。[Masegosa原论文摘要](https://arxiv.org/abs/1912.08335)仅作为错设局限背景，未使用PAC-Bayes定理。归一化/中位数属于已有工具；本研究是否有预测创新需由真实数据与匹配消融决定，不能用算法名称替代增量。

初次温度种类名T/curve匹配错误、失败日志和V1部分文件均保留；修正只改变统计计数，不改变预测或标签。初版报告公式字符串被Python控制字符转义，已改为原始字符串；原报告/PDF/代码保留，数值结果不变，不能只凭编译无警告认定公式正确。阶段1—21冻结结果保留。整体goal仍active，没有新增实体实验、DFT/MD或稀土定量验证，不预设有利精度或发表结论。'''])
    report=OUT/'ML与Sandbox_有限状态学习与预测检验.md';report.write_text('\n\n'.join(blocks)+'\n')
    s.energy.write(OUT/'report_claims.json',dict(numeric_table_claims=len(CLAIMS),report_sha256=s.energy.sha(report),claims=CLAIMS))
    print('report',report,'claims',len(CLAIMS),flush=True)


if __name__=='__main__':main()
