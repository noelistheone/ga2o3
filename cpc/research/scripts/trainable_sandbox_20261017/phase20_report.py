"""Write the bounded prediction checkpoint, with source-linked numeric tables."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv as read
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase20_source_inventory';CLAIMS=[]
METHODS=('OriginalHall','OriginalBudget','RelativeInventory','RelativeInventoryReadout','SharedReadout',
    'DirectRidge','MedianActivation','DoseLine','UnitDose','UnitDoseH')


def val(rel,selector,column):
    path=OUT/rel;d=read(path)
    for k,v in selector.items():d=d[d[k].eq(v)]
    assert len(d)==1,(rel,selector)
    value=float(d[column].iloc[0]);CLAIMS.append(dict(path=str(path.relative_to(ROOT)),sha256=energy.sha(path),
        selector=selector,column=column,value=value));return value


def budgettable(rel,methods,base,column='MAE'):
    lines=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for method in methods:
        values=[val(rel,dict(**base,method=method,budget=b),column) for b in (0,1,2)]
        lines.append('| '+method+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    return '\n'.join(lines)


def figures():
    folder=OUT/'figures';folder.mkdir(exist_ok=True);d=read(OUT/'external_leedy/predictions.csv')
    d=d[d.scenario.eq('primary_graph')&d.pressure_atm.eq(1e-5)]
    obs=d[d.method.eq('RelativeInventory')&d.budget.eq(0)&d.policy.eq('trained')].sort_values('incorporated_cm3')
    fig,ax=plt.subplots(figsize=(7,4.3));ax.scatter(obs.incorporated_cm3,10**obs.observed_log_nH,c='black',label='Published Hall readouts',zorder=5)
    for method,policy,b,label in [('RelativeInventory','trained',2,'Frozen learned policy (no update)'),
            ('RelativeInventory','unpenalized',2,'Internal adaptation; diagnostic'),
            ('RelativeInventoryReadout','unpenalized',2,'Output adaptation; diagnostic'),
            ('MedianActivation','trained',0,'Median activation; zero anchors')]:
        g=d[d.method.eq(method)&d.policy.eq(policy)&d.budget.eq(b)].sort_values('incorporated_cm3')
        ax.plot(g.incorporated_cm3,10**g.predicted_log_nH,'s--',ms=4,label=label)
    ax.set(xscale='log',yscale='log',xlabel='Actual incorporated Si (cm^-3)',ylabel='Hall density (cm^-3)',title='Known-source development test; all five conditions shown')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'conditional_source_prediction.{ext}',dpi=160)
    plt.close(fig)
    p=read(OUT/'source_adaptation_v2/anchor_sensitivity.csv');p=p[p.budget.eq(2)]
    fig,ax=plt.subplots(figsize=(6.5,3.8));ax.scatter(abs(p.first_jacobian),p.relative_information,s=25)
    ax.set(xscale='log',yscale='log',xlabel='First-anchor |d log10(nH) / du|',ylabel='Sum of anchor J^2 / first-anchor J^2',title='Local conditioning of the frozen two-anchor designs')
    ax.grid(alpha=.2);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'anchor_conditioning.{ext}',dpi=160)
    plt.close(fig)


def main():
    a=json.loads((OUT/'audit/contract_audit.json').read_text());c=json.loads((OUT/'cohort_properties/audit.json').read_text());e=json.loads((OUT/'external_leedy/audit.json').read_text())
    assert a['passed'] and c['passed'] and e['passed'];figures()
    old=budgettable('audit/summary.csv',METHODS,dict(scope='all_query'))
    common=budgettable('audit/summary.csv',METHODS[:5],dict(scope='common18'))
    manifest=json.loads((OUT/'delivery_v2/predictor_bundle_v7/manifest.json').read_text())
    external=budgettable('external_leedy/summary.csv',list(METHODS)+manifest['energy_methods'],
        dict(scenario='primary_graph',policy='trained',pressure_atm=1e-5,scope='all_query'),'nH_MAE')
    diag=['| 方法 | 训练策略1锚点 | 无惩罚1锚点 | 训练策略2锚点 | 无惩罚2锚点 |','| --- | --- | --- | --- | --- |']
    for m in ('RelativeInventory','RelativeInventoryReadout','SharedReadout','DirectRidge','MedianActivation'):
        values=[val('external_leedy/summary.csv',dict(scenario='primary_graph',scope='common3',method=m,budget=b,policy=p,pressure_atm=1e-5),'nH_MAE')
            for b,p in ((1,'trained'),(1,'unpenalized'),(2,'trained'),(2,'unpenalized'))]
        diag.append('| '+m+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    prop=['| 方法 | 变温nH | 室温muH | 变温muH | 室温派生sigma |','| --- | --- | --- | --- | --- |']
    for m in ('OriginalHall','RelativeInventory','SharedReadout','DirectRidge','MedianActivation','UnitDose'):
        values=[val('cohort_properties/summary.csv',dict(method=m,budget=0,scope='query',endpoint=ep,kind=k),'MAE')
            for ep,k in (('nH','curve'),('muH','RT'),('muH','curve'),('sigma_derived','RT'))]
        prop.append('| '+m+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    content=body(old,common,external,'\n'.join(diag),'\n'.join(prop),a,c,e)
    report=OUT/'ML与Sandbox_相对库存学习与预测适应检验.md';report.write_text(content)
    energy.write(OUT/'report_claims.json',dict(numeric_table_claims=len(CLAIMS),claims=CLAIMS,report_sha256=energy.sha(report)))
    print('report',report,'numeric table claims',len(CLAIMS),flush=True)


def body(old,common,external,diag,prop,a,c,e):
    return r'''# ML与Sandbox：相对库存学习与预测适应检验

## 1. 目标没有转向：从掺杂输入预测性质

本阶段继续训练与验证ML＋Sandbox，以掺杂元素、实际掺入量和可追溯条件为输入，以载流子为首要预测终点，再核验迁移率及相容电导率。理论、测量与可识别性用于改进预测。单独的机制解释、守恒证明或接口交付不能完成目标。系统goal保持active；本轮使用CPU，GPU1授权保留，GPU0不使用，没有新DFT、MD或物理样品。

实际交付包括：一个已训练的共享内部库存关系、同信息的输出控制、0/1/2锚点来源适应、完整强对照、版本7的17预设推断入口、来源排除的性质预测，以及已知Leedy失配来源的固定诊断。主要结果是：当前自动选择未建立稳定内部学习增益；指定条件下的内部适应具有输出偏移不具备的剂量形状修正能力，但何时可靠启用尚待改进。

## 2. 为什么学习复合体与孤立掺杂的相对份额

冻结模型544次扫描、68次零更新回放先定位失配。Leedy模型状态中约89.2%—97.3%的负补偿电荷来自含掺杂元素的复合体；仅压低本征库存不能有效恢复剂量斜率。这是模型内归因，不是实测缺陷组成。阶段17本来已有native/complex来源接口；当时实际来源验证采用输出偏移，不能把接口存在写成已经完成内部适应测试。

KROGER连接杂化泛函缺陷能量、温度依赖的半导体热力学与工艺条件，本身讨论了复合体动力学及冻结近似的限制。当前层是给定能量目录下的有效分配闭合，未补齐真实薄膜热史、振动熵或动力学。[KROGER原作者论文](https://arxiv.org/abs/2501.17373)。

## 3. 新学习层的数学结构

令高温求解器给出的母类库存为c0_j，单位cm^-3；选纯X的含Ga空位复合体集合C及孤立X_Ga集合I。k_j为该母类所含X原子个数。定义两池原子量及参考份额：

\[
 U=\sum_{j\in C}k_jc^0_j,\quad V=\sum_{j\in I}c^0_j,\quad
 W=U+V,\quad p_j^0=k_jc_j^0/W.
\]

U、V、W是原子浓度；p0是归一化份额。学习无量纲共享参数h，最小化指定的相对熵与孤立池偏好：

\[
 \min_{p\ge0,\,\sum_jp_j=1}
 \left\{\sum_jp_j\ln(p_j/p_j^0)-(\ln10)h\sum_{j\in I}p_j\right\}.
\]

对归一化乘子a求驻点：ln(p_j/p0_j)+1−ln(10)h 1_I(j)+a=0。故p_j正比于p0_j乘以exp[ln(10)h 1_I(j)]。给定所有p0正、有限h时，严格凸性保证分配解唯一；它不保证实验唯一识别h、Tf或微观库存。令ell=ln(V/U)+ln(10)h，两池份额分别为sigmoid(ell)与sigmoid(−ell)，于是：

\[
 c'_j=\begin{cases}
 (W\operatorname{sigmoid}(-\ell)/U)c_j^0,&j\in C,\\
 (W\operatorname{sigmoid}(\ell)/V)c_j^0,&j\in I,\\
 c_j^0,&\text{其他母类}.
 \end{cases}
\]

h=0逐母类回到原库存；h正负允许双向变化。两池X原子量和为W；其他外来原子及O预算不变；Ga由开放库交换。不能宣称封闭Ga守恒、真实反应路径、热力学细致平衡或自由能耗散。这里的KL是统计距离，并非从第一性原理推得的缺陷混合熵。

## 4. 训练怎样穿过物理求解器

共享训练参数为有效冻结温度坐标、log10背景浓度和h。有效Tf在600—1950K；背景在1e14—1e18cm^-3；共享h在−6至6。Tf不冒充实测生长或峰值温度；旧29条记录缺少完整热史和定量氧分压，主要使用声明的1e-5atm氧情景。

高温根R(s;theta,X)=0给库存c0；分配层得c'；固定c'在观测温度解S(x;c',theta,X)=0。theta是学习参数，X是输入，s、x是相应状态。局部根光滑且雅可比可逆时：

\[
 \frac{ds}{d\theta}=-R_s^{-1}R_\theta,\qquad
 \frac{dx}{d\theta}=-S_x^{-1}\left(S_c\frac{dc'}{d\theta}+S_\theta\right).
\]

下标为偏导矩阵，库存总导数包括高温根与份额分配两部分。电荷占据、自由载流子及带电散射密度Nq²从同一状态重算，再计算Hall因子、mu和sigma。Hall读出nH=n/rH、muH=rH mu_drift，故两者乘积给相容的电导率。输出偏移控制没有改变内部状态，需明确标为读出校准。

主要训练损失是来源等权的室温nH平方误差，加上同一Tf/背景下相对h=0的预测响应平方罚。gamma在0、0.001、0.01、0.1、1、无穷中按训练来源留一法选择；无穷复用旧物理参数与h=0。输出控制用相同3参数及可观测dex罚。实际未使用mu或变温nH监督；所有能量输入保持原873条。

262项核心门槛与59项学习门槛通过；修复后重复59项仍通过。它们检验守恒、根及解析导数，不是材料实验准确度。数值优化中1394个明确标记的电子分支延拓试探保留，最终训练状态和科学预测必须满足电子主导与Hall下限；不能把优化器success当作物理有效性。

## 5. 锚点适应、反求修复与计算预算

来源内部位移u改变库存h；匹配输出控制则直接平移log10 nH。为统一罚的单位，原协议采用最低剂量锚点响应zeta=g_a(h+u)−g_a(h)，以dex计。单锚点二次罚给zeta=r/(1+lambda)，r为锚点残差；两锚点使用真实非线性状态算子。比较二次罚和固定1dex尺度log尾罚，lambda为0、0.001、0.01、0.1、1、10、无穷。查询标签不进入适应。

这是一种条件于锚点设计的响应正则化，不是实验设计无关的微观先验。有限网格与局部求解不构成全局最优定理，也没有已校准后验区间。

V1反求把两个不同内部坐标、但完全相同的浮点观测根误判为缓存顺序错误。真实故障已复现：两个观测误差均严格为0，缓存没有下降。V2先返回精确命中，再做严格包围；公式、容差、参数、候选与预算不变。实际故障案例12个解全部通过，最大锚点反求误差约8.49e-15。V1代码、错误与结果保留；修复后最终选择预测最大变化约1.42e-13dex，不能将修复本身当作预测增益。

旧任务330个拟合记录含150次新增非线性拟合、300次启动；部署新增7个记录，仅共享输出头有2次新非线性启动。额外物理预算控制含16个上下文、160次新启动，允许更充分搜索但不增加物理自由度。合计353个记录、462次新启动，最终所选训练状态有效，没有状态求解失败。初始打印序列化失败及V1反求失败均保留。闭式控制实际花费更少，不能称严格等墙钟竞赛；固定预算与实际成本分开列出。

## 6. 原29/5来源排除预测：尚无稳定共享学习增量

训练只用旧29个室温样本、5个来源，外层完整排除待预测DOI；global gamma先在训练内零锚点任务选择，再固定gamma选择来源罚。0/1/2预算分别29/23/18个查询；guard不额外供标签。下表是来源等权log10绝对误差，单位dex，均要求完整覆盖。

'''+old+r'''

RelativeInventory的4/5个外层选择无穷gamma，全部10个非零锚点策略选择无穷lambda。全来源部署也选择共享h=0。因此上述一两个平均误差的差别不能算有效内部适应带来的收益。预算增强OriginalBudget与原物理几乎相同，原优化搜索不足不是本轮主要差别。

固定共同18点才适合比较预算变化：

'''+common+r'''

完整较早强对照仍保留在phase56_validation/canonical_comparison.csv；两条额外能量的Ridge约0.259106/0.228140/0.212431dex，简单能量均值非常接近。它们供应的能量标签、电子统计及过程闭合与本轮不同，不能混作完全相同模型消融；当前新库存层没有稳定超过这些旧结果。

## 7. 为什么有限来源罚被舍弃：完整覆盖与条件数

V2全部57个内部episode、733个候选解完成反求且没有缓存故障，包括部署补充的2个episode。但锚点反求成功不等于查询物理状态合法。全部候选及逐来源完整覆盖在audit/candidate_by_family.csv和candidate_summary.csv，未只保留最优者。

第一锚点局部导数最低约3.48e-7；两锚点总导数平方相对第一锚点平方最大约1.28e6。小的第一锚点响应，可能对应很大的内部库存变化，继而使高剂量查询变为孔主导或低于观测下限。例如二次罚lambda=10时，单锚点任务完整有效查询仅5/23，两锚点12/18；缺失主指标保留为空，不能只对余下有效点报较低MAE。

![两锚点设计的局部条件数诊断；有限采样不证明全局单调](figures/anchor_conditioning.png)

局部令g_i(u)−g_i(0)约等于j_i u，唯一锚点残差为r。当前罚给u约等于r/[(1+lambda)j_a]；因此查询响应约为(j_q/j_a)r/(1+lambda)。第一锚点很弱时，匹配其响应单位并不足以控制整条剂量曲线。该线性诊断不证明越过分支后仍适用，也不把所有失配归于唯一原因。

## 8. 已知Leedy开发来源：区分表达能力与训练策略

5个公开条件来自Leedy2018 DOI10.1063/1.5047214；4个图读数加1个正文配对，实际SIMS是化学输入。0.05wt薄膜深度不均匀，0.025wt图文冲突，3个剂量超出旧Si上限；300K是比较约定，未冒充仪器温度记录。该来源已推动模型设计，故只称开发迁移测试。相关2017前作在旧条件mu队列，另给排除该前作的49对/10来源CT头。

保持5组标签/输入情景，主要压力1e-5atm；另做报道氧分压固定情景。17预设主要情景1275个预测，加200个固定策略诊断、255个压力情景，共1730次全部有数值输出。新标签没有进入共享训练或超参数选择。下面是冻结训练策略的全部17预设，预算对应5/4/3个查询：

'''+external+r'''

默认RelativeInventory没有改善原冻结平台；各预算查询组成变化造成MAE变化。共同3点的默认误差均为约1.637792dex，不能说增加锚点损坏了未更新的模型。

预先固定两个诊断政策：不更新，以及无惩罚适应。所有10个新预设/控制都采用相同政策，不由查询误差选择优胜策略。固定共同3点的对照如下：

'''+diag+r'''

![开发来源的掺杂性质预测；两锚点适应用0.025/0.1标签，其余3点为查询](figures/conditional_source_prediction.png)

内部无惩罚两锚点约0.046069dex，输出适应约0.344359dex；中位数比例同政策约0.051399dex。这里不能宣称统计显著优于比例：只有3个查询、非独立实验室验证，且相近读图误差/输入情景仍须考虑。它证明的是本条件下内部适应能够改变剂量曲线形状，输出平移不能作相同变化；并未证明当前共享h或自动策略有增量。

同一内部适应的共同3点mu误差，1/2锚点约0.221005/0.247944dex，输出适应均约0.640891dex；电导率约0.196443/0.258594dex。后一项为配对Hall量的派生值。两个锚点使nH变好而mu变差，说明不能只用载流子成绩宣布整个性质链改善。

## 9. 变温与迁移率：在各自温度实际运行

来源排除的50个独立模型包在新进程中执行6900个请求，不读性质表、拟合表或来源episode。870个室温选中预测与原保存结果一致，最大偏差约1.63e-13dex；全部6900个输出通过当前门槛。它们是数值与集成证据，不增加物理样本数。

变温nH为旧142读数、7片Si、2个来源；室温mu仍是5个配对/2DOI；变温mu仍是旧54读数/3片Si/1DOI。所有预算只供室温样品锚点，不把150K曲线点偷换为新锚点。没有新变温或mu监督。0锚点、来源等权并在来源内先按样品平均的结果为：

'''+prop+r'''

物理同态读出比直接CT头在这些旧mu任务更合适，但当前内部学习与原物理相同，没有新学习的mu增量。Nq²、Hall因子与预测n相容，不证明Born/RTA或205经验附加散射在所有状态成立。完整all/query/common子集均保留；sigma仅在室温实际配对处计算，不拼接不同温度读数。

## 10. 创新边界与下一项可检验改进

KL分配、隐式求导、正则化与层次学习都是已有方法。不能只给传统参数拟合换名字。已有元学习明确训练跨任务共享关系以方便少样本适应；层次贝叶斯则用共享超参数约束任务参数，本研究当前没有因此自动获得后验或独立泛化证明。[Finn等原论文摘要](https://proceedings.mlr.press/v70/finn17a.html)，[Grant等第2—3节和4.1节](https://arxiv.org/pdf/1801.08930)。

保留的研究路线是：训练过的内部关系、守恒状态求解及数据适应共同预测掺杂性质，并用来源排除和性质链检验其实际增量。当前应先改变已定位失配的适应约束，检验声明剂量范围的整体响应罚及状态有效约束，保留第一锚点罚、扩展有限lambda网格、输出校准与不更新控制。只换先验尺度，先不加网络或重新训练共享权重。

下一方案的逐符号局部推导、输入范围约定及公平对照见docs/ML_SANDBOX_CURVE_RESPONSE_NEXT_20261018.md。它尚未实现，不预设能提高准确度；若单锚点只用全锚点RMS仍与旧方案相同，不能作为问题已解决的证据。

更远一步的连续层次先验/模型平均需要明确似然和偏差来源，不用已知Leedy查询挑策略。稀土和未知元素尚无当前工具的可靠验证，不能输出无依据排名。目标继续active，本检查点保留完整成败后进入下一项预测研发。

## 11. 文件与复现入口

正式工具为delivery_v2/predictor_bundle_v7，8个模型文件逐字节继承V1，仅修复反求并重验来源策略。17预设明确能量预算；旧能量方法供应Si844/Sn850条加0或2条，内部库存和完整能量参照使用873条，不是零数据未知元素发现。

从项目根目录执行scripts/trainable_sandbox_20261017/predict_properties_v7.py，参数为--input-json与--output；环境为ga2o3且PYTHONPATH=src。README.md给完整示例。预测仅需要显式化学/观测输入、可选RT锚点、保存模型及原私有KROGER输入；查询目标标签不在推断进程。

audit/contract_audit.json记录6092项协议核对，区别于物理/梯度门槛；audit/training_compute.csv给实际计算。source_adaptation保留V1，source_adaptation_v2为正式修复验证；cohort_properties和external_leedy保存逐点及所有子集结果。report_claims.json逐项定位表格数值和输入哈希；CSV统一使用csv_io_v2读取标识符。最终冻结保留旧阶段全部产物，未修改稿件。
'''


if __name__=='__main__':main()
