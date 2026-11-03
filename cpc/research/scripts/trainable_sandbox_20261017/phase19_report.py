"""Evidence audit, goal check, source transfer report and usable model guide."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv as read
import phase2_energy as energy

BASE=energy.BASE;OUT=BASE/'phase19_delivery';CLAIMS=[]


def val(rel,selector,column):
    path=BASE/rel;d=read(path)
    for k,v in selector.items():d=d[d[k].eq(v)]
    assert len(d)==1,(rel,selector)
    value=float(d[column].iloc[0]);CLAIMS.append(dict(path=str(path.relative_to(ROOT)),sha256=energy.sha(path),
        selector=selector,column=column,value=value));return value


def table(methods,scope='all5',rel='phase19_delivery/leedy2018_v2/summary.csv'):
    lines=['| 方法 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for method in methods:
        scores=[val(rel,dict(scenario='primary_graph',scope=scope,method=method,budget=b),'nH_MAE') for b in (0,1,2)]
        lines.append('| '+method+' | '+' | '.join(f'{x:.6f}' for x in scores)+' |')
    return '\n'.join(lines)


def main():
    gates=json.loads((OUT/'numerical_gates.json').read_text());runtime=json.loads((OUT/'runtime_verification.json').read_text())
    replay=json.loads((OUT/'training_api_replay/verification.json').read_text());external=json.loads((OUT/'leedy2018_v2/audit.json').read_text())
    oxygen=json.loads((OUT/'oxygen_sensitivity/audit.json').read_text());manifest=json.loads((OUT/'predictor_bundle_v5/manifest.json').read_text())
    assert all(x['passed'] for x in (gates,runtime,replay,oxygen))
    fits=[json.loads(p.read_text()) for p in (OUT/'fits').glob('*.json')]
    assert len(fits)==6 and all(f['selected_success'] for f in fits) and not any(f['state_failures'] for f in fits)
    assert external['rows']==1200 and external['numeric_rows']==1200
    fullsource_existing=json.loads((BASE/'predictor_bundle_v4/manifest.json').read_text())
    assert all(v['training_records']==29 and len(v['training_families'])==5 for modes in fullsource_existing['process_models'].values() for v in modes.values())
    # Check all fixed old artifacts before changing live status/goal documents.
    names=['protected_inputs_before.json']+[f'artifact_manifest_phase{p}.json' for p in (1,2,34,56,7,8,9,10,11,12,13,14,15,16,17,18)]
    checked={}
    for name in names:
        payload=json.loads((BASE/name).read_text());rows=payload['files'] if isinstance(payload,dict) else payload
        bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or energy.sha(ROOT/r['path'])!=r['sha256']]
        assert not bad,bad;checked[name]=len(rows)
    fig=OUT/'figures';fig.mkdir(exist_ok=True)
    d=read(OUT/'leedy2018_v2/predictions.csv');d=d[d.scenario.eq('primary_graph')&d.budget.eq(0)]
    fig1,ax=plt.subplots(figsize=(6.6,4.2))
    sample=d[d.method.eq('median_activation')].sort_values('incorporated_cm3')
    ax.scatter(sample.incorporated_cm3,10**sample.observed_log_nH,c='black',marker='o',s=37,label='Leedy2018 readouts')
    for m,l in [('original_RT','Original shared closure'),('goal_ridge_2','Ridge + two energies'),('nonlinear_kernel','Nonlinear energy learner'),('median_activation','Median activation control')]:
        g=d[d.method.eq(m)].sort_values('incorporated_cm3');ax.plot(g.incorporated_cm3,10**g.predicted_log_nH,'s--',ms=4,label=l)
    ax.set(xscale='log',yscale='log',xlabel='Actual incorporated Si (cm^-3)',ylabel='Hall density (cm^-3)',title='Frozen parameters; old pressure scenario; zero anchors')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig1.tight_layout()
    for ext in ('png','pdf'):fig1.savefig(fig/f'new_source_prediction.{ext}',dpi=160)
    plt.close(fig1)
    keys=['original_RT','goal_ridge_2','nonlinear_kernel','ridge_energy_tuned','stock_const_internal','median_activation','unit_dose']
    alltable=table(keys);commontable=table(keys,'common3')
    pressuretable=['| 方法 | 原假设氧压 | 报道氧分压情景 |','| --- | --- | --- |']
    for method in ('original_RT','goal_ridge_2','nonlinear_kernel','ridge_energy_tuned'):
        scores=[val('phase19_delivery/oxygen_sensitivity/summary.csv',dict(method=method,scenario=s),'MAE') for s in ('old_unmeasured_pressure','reported_partial_pressure')]
        pressuretable.append('| '+method+' | '+' | '.join(f'{v:.6f}' for v in scores)+' |')
    oldtable=['| 方法及额外能量预算 | 0锚点 | 1锚点 | 2锚点 |','| --- | --- | --- | --- |']
    for m,label in [('Ridge_diagonal_xi_k2','Ridge＋2能量'),('mean_goal_k2','均值＋2能量'),('oracle_full','全部能量物理参照')]:
        scores=[val('phase56_validation/canonical_comparison.csv',dict(origin='phase2',method=m,scope='all_query',budget=b),'MAE') for b in (0,1,2)]
        oldtable.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in scores)+' |')
    content=r'''# ML与Sandbox：目标核对、全来源预测工具与外部迁移检验

## 1. 当前目标与本轮完成的事情

用户要求的研究方向保持为：掺杂元素、实际掺入量及可追溯工艺信息，经过数据训练的ML关系与物理Sandbox耦合，预测掺杂后材料性质。载流子是第一主终点，迁移率、派生电导率及有依据的候选排序继续属于目标。理论与可识别性分析用于改进这一预测链，不能单独替代预测方法。

本轮核对了完整目标与阶段1—18实际产物，补齐6个后续模型的全来源版本，导出16预设统一入口，并对Leedy2018作冻结参数外推。没有修改原论文，没有新DFT/MD或物理样品；训练使用CPU。系统goal保持active，GPU1授权继续有效，GPU0不使用。

修正两条状态记录：能量学习v4早已有全部29条、5来源训练的部署参数；此前“无全来源模型”只适用于后续新模型。阶段16也已经做过nH/mu联合监督，“mu监督尚未训练”的旧待办已过时。核对旧记录能避免重复工作和错误贡献表述。

## 2. ML实际学什么，物理求解器做什么

能量路线使用同母类带电态差定义的割线交叉坐标：

\[
 \xi_q=-\frac{A_q-A_{\rm ref}}{q-q_{\rm ref}},\qquad
 \widehat A_q=A_{\rm ref}-(q-q_{\rm ref})f_\psi(x_q).
\]

A为已声明参考下的形成能截距，单位eV；q为整数电荷数；ref为已提供参考电荷态；x为元素/结构/电荷描述；f为训练过的ML模型，参数为psi。xi也是eV；它不一定是热力学稳定电荷转变能级。相同原子组成的差可以消掉原子化学势参考，电子能量参考仍须一致。推断时不读取被屏蔽的真实Aq，额外揭示的2条能量须明示供应。

物性监督路线学习共享的有效冻结闭合、结合能/屏蔽关系或库存分配关系，然后调用守恒及电荷求解器。以统一形式写为：

\[
 R(s;X,\theta)=0,\qquad c=\mathcal C(s;X,\theta),\qquad
 S(x;c,X,\theta)=0,\qquad \widehat y=\mathcal O(x,c;X).
\]

X为已知掺杂/测量输入；s为高温状态；c为冻结母类库存，单位cm^-3；x为观测温度下的电荷状态；theta含学到的共享关系参数；O将状态映射为Hall等观测。预测查询不输入真实nH或mu。实际热史未知时，共享Tf是有效模型参数，不能当成实测生长或冻结温度。

在根局部光滑、雅可比可逆及求解残差合格时，隐式求导为：

\[
 \frac{ds}{d\theta}=-R_s^{-1}R_\theta,\qquad
 \frac{dx}{d\theta}=-S_x^{-1}\left(S_c\frac{dc}{d\theta}+S_\theta\right).
\]

下标表示相应偏导矩阵。总导数必须包含高温库存随参数改变的项，不能只对最终电荷再平衡求导。这是可训练物理计算图的数学条件，并不自动保证形成能、散射近似或预测精度正确。

## 3. 全来源工具及预算

新增6次全来源拟合、12个初始优化，均正常停止，未触发360步统一细化，未记录状态求解失败。原始数据仍为29片/5DOI；联合变温学习复用旧曲线；联合mu模型还使用旧mu标签。全来源训练本身不产生新的留出成绩。

统一入口包括7种旧能量方法，原RT、结合屏蔽联合、同态Hall纯离化散射、Hall附加固定散射、nH/mu联合训练、常数库存及其内部来源适应，以及单位剂量/中位数激活两个简单对照。命名预设供比较，不自动把当前外测最佳者冒充预先选定的赢家。

旧能量补全零额外标签时，Si供应844/873、Sn850/873条能量；加2标签变为846/852。完整能量参照及后续物性监督模型供应873条。它们不是从零推断未知元素的同预算竞赛。

'''+'\n'.join(oldtable)+r'''

上表来自旧29/5回顾性测试。Ridge＋2和均值＋2很接近，且Sn揭示集合不同，不能把其差全部归为ML贡献。全部能量参照也不保证最小物性误差：过程闭合可能与能量近似发生补偿。较小误差不等价于更真实的微观参数。

本轮92项数值检查全部通过，最大误差约4.22e-8。独立进程2318次接口请求验证不读取性质表；其中数值输出与父进程一致，重复剂量锚点拒绝也保留。另用保存的训练预测作独立参考，497条nH/mu自身温度输出复现，最大偏差9.54e-12；54个合法不同剂量锚点请求通过。后两项是集成与一致性检验，不是新增准确度实验。

## 4. Leedy2018的数据边界

原始作者论文及机构入口：[Leedy等，APL Materials 6,101102](https://corescholar.libraries.wright.edu/physics/1482/)。采用550°C系列实际SIMS薄膜Si浓度，按靶材标签配对本研究Hall读数。4点由Fig8读取，1wt点采用正文数值；Fig8未另造1wt点。名义靶材百分比不换算为实际薄膜剂量。

该组共5个样品条件。0.05wt薄膜SIMS存在深度变化，故保存全部5点，也预设剔除该非均匀输入的4点对照。正文0.025wt的n/mu与图示有差别，另存正文替代情景。图读数固定3像素范围只是读取敏感性，不是测量置信区间。Hall方法段报告0.5T，未定量给出精确测量温度；300K是本轮比较约定，不冒充仪器温度记录，见source_metadata_amendment.json。5组标签/输入情景合计1200条预测全部保存，未丢弃失败或不利模型。

3点Si剂量超过旧训练上限1.4e20cm^-3；另2点仍在范围内。来源是新的载流子DOI，但条件mu训练集包含同团队2017前作；另外给出排除该前作、仅49对/10来源训练的CT头。不能称独立实验室盲测或全程无关联研究。全部来源已按回顾性审计处理，未用新标签重训参数或选择超参数。

## 5. 冻结模型对真实新来源的预测

表中为nH的平均绝对log10误差，单位dex。0/1/2预算分别留下5/4/3个查询；1锚点用0.025wt样品，2锚点再加0.1wt样品，规则在计算前固定。

'''+alltable+r'''

![零锚点实际掺入量到载流子预测；物理模型出现低斜率平台](figures/new_source_prediction.png)

这组结果不支持当前ML＋Sandbox已可靠泛化。原模型和Ridge＋2明显低估高剂量载流子；中位数激活比例强对照更好。非线性能量模型和内部库存更新有局部改善，仍不能弥补差距。4个较均匀输入的对照也保留同一结论，不能把全部失败归咎于非均匀薄膜。

跨锚点比较必须固定查询集。共同3个查询如下：

'''+commontable+r'''

原RT和Ridge＋2的共同查询误差不变，因为训练选择了不更新。前表随预算变差不能称为“新增锚点损坏模型”，它还含查询组成变化。中位数激活的共同两点MAE偶然不变也不意味着它未更新；MAE在异号残差间可出现平坦区。

## 6. 报道氧分压能否消除偏差

该实验气体条件对应约6.42e-7atm的O2分压，而旧训练全来源使用1e-5atm情景。冻结所有能量、ML权重、Tf和背景，仅替换氧输入；14个非经验预设、2情景、5点，共140预测。原情景70次回放最大偏差4.91e-11。

'''+'\n'.join(pressuretable)+r'''

氧输入改变能降低约0.3dex误差，但在当前闭合下仍远逊于中位数激活。因此“只要补齐氧压就能解决”不成立。PLD也不是已证实在有效Tf达到气相热力学平衡，这只是受控输入敏感性，不能唯一确定真实失效机制。

## 7. 对目标与后续研发的约束

本轮增加的是可调用完整工具与一次外部迁移检验，未建立新ML优越性。目标保持active。所有初始失败也保留：这轮Leedy CSV初版把数字状靶材标签读成float，导致定位失败；v2从明确样品ID恢复文本标签，未修改数值/模型/规则。独立推断V1产生的重复剂量拒绝保留，另以54个合法锚点任务补核。

下轮首先核对“全局内部物理闭合产生低斜率平台”与“来源适应先验过强/坐标不合适”两种预测风险，复用阶段17已有native/complex来源接口，避免重新命名已实现模块。只有确认尚未验证的具体假设后，才冻结一个低维、训练内选择的来源推断改进，并同时给输出校准和简单剂量比例相同锚点/选择预算。新来源已暴露，后续使用它改模型必须称开发，不能再作盲测。

需要增加的证据优先是带实际化学剂量、明确氧/热条件的独立来源，而不是继续在旧29条上增大函数容量。若只有单点Hall，不能把未知工艺史或化学剂量从标签反推填回输入。能量学习、目标导向计算选择与实际物性终点仍须连起来验证。

完整迁移率的实用精度、可校准预测区间、工艺条件驱动的泛化及新稀土排名仍未建立。统计形式、求解残差和概率守恒的正确性，均不能替这些物性证据过关。研究方向是继续做可检验的ML＋Sandbox预测改进，不能把这份解释和审计报告当作最终目标完成。

## 8. 调用与复现入口

统一库为src/sandbox/property_predictor_v5.py，模型清单位于本目录predictor_bundle_v5/manifest.json；能量模型继续依赖父目录predictor_bundle_v4。输入JSON示例见example_input.json；CLI为scripts/trainable_sandbox_20261017/predict_properties_v5.py。运行顺序和完整命令见本目录README.md。

模型仅支持Si/Sn。能量预设只支持300K、未知H的零供应情景；后续内部预设支持显式H及自身温度，实际Sn低温迁移未验证。growth_T_C在本批预设只是记录，不能宣称它参与了工艺学习。变压仅作为上节受控研究情景，不是已校准的生产接口。电导率由q*nH*muH派生，无新独立sigma标签。
'''
    report=OUT/'ML与Sandbox_目标核对与跨来源预测检验.md';report.write_text(content)
    energy.write(OUT/'report_claims.json',dict(code_sha256=energy.sha(Path(__file__)),claims=CLAIMS))
    audit=dict(created_utc=energy.now(),system_goal_status='active',goal_complete=False,
        direction='trained ML + Sandbox predicts dopant properties;theory serves actualprediction',
        parent_manifests_unchanged=checked,parent_entries_verified=sum(checked.values()),
        earlier_energy_full5_model_already_existed=True,new_full5_model_fits=len(fits),
        optimizer_starts=sum(len(f['trials']) for f in fits),all_selected_stopped=True,model_presets=len(manifest['methods']),
        numerical_checks=gates['checks'],independent_interface_queries=runtime['inference_queries'],
        training_API_replays=replay['saved_training_prediction_replays'],distinctdose_anchor_checks=replay['distinct_dose_anchor_tasks'],
        new_source_specimens=5,new_source_scenarios=5,new_source_predictions=1200,oxygen_scenario_predictions=140,
        original29_unpooled=True,newsource_used_to_select_models=False,independent_lab_claim=False,
        stable_increment_established=False,conditional_transport_precursor_disclosed=True,
        new_physical_measurements=0,new_DFT_MD=0,GPU_used=False,GPU_authorized='GPU1 only',
        retained_failure='LeedyV1 numeric-looking sample_label conversion;V2 reconstructs from sample_id',
        theory_revisions=['do not equate query composition changes with harmful anchor updates',
            'oxygen input alone insufficient within frozen closure','earlier full5 energy models already available',
            'cross-source prediction remains actual endpoint,not explanation-only'])
    energy.write(OUT/'goal_direction_check.json',audit)
    # Save live audit before replacing its stale descriptions.
    oldpath=BASE/'goal_requirements_audit.json';snapshot=OUT/'goal_requirements_before_phase19.json'
    if not snapshot.exists():shutil.copyfile(oldpath,snapshot)
    req=json.loads(snapshot.read_text());req.update(checked_utc=energy.now(),goal_status='active',
        previous_goal_turn='phase18 completed and frozen;no stable newlogit gain',
        current_turn='phase19 progress:full-goal audit,full-source collection and newsource frozen prediction',
        current_execution='six full5 fits and16presets delivered;Leedy5studyconditions/1200predictions;oxygen140scenario rows;fullgoal remainsactive',
        active_jobs=[],sufficient_to_mark_goal_complete=False)
    req['implemented_and_verified'].append(dict(requirement='fullgoal audit,explicitbudget fullsource predictor and externalactualdose propertytest',
        evidence=['phase19_delivery/goal_direction_check.json','phase19_delivery/predictor_bundle_v5/manifest.json',
            'phase19_delivery/training_api_replay/verification.json','phase19_delivery/leedy2018_v2/audit.json','phase19_delivery/oxygen_sensitivity/audit.json'],
        scope='29oldtrain;5newpublishedconditions notblind;notstablegain;earlierv4full5acknowledged'))
    req['outstanding_prediction_requirements']=[
        dict(requirement='继续优化核心ML+Sandbox掺杂性质预测',status='active',reason='共享关系已训练并统一部署；新源低斜率平台与来源适应失配仍未解决；不以解释替代预测'),
        dict(requirement='可靠跨研究/工艺迁移',status='not_established',reason='旧Azizie及新增Leedy外推均保留失败；Leedy氧输入修正不足以解决，3点超旧Si范围'),
        dict(requirement='完整迁移率链的实用准确度',status='not_established',reason='阶段16已进行nH/mu联合监督；旧5RT配对/2DOI、54T读数仍有限；Leedy新5含读图/相关前作，非独立实验室确认'),
        dict(requirement='工艺输入与新掺杂候选排序',status='not_established',reason='当前共享模型工艺条件覆盖不足，未验证稀土；不把只有Si/Sn的调用当作完整新元素能力'),
        dict(requirement='方法学创新的增量证据',status='not_proven',reason='需要相同信息/计算预算的稳定收益与文献定位；不把普遍优于基线预设为必须得到的结论')]
    req['next_concrete_work']=['audit old native/complex source-coordinate and priors against newsource plateau before any newcapacity',
        'freeze one training-only source-inference hypothesis plus samebudget direct/output controls;newLeedy nowdevelopmentnotblind',
        'seek actualdose+oxygen+thermal supported independent study;retain original29 and new5 separately until protocolfixed']
    req['phase19']=audit;energy.write(oldpath,req)
    example=dict(dopant='Si',incorporated_cm3=1e18,method='ridge_energy_tuned',observation_T_K=300.,H_known=False,H_cm3=0.,anchors=[])
    energy.write(OUT/'example_input.json',example)
    guide='''# 全来源ML＋Sandbox预测入口与本轮结果

先读ML与Sandbox_目标核对与跨来源预测检验.pdf。goal仍active，主目标是预测掺杂性质。

原能量学习v4已有全29/5来源参数；本轮新增6个后续全来源模型，统一16预设。新源Leedy外推未解决，不把接口交付称为准确度提升。

从Ga2O3-Sandbox目录运行：

    env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/lawrence/anaconda3/envs/ga2o3/bin/python scripts/trainable_sandbox_20261017/predict_properties_v5.py --input-json results/trainable_sandbox_20261017/phase19_delivery/example_input.json --output /tmp/ga2o3_prediction_v5.json

复现顺序（不可修改冻结协议后覆盖重跑）：

    phase19_verify.py --gates
    phase19_delivery.py --train --export --workers 2
    phase19_verify.py
    phase19_training_replay.py
    phase19_leedy_v2.py --extract --evaluate
    phase19_pressure.py
    phase19_report.py

脚本均位于scripts/trainable_sandbox_20261017。全部CPU单线程；不使用GPU0。父v4结果和KROGER私有输入须保留。需要GPU时仅GPU1，先检查占用。

读图5样品、5标签/输入情景，1200预测；非均匀薄膜/正文冲突/超训练范围均披露。oxygen_sensitivity是固定参数情景，不是新工艺训练。leedy2018/与leedy.log保留最初CSV标签转换错误；正式结果为leedy2018_v2/。旧模型训练结果/主29数据均未覆盖。

numerical_gates：92门槛；runtime_verification：2318接口请求，含重复剂量拒绝；training_api_replay：497保存训练预测+54合法不同剂量锚点。均非材料新精度。report_claims.json按路径/选择器/哈希追溯所有表格数值。

API只提供明确范围内的研究预测；成长温度在本批预设为元数据。能量预算、实际剂量范围、H状态、源适应坐标均返回。无绝对寿命/PDR、稀土排名或校准置信区间。
'''
    (OUT/'README.md').write_text(guide)
    print('report',report,'parents',sum(checked.values()),'claims',len(CLAIMS),flush=True)


if __name__=='__main__':main()
