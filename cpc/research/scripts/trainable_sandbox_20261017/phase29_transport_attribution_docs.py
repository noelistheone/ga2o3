"""Source-bound explanation of the completed, post hoc transport diagnostic."""
from pathlib import Path
import sys
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from phase29_report_claims import Evidence,sanitize_units,audit
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE=e.BASE/'phase29_joint_scattering'
OUT=BASE/'transport_attribution_v1'


def main():
    target=OUT/'固定状态散射诊断.md'
    assert not target.exists(),'preserve previous explanation'
    assert json.loads((OUT/'audit.json').read_text())['passed']
    e.write(OUT/'docs_protocol_before.json',dict(created_utc=e.now(),
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in
            (Path(__file__),Path(__file__).with_name('phase29_report_claims.py'))},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in
            (OUT/'summary.csv',OUT/'points.csv',OUT/'audit.json',BASE/'prediction/plan.json')},
        posthoc_interpretation=True,no_new_fitting_or_model_selection=True))
    plan=json.loads((BASE/'prediction/plan.json').read_text())['plans'];rows=[]
    for p in plan:
        if p['variant']!='JointProcess_internal':continue
        path=ROOT/p['fit_path'];assert e.sha(path)==p['fit_sha256']
        fit=json.loads(path.read_text())
        rows.append(dict(context=p['outer_held'] or 'full',
            gamma=fit['config']['gamma'],alpha=fit['config']['alpha'],
            mu0_cm2Vs=205*10**fit['parameters'][-1],fit_path=p['fit_path'],fit_sha256=e.sha(path)))
    e.csvwrite(OUT/'selected_primary_contexts.csv',rows)
    ev=Evidence();summary=OUT/'summary.csv';report=OUT/'audit.json'
    text=r"""# 固定状态散射诊断：联合模型的迁移率失配来自哪里

## 1. 这项诊断回答什么

本项在查看本轮来源留出结果后进行，分析对象为已固定的主方法JointProcess_internal。只考察零锚点任务：原子库存、自由电子浓度、电荷平方浓度和全部已训练参数保持原记录，单独把额外散射尺度恢复到原205设置。它是一项事后模型内干预，输出不作为新训练模型或新的独立验证成绩。

完整主实验中的候选、超参数、来源适应、留出预测和模型权重均保持原记录。本诊断使用相同的来源、条件、情景和已报告的标签。

## 2. 操作怎样实施

从原预测的coherent_transport记录读取自由电子浓度 \(n\)、总电荷平方浓度 \(M_2\)、温度 \(T\)、质量、介电常数及积分设置。总量定义为

\[
M_2=N_b+\sum_i c_i\sum_q q^2\pi_{iq}.
\]

\(N_b\) 是按+1中心处理的有效正背景；\(c_i\) 是母类库存；\(\pi_{iq}\) 是电荷态概率；\(q\) 以元电荷为单位。浓度单位cm⁻³。这里保存的状态是原模型的内部结果，不是独立测得的缺陷库存。

先用原有散射尺度重构原输运输出，再只改变

\[
\tau(E)^{-1}=R_{\mathrm{ion}}(E;n,M_2,T)+\tau_0^{-1},
\qquad
\tau_0=\frac{m\mu_0\,10^{-4}}{e},\quad
\mu_0\leftarrow205\ \mathrm{cm^2/(V\,s)} .
\]

\(E\) 是电子动能，单位J；\(R_{\mathrm{ion}}\) 单位s⁻¹；\(\tau,\tau_0\) 单位s；\(m\) 单位kg；\(e>0\) 是元电荷；换算因子把迁移率转为SI。205是沿用的经验设置，不是新推导的材料常数。

"""
    text+='实际读取'+ev.fact(report,['actual_saved_states'])+'个保存状态，执行'+ev.fact(report,['transport_evaluations'])+'次输运求值；新冷热求根'+ev.fact(report,['new_hot_or_cold_solves'])+'次，新拟合'+ev.fact(report,['new_fits'])+'次。原输运最大重构差为'+ev.fact(report,['maximum_original_transport_replay_error_dex'],'.3e')+'dex。\n\n'
    text+=r"""## 3. 数学上为什么Hall浓度和迁移率会同时变化

设 \(w(E)\) 是固定 \(n,T\) 下的正、归一化输运权重，\(t_j=\langle\tau^j\rangle_w\)。当前单带RTA读出为

\[
r_H=t_2/t_1^2,\quad n_H=n/r_H,\quad
\mu_H=(e/m)(t_2/t_1)10^4,\quad
\sigma=e n_H\mu_H .
\]

定义 \(u=\log_{10}(\mu_0/[\mathrm{cm^2/(V\,s)}])\)、\(r=\tau/\tau_0\)，再定义归一化权重 \(v=w\tau/t_1\)、\(a_1=\langle r\rangle_v>0\) 和 \(V=\operatorname{Var}_v(r)\ge0\)。现有正散射推导给出

\[
\partial_u\log_{10}n_H=-2V/a_1,\quad
\partial_u\log_{10}\mu_H=a_1+2V/a_1,\quad
\partial_u\log_{10}\sigma=a_1 .
\]

因此固定状态下，

\[
0\le
\frac{|\partial_u\log_{10}n_H|}
{\partial_u\log_{10}\mu_H}
=\frac{2V}{a_1^2+2V}<1.
\]

最后一步只用了 \(a_1>0,V\ge0\)。当额外常数通道占主导时，\(r\) 接近1、\(V\) 接近0，Hall浓度对该尺度可以很不敏感，而迁移率仍明显变化。因此Hall浓度的选参成绩本身不能保证输运泛化。实际训练中的库存、温度、背景也可变化；上述固定状态偏导不能直接当作全部共享参数的总效应。

这项结论受当前标量单带、抛物线和弛豫时间近似限制，不是所有材料的普遍输运定理。

## 4. 共同查询上的实际结果

误差为DOI等权的绝对对数误差，单位dex。先在同一报告条件内平均情景误差，再对条件和DOI平均。以下干预结果没有参与主模型选择。

| 性质 | 原联合模型 | 同状态恢复205 | 独立训练的ShareProcess205 |
| --- | --- | --- | --- |
"""
    for prop in ('nH','mu','sigma'):
        selector=dict(scope='common20',property=prop)
        text+='| '+prop+' | '+' | '.join(ev.csv(summary,selector,k) for k in ('original','same_state205','control'))+' |\n'
    text+='\n载流子比较覆盖'+ev.csv(summary,dict(scope='common20',property='nH'),'conditions','.0f')+'个条件、'+ev.csv(summary,dict(scope='common20',property='nH'),'DOIs','.0f')+' DOI；迁移率与派生电导覆盖'+ev.csv(summary,dict(scope='common20',property='mu'),'conditions','.0f')+'个配对条件、'+ev.csv(summary,dict(scope='common20',property='mu'),'DOIs','.0f')+' DOI，均来自原Si配对队列。新增Sn的固定情景诊断保留在summary.csv中。\n\n'
    text+='![同一状态的散射尺度干预；事后诊断结果，不是新增验证方法。](fixed_state_intervention.png)\n\n'
    text+=r"""## 5. 怎样严格分解误差差额

令 \(J\) 为已训练联合模型，\(C\) 为只恢复205的固定状态输出，\(S\) 为独立训练的ShareProcess205，\(y\) 为同一观测。对每个读数，

\[
|J-y|-|S-y|
=\bigl(|J-y|-|C-y|\bigr)
+\bigl(|C-y|-|S-y|\bigr).
\]

在相同有效覆盖与相同线性平均权重下，该恒等式对MAE仍成立。第一项对应所定义的固定状态散射干预，第二项包含其余参数、训练损失和配置选择的差异，不能全部解释为单独的库存因果效应。交换干预顺序一般会得到不同的分配。

| 性质 | 总MAE差额 J−S | 散射干预项 J−C | 其余拟合与选择项 C−S |
| --- | --- | --- | --- |
"""
    for prop in ('nH','mu','sigma'):
        selector=dict(scope='common20',property=prop)
        text+='| '+prop+' | '+' | '.join(ev.csv(summary,selector,k,'+.6f') for k in
            ('total_MAE_delta','scattering_MAE_delta','remaining_fit_and_selection_MAE_delta'))+' |\n'
    text+=r"""

这明确表明：本轮联合模型相对固定205过程模型的迁移率退化，较大部分出现在所学习的额外散射尺度上。恢复该尺度时，载流子误差略有变大。该结果暴露了当前训练与迁移中的取舍，也保留了恢复205后仍然存在的较大输运误差。

电导误差还受载流子与迁移率有符号误差的叠加和抵消影响；三种MAE不能直接相加。上述分解是指定模型内的数值恒等式，不唯一确定样品的真实散射机制。

## 6. 配置与来源差异

下表直接回读原保存拟合；gamma是配对电阻率方向的训练权重，alpha是声明的统计正则强度，mu0为有效散射尺度。full使用全部训练组，不是来源留出成绩。

| 排除组或上下文 | gamma | alpha | mu0 cm²/(V·s) |
| --- | --- | --- | --- |
"""
    for row in rows:
        selector=dict(context=row['context'])
        text+='| '+row['context']+' | '+' | '.join(ev.csv(OUT/'selected_primary_contexts.csv',selector,k,'.6g') for k in ('gamma','alpha','mu0_cm2Vs'))+' |\n'
    text+=r"""

不能简单归因为“没有使用迁移率监督”：部分拟合确实采用非零配对权重。现有证据支持检查共享散射尺度的跨来源稳健性，以及只按零锚点载流子误差选择配置时对输运性能的约束。组成、微结构、空间非均匀和单带RTA适用性仍可能参与失配。

## 7. 对后续预测设计的实际约束

下一项可检验方向是训练内联合审计载流子与电阻率，并研究低维来源输运收缩。若使用支持样品的迁移率，须明确增加了第二个原始Hall观测方向，给直接ML、经验输运和内部模型相同的支持信息；保留当前nH-only任务作为参照。

当前干预输出不能直接替代原主方法的留出成绩。新增设计应使用独立版本、固定协议和完整对照，检验改善是否能够跨来源成立。

## 8. 证据索引

points.csv保留全部保存状态、原输出、固定状态干预、匹配对照和各观测；summary.csv保留所有查询、共同查询和新增Sn情景下的结果。audit.json核查原输运重构、固定状态单调性、误差恒等式及原主表回放。数值门槛只用于确认诊断实现，没有新增材料测量。

diagnostic_claims.json将文中动态数值绑定到原文件SHA、行筛选或JSON键及正文字符区间。原主实验的7056行情景输出、48个变体、训练参数和服务预设均保持原文件。
"""
    text,units=sanitize_units(text);text,spans=ev.render(text);target.write_text(text)
    claims=dict(created_utc=e.now(),claims=ev.claims,spans=spans,report_sha256=e.sha(target),
        unit_substitutions=units,posthoc_diagnostic_only=True)
    e.write(OUT/'diagnostic_claims.json',claims)
    result=audit(target,claims);e.write(OUT/'documentation_audit.json',result)
    print('Diagnostic explanation',result['passed'],'claims',result['claims_checked'],flush=True)
    if not result['passed']:raise RuntimeError('Preserve explanation claim failure')


if __name__=='__main__':main()
