"""Document the actual verified48-method bundle and the carbon-scope wrapper."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import phase2_energy as e

BASE = e.BASE / 'phase27_charge_feasible'
BUNDLE = BASE / 'delivery/predictor_bundle'


def main():
    for name in ('delivery/cohort_replay/audit.json', 'external_Azizie/audit.json',
        'delivery/final_checks/audit.json', 'carbon_scope/audit.json', 'input_scope_annotations/audit.json'):
        assert json.loads((BASE / name).read_text())['passed'], name
    assert not (BASE / 'delivery/documentation_audit.json').exists()
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    for f in manifest['files']: assert e.sha(BUNDLE / f['path']) == f['sha256'], f['path']
    rows = []
    for name, spec in sorted(manifest['presets'].items()):
        fit = json.loads((BUNDLE / 'models' / (spec['model_key'] + '.json')).read_text())
        rows.append(dict(preset=name, context=spec['context'], variant=spec['variant'], model_type=spec['model_type'],
            declared_method=spec['method'], actual_evaluation_method=fit.get('evaluation_method', fit['method']),
            parameter_count=len(fit['parameters']) if fit.get('parameters') is not None else 0,
            serving_qualified=bool(spec['serving_qualified']), training_valid=fit['training_valid'],
            selected_success=fit.get('selected_success'), explicit_reference_fallback=bool(fit.get('reference_fallback')),
            source_kind=spec['source_kind'], shared_regularization=spec.get('shared_reg'),
            source_lambda0=spec['source_lambdas']['0'], source_lambda1=spec['source_lambdas']['1'],
            source_lambda2=spec['source_lambdas']['2'], model_key=spec['model_key'],
            training_support=spec.get('training_support')))
    assert len(rows) == 624 and len(manifest['files']) == 390
    unqualified = sum(not r['serving_qualified'] for r in rows)
    e.csvwrite(BASE / 'delivery/preset_catalogue.csv', rows)
    guide = f'''# 第27阶段保存模型使用说明

## 实际交付

{len(rows)}个预设、{len(manifest['files'])}个最小参数文件，覆盖全部48方法、DOI/保守关联组排除及全来源上下文。其中{unqualified}个预设不满足训练/停止资格，保留并由接口拒绝；资格不等于材料精度。目录见preset_catalogue.csv。

原定主方法是cluster::full::FeasibleProcess_internal。名字只标记预先固定的研发主方法，不表示它已经是所有来源/预算最优。实际结论以完整结果报告为准，不能按目录或接口成功率选择性宣称优势。

新族学习补偿边界以上的内部原子份额；Envelope是同端点信息的直接Hall回归控制。参考回退以actual_evaluation_method及实际参数数目为准。Envelope若回退ParentReference，其实际输出是物理参考。旧36方法和全部不利结果保留。

## 显式输入

每个条件必须有element、incorporated_cm3、Tobs_K、growth_T_C、H_known、H_cm3、carbon_known、carbon_cm3、scenario_pO2_atm、oxygen_input_kind。当前只支持Si或Sn及300K比较任务；每次来源适应只包含一种掺杂元素。

incorporated_cm3为实际掺入量，或明确声明的化学过程代理。名义靶材比例、Hall浓度、C-V净电荷不能静默替代化学浓度。growth_T_C是名义摄氏温度，不是完整退火/淬火历史。

未知H用H_known=false、H_cm3=0表示声明的未知零情景，不能说实测无H。氧分压必须为正并说明来源；O3、等离子体压强与未报道氧压都不能自动等同平衡O2。

物理缺陷层不包含C。已知正C输入不会改变该物理库存，公开V3包装器会输出reported_carbon_unmodeled_in_physical_state及警告。DirectRidge可以使用碳表格特征，但不由此得到物理C库存。队列样品D是表面富集的平均读数，不能直接补成均匀体掺杂。详见docs/ML_SANDBOX_REPORTED_CARBON_SCOPE_CORRECTION_20261018.md。

## 使用带范围提示的公开CLI

从项目根目录运行：

    env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/lawrence/anaconda3/envs/ga2o3/bin/python scripts/trainable_sandbox_20261017/predict_feasible_properties_v2.py --bundle results/trainable_sandbox_20261017/phase27_charge_feasible/delivery/predictor_bundle --request results/trainable_sandbox_20261017/phase27_charge_feasible/delivery/final_checks/example_prediction_request.json --output /tmp/ga2o3_feasible_prediction.json

请求包含preset、inputs及anchors三个字段。零锚点为anchors=[]。最多两项锚点各含input_index和measured_nH_cm3；输入索引对应inputs，两个锚点须为不同输入条件。只给真实允许的Hall锚点，不把待预测查询标签放入适应。

共享权重和来源lambda来自训练内选择，预测时不重选。lambda=null表示不更新。新族shared_reg=null是显式参考候选，不是把有限线性logit系数神奇压成一个可表示所有参考的函数。

模型只读取最小保存参数、明确输入/锚点和指定物理参考；不读取训练或查询实验表。第三方KROGER数字库未复制进模型包，运行需要本工作区已有且SHA一致的参考NPZ。

## 输出与范围

predictions逐点给出numeric、valid、log10_nH_cm3、log10_muH_cm2Vs和log10_sigma_S_cm。线性值单位为cm^-3、cm2/(V s)、S/cm。电导由前两项派生，不是独立验证标签。物理族使用同状态输运；直接族使用CT(predicted nH)，显式参考回退按其真实模型读出。

Feasible的自由电子下限仅是Hall可接受性的必要条件；原电子占优、Hall下限和数值检查继续保留。它是已声明电子导电Si/Sn域的条件预测，不是判断任意新样品是否导电或是否p型的分类器。

部分点失败时保留其他有效点，失败点明确标志；无效数值延续不当作合格材料预测。valid是实现/单电子观测域标志，不认证完整组成、真实微观库存、Born近似或新来源精度。含已知C的提示也不会静默删点或改变原主分数。

剂量/温度超出训练支持时返回范围提示。当前无已校准预测置信区间，不验证稀土、真实反应率、寿命或PDR。预测μ不应混同给定实测nH的条件迁移率任务。

## 梯度与版本证据

增加--zero-anchor-gradient并使用preset、input请求可导出三性质局部雅可比；Feasible含阈值和共享特征的输入链，Envelope是直接端点读出导数，参考回退只有真实参考参数。它不包含CV选择或来源重拟合的导数，也不是已验证的变温材料精度。

正式9504点独立进程主回放使用冻结V2工厂；公开V3只补回已知C范围提示。V3在8个实际快照上的数值/资格不变和提示由carbon_scope/audit.json核验，不称为V3额外全队列回放。主行的范围注释见input_scope_annotations/all_main_rows.csv。

源内/来源排除/新化学代理任务的真实性、数据交集及历史曝光情况见完整结果报告。全部48方法的原始预测、覆盖、能量先验和不利比较共同决定适用范围，不能用审计绿灯替代预测增量。
'''
    dest = BASE / 'delivery/README.md'; dest.write_text(guide)
    e.write(BASE / 'delivery/documentation_audit.json', dict(passed=True, created_utc=e.now(),
        presets=len(rows), models=len(manifest['files']), unqualified_presets=unqualified,
        manifest_sha256=e.sha(BUNDLE / 'manifest.json'), guide_sha256=e.sha(dest),
        catalogue_sha256=e.sha(BASE / 'delivery/preset_catalogue.csv'),
        public_metadata_wrapper='FeasiblePropertyPredictorV3', main_numerical_replay='V2,9504points',
        reported_carbon_disclosed=True, no_new_fits_or_predictions=True, goal_complete=False,
        code_sha256=e.sha(Path(__file__))))
    print('Documented actual48-method predictor', len(rows), len(manifest['files']), 'unqualified', unqualified, flush=True)


if __name__ == '__main__': main()
