"""Write the actual saved-model catalogue and usage guide after full replay."""
from pathlib import Path
import sys,json

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e

OUT=e.BASE/'phase26_pool_share';BUNDLE=OUT/'delivery/predictor_bundle'


def main():
    for p in ('delivery/cohort_replay/audit.json','external_Azizie/audit.json','delivery/final_checks/audit.json'):
        assert json.loads((OUT/p).read_text())['passed'],p
    manifest=json.loads((BUNDLE/'manifest.json').read_text());rows=[]
    for f in manifest['files']:assert e.sha(BUNDLE/f['path'])==f['sha256'],f['path']
    for name,spec in sorted(manifest['presets'].items()):
        f=json.loads((BUNDLE/'models'/(spec['model_key']+'.json')).read_text())
        rows.append(dict(preset=name,context=spec['context'],variant=spec['variant'],model_type=spec['model_type'],
            declared_method=spec['method'],actual_evaluation_method=f.get('evaluation_method',f['method']),
            serving_qualified=bool(spec['serving_qualified']),training_valid=f['training_valid'],selected_success=f.get('selected_success'),
            explicit_reference_fallback=bool(f.get('reference_fallback')),source_kind=spec['source_kind'],
            shared_regularization=spec.get('shared_reg'),source_lambda_0=spec['source_lambdas']['0'],
            source_lambda_1=spec['source_lambdas']['1'],source_lambda_2=spec['source_lambdas']['2'],
            parameter_count=len(f['parameters']) if f.get('parameters') is not None else 0,
            model_key=spec['model_key'],training_support=spec.get('training_support')))
    assert len(rows)==468;e.csvwrite(OUT/'delivery/preset_catalogue.csv',rows)
    modelcount=len(manifest['files']);unqualified=sum(not r['serving_qualified'] for r in rows)
    guide=f'''# 第26阶段保存模型使用说明

## 当前交付

本目录包含实际导出的{len(rows)}个预设、{modelcount}个最小参数文件。{unqualified}个预设未满足训练/停止资格，保留在目录中并由预测接口明确拒绝。完整列表见preset_catalogue.csv；资格标志是代码服务条件，不是材料预测准确性保证。

主方法预设为cluster::full::ShareEffectiveProcess_internal。它在外层分数产生前指定，不能将“主方法”理解为所有来源或锚点预算下的已证最优方法。全部36方法均有预设。方法比较和适用范围以完整结果报告为准，不按本目录的顺序解释成性能排名。

## 显式输入

每个条件包含以下10个字段：element、incorporated_cm3、Tobs_K、growth_T_C、H_known、H_cm3、carbon_known、carbon_cm3、scenario_pO2_atm、oxygen_input_kind。实际JSON例子在final_checks/example_prediction_request.json；同次实际输出在final_checks/example_prediction_response.json。

element当前只允许Si或Sn，Tobs_K为300。incorporated_cm3必须是正的实际掺入量或明确标记的化学代理，不能将名义靶材比例、C-V净电荷或Hall读数偷偷当成化学真值。growth_T_C为摄氏数值；记录它不代表已知完整退火/冷却历史。

未知H用H_known=false且H_cm3=0表示一个情景，不是实测无H。当前模型不支持非零碳输入，不能静默忽略已知碳。氧分压必须为正，oxygen_input_kind记录其来源；O3/等离子体腔压不能直接当平衡分子O2，缺失时须明确标注有效情景。

## 运行保存模型

在Ga2O3-Sandbox项目根目录运行：

    env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/lawrence/anaconda3/envs/ga2o3/bin/python scripts/trainable_sandbox_20261017/predict_pool_properties_v1.py --bundle results/trainable_sandbox_20261017/phase26_pool_share/delivery/predictor_bundle --request results/trainable_sandbox_20261017/phase26_pool_share/delivery/final_checks/example_prediction_request.json --output /tmp/ga2o3_pool_prediction.json

请求使用preset、inputs和anchors三个顶层字段。anchors=[]为零锚点。需要适应时加入最多两项，每项为input_index与measured_nH_cm3；measured_nH_cm3应是该条件的真实Hall浓度，不能放入待预测查询的标签。两个锚点须是不同输入/剂量条件，索引对应inputs数组。

保存模型只读取其最小参数、预设及精确物理参考，不读取训练/查询实验表。模型包没有复制第三方KROGER数据库；运行时需要本工作区已有的、SHA一致的物理NPZ。缺少该输入应报错，不能偷偷使用其他数据库。

## 结果含义

predictions逐点给出numeric、valid、log10_nH_cm3、log10_muH_cm2Vs及log10_sigma_S_cm。线性值单位依次为cm^-3、cm2/(V s)和S/cm。电导率由前两项派生，不是独立实验验证。物理预设使用同一电荷/输运状态；直接ML预设采用CT(predicted nH)经验读出。

物理状态与观测域失败均保留；invalid数值延续不能作为合格材料预测。仅部分点失败时，其他有效点仍可输出。物理内部来源更新实际改变库存并重解；offset来源更新在输出层校准，same_physical_state_for_output会标记两者区别。

请求中的source_lambda来自训练内选择，null表示不更新；shared_reg=null对新份额族是精确参考离散候选，并非低维logit的连续无限正则极限。读取actual_evaluation_method和参数维数，不能因预设名字含Share就假设该折一定学到了份额系数。

## 梯度和适用范围

物理预设可通过CLI的--zero-anchor-gradient导出三性质的参数/输入雅可比；实际请求例子为final_checks/DOI_ShareEffectiveProcess_request.json等。梯度是固定已训练参数的局部IFT，含共享函数输入链，不包含CV选择或来源重新拟合的导数。参数维数随实际参考回退而变。

训练支持范围、来源排除关系和预警保留在manifest与逐点输出中。当前只有回顾性Si/Sn条件验证，不提供已校准置信区间，也不验证稀土、任意新工艺、全温输运、真实微观库存、寿命或光暗电流比。已有Hall标签数量和历史缺陷能量/DOS信息成本分别报告。

## 复现证据

cohort_replay/audit.json是全部36方法的独立进程回放记录，external_Azizie/audit.json位于阶段上一级。它们检查复用和固定输入/锚点协议；材料精度必须阅读实际逐来源预测表，不能用绿灯代替。修改代码或输入前先核对协议与SHA，保留原始失败和冻结结果。
'''
    dest=OUT/'delivery/README.md';dest.write_text(guide)
    e.write(OUT/'delivery/documentation_audit.json',dict(passed=True,created_utc=e.now(),presets=len(rows),models=modelcount,
        unqualified_presets=unqualified,manifest_sha256=e.sha(BUNDLE/'manifest.json'),guide_sha256=e.sha(dest),
        catalogue_sha256=e.sha(OUT/'delivery/preset_catalogue.csv'),no_new_predictions_or_fits=True,
        named_primary_not_guaranteed_best=True,goal_complete=False,code_sha256=e.sha(Path(__file__))))
    print('Saved actual pool predictor guide/catalogue',len(rows),modelcount,'unqualified',unqualified,flush=True)


if __name__=='__main__':main()
