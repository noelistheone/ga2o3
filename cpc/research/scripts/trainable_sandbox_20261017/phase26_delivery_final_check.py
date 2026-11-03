"""Exercise selected full-source CLI gradients and explicit input-chain replay."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,copy,subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.pool_property_predictor import PoolPropertyPredictor
import phase2_energy as e

BASE=e.BASE/'phase26_pool_share';BUNDLE=BASE/'delivery/predictor_bundle';OUT=BASE/'delivery/final_checks'


def vector(p):return np.array([p.log_nH,p.log_muH,p.log_sigma])


def main():
    assert json.loads((BASE/'delivery/cohort_replay/audit.json').read_text())['passed']
    predictor=PoolPropertyPredictor(BUNDLE);checks=[];requests=0
    def check(name,x,tol=0.):checks.append(dict(name=name,error=float(x),tolerance=tol,passed=bool(x<=tol)))
    for scheme in ('DOI','cluster'):
        for method in ('ShareEffectiveDose','ShareEffectiveProcess'):
            preset=scheme+'::full::'+method
            row=dict(element='Si',incorporated_cm3=2e17,Tobs_K=300.,growth_T_C=550.,H_known=False,H_cm3=0.,
                carbon_known=False,carbon_cm3=0.,scenario_pO2_atm=1e-5,oxygen_input_kind='explicit unknownO2 effective scenario;not measured pressure')
            stem=scheme+'_'+method;request=OUT/(stem+'_request.json');response=OUT/(stem+'_response.json')
            e.write(request,dict(preset=preset,input=row))
            run=subprocess.run([sys.executable,str(Path(__file__).parent/'predict_pool_properties_v1.py'),
                '--bundle',str(BUNDLE),'--request',str(request),'--output',str(response),'--zero-anchor-gradient'],
                cwd=ROOT,capture_output=True,text=True)
            (OUT/(stem+'_CLI.log')).write_text(run.stdout+'\n'+run.stderr);run.check_returncode();requests+=1
            result=json.loads(response.read_text());model,spec=predictor.model(preset);X=np.asarray(result['input_jacobian'])
            check(stem+'/three_properties',abs(len(result['properties'])-3))
            check(stem+'/actual_method',float(result['actual_evaluation_method']!=model.fit['evaluation_method']))
            check(stem+'/actual_parameter_dimension',abs(np.array(result['parameter_jacobian']).shape[1]-len(model.fit['parameters'])))
            for coordinate,h in [('log10_dose_cm3',1e-4),('growth_T_C',1e-2)]:
                check(stem+'/'+coordinate+'/unique_input_axis',abs(result['input_names'].count(coordinate)-1))
                column=result['input_names'].index(coordinate);plus=copy.deepcopy(row);minus=copy.deepcopy(row)
                if coordinate=='log10_dose_cm3':plus['incorporated_cm3']*=10**h;minus['incorporated_cm3']/=10**h
                else:plus[coordinate]+=h;minus[coordinate]-=h
                fd=(vector(model.point(SimpleNamespace(**plus)))-vector(model.point(SimpleNamespace(**minus))))/(2*h)
                check(stem+'/'+coordinate+'/full_saved_chain',np.max(abs(fd-X[:,column])/np.maximum(1,abs(X[:,column]))),1e-4)
    nodes=[dict(row,incorporated_cm3=c) for c in (2e17,2e20)]
    req=dict(preset='cluster::full::ShareEffectiveProcess_internal',inputs=nodes,anchors=[])
    e.write(OUT/'example_prediction_request.json',req);e.write(OUT/'example_prediction_response.json',predictor.predict(**req))
    value=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,CLI_gradient_requests=requests,
        source_and_hyperparameter_refit_derivatives=False,actual_evaluation_methods_and_dimensions_checked=True,
        query_labels_used=False,no_new_fits=True,material_accuracy_validated_by_these_checks=False,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (Path(__file__),ROOT/'src/sandbox/pool_property_predictor.py')})
    e.write(OUT/'audit.json',value);print('phase26 fullmodel CLI/inputchain',value['passed'],len(checks),flush=True)
    if not value['passed']:raise RuntimeError('Retain failed fullmodel derivative/CLI checks before repair')


if __name__=='__main__':main()
