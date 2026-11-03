"""Exercise final CLI gradients and policy aliases on explicit input-only designs."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,subprocess,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.context_property_predictor_v2 import ContextPropertyPredictorV2 as ContextPropertyPredictor
from sandbox.context_method_policy import ValidatedContextPolicy
import phase2_energy as e

BASE=e.BASE/'phase25_shared_process';BUNDLE=BASE/'delivery/predictor_bundle';OUT=BASE/'delivery/final_checks'


def vector(p):return np.array([p.log_nH,p.log_muH,p.log_sigma])


def main():
    assert json.loads((BASE/'delivery/cohort_replay_v2/audit.json').read_text())['passed']
    assert json.loads((BASE/'method_policy/audit.json').read_text())['passed']
    predictor=ContextPropertyPredictor(BUNDLE);checks=[]
    def check(name,error,tol=0.):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
    for method,C in [('InventoryDose',2e17),('InventoryProcess',2e20)]:
        row=dict(element='Si',incorporated_cm3=C,Tobs_K=300.,H_known=False,H_cm3=0.,carbon_known=False,carbon_cm3=0.,
            growth_T_C=550.,scenario_pO2_atm=1e-5,oxygen_input_kind='explicit unknownO2 effective scenario;not a measured pressure')
        preset='cluster::full::'+method;request=dict(preset=preset,input=row)
        q=OUT/(method+'_gradient_request.json');dest=OUT/(method+'_gradient_response.json');e.write(q,request)
        run=subprocess.run([sys.executable,str(Path(__file__).parent/'predict_context_properties_v2.py'),
            '--bundle',str(BUNDLE),'--request',str(q),'--output',str(dest),'--zero-anchor-gradient'],cwd=ROOT,capture_output=True,text=True)
        (OUT/(method+'_CLI.log')).write_text(run.stdout+'\n'+run.stderr);run.check_returncode()
        g=json.loads(dest.read_text());m,_=predictor.model(preset);J=np.array(g['input_jacobian'])
        for name,h in [('log10_dose_cm3',1e-4),('growth_T_C',1e-2)]:
            assert g['input_names'].count(name)==1,(name,g['input_names'])
            column=g['input_names'].index(name)
            plus=copy.deepcopy(row);minus=copy.deepcopy(row)
            if name=='log10_dose_cm3':
                plus['incorporated_cm3']*=10**h;minus['incorporated_cm3']/=10**h
            else:plus[name]+=h;minus[name]-=h
            yp=vector(m.point(SimpleNamespace(**plus)));ym=vector(m.point(SimpleNamespace(**minus)))
            fd=(yp-ym)/(2*h);ana=J[:,column]
            check(method+'/'+name+'/complete_input_chain',np.max(abs(fd-ana)/np.maximum(1,abs(ana))),1e-4)
        check(method+'/threeproperties',abs(len(g['properties'])-3))
        check(method+'/dosechainflag',float(not g['includes_dg_dh_times_dh_dlogC']))
    policy=ValidatedContextPolicy(BUNDLE,BASE/'method_policy/predictor_policies.json')
    # These are synthetic requested conditions, with no measured query target.
    x=[dict(row,incorporated_cm3=c) for c in (2e17,2e20)]
    result=policy.predict(policy='cluster::full::BestOverall',inputs=x)
    e.write(OUT/'policy_zero_anchor_response.json',result)
    check('policy/zero_anchor',abs(result['Hall_anchor_budget']))
    check('policy/twooutputs',abs(len(result['predictions'])-2))
    try:policy.predict(policy='cluster::full::BestOverall',inputs=x,anchors=[{}, {}, {}])
    except ValueError:check('policy/overbudgetrefused',0)
    else:check('policy/overbudgetrefused',1)
    # Save a ready-to-run ordinary prediction request as well as gradients.
    e.write(OUT/'example_prediction_request.json',dict(preset='cluster::full::InventoryProcess_internal',inputs=x,anchors=[]))
    e.write(OUT/'example_prediction_response.json',predictor.predict(preset='cluster::full::InventoryProcess_internal',inputs=x))
    value=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,
        final_full33_saved_models=True,CLI_gradient_requests=2,finite_difference_input_directions=4,
        physical_new_material_validation=False,source_hyperparameters_or_shared_weights_refitted=False,
        query_target_labels_used=False,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in [Path(__file__),ROOT/'src/sandbox/context_method_policy.py']})
    e.write(OUT/'audit.json',value);print('Final saved-model CLI and policy check',value['passed'],len(checks),flush=True)
    if not value['passed']:raise RuntimeError('Retain final CLI failure before versioned repair')


if __name__=='__main__':main()
