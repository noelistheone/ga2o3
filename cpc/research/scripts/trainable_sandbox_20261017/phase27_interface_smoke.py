"""Public mixed-model interface on the exact isolated-source smoke snapshots."""
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.feasible_property_predictor import FeasiblePropertyPredictor as PoolPropertyPredictor
from sandbox.context_property_predictor_v2 import ContextPropertyPredictorV2
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.kroger_db import NPZ
import phase27_training as t
import phase2_energy as e


def main():
    out=t.OUT/'interface_smoke';assert not (out/'audit.json').exists()
    source=t.OUT/'source_smoke';tasks=json.loads((source/'tasks.json').read_text())['tasks'];lineage=json.loads((source/'snapshot_lineage.json').read_text())
    reference={r['id']:r for r in json.loads((source/'outputs.json').read_text())['records']};presets={};files=[]
    real={r['id']:json.loads((ROOT/r['fit_path']).read_text()) for r in lineage['snapshots']}
    for task in tasks:
        f=task['fit'];path=out/'bundle/models'/(f['key']+'.json');e.write(path,f)
        files.append(dict(path=str(path.relative_to(out/'bundle')),sha256=e.sha(path)))
        if task['id'] in real:
            r=real[task['id']];tr=t.parent.subset(r['scheme'],r['train_units'])
            support={element:dict(dose_cm3=[float(g.incorporated_cm3.min()),float(g.incorporated_cm3.max())],
                growth_T_C=[float(g.growth_T_C.min()),float(g.growth_T_C.max())]) for element,g in tr.groupby('element')}
        else:support={}
        presets[task['id']]=dict(model_key=f['key'],model_type='charge_feasible',method=f['method'],source_kind=task['kind'],
            source_lambdas={'0':None,'1':.1,'2':.1},training_support=support,
            scope='actual partial-training interface fixture;not final33 deployment or selected source policy')
    old=t.OUT.parent/'phase26_pool_share/delivery/predictor_bundle';manifest=json.loads((old/'manifest.json').read_text())
    oldspec=manifest['presets']['DOI::full::FixedPhysical'];key=oldspec['model_key'];f=json.loads((old/'models'/(key+'.json')).read_text())
    path=out/'bundle/models'/(key+'.json');e.write(path,f);files.append(dict(path=str(path.relative_to(out/'bundle')),sha256=e.sha(path)))
    presets['legacy_reference']=dict(oldspec,model_type='legacy')
    e.write(out/'bundle/manifest.json',dict(version=1,presets=presets,files=files,
        profile_sha256=published_coherent_profile().fingerprint,physical_reference_sha256=e.sha(NPZ),
        scope='software smoke fixtures;not full33models',goal_complete=False))
    model=PoolPropertyPredictor(out/'bundle');checks=[];points=0
    def check(name,x,tol=0.):checks.append(dict(name=name,error=float(x),tolerance=tol,passed=bool(x<=tol)))
    for task in tasks:
        if task['id']=='invalid_reference':
            try:model.predict(preset=task['id'],inputs=task['nodes'])
            except ValueError:check('invalid_fit_refused',0.)
            else:check('invalid_fit_refused',1.)
            continue
        for ep in reference[task['id']]['episodes']:
            b=ep['budget'];lam=None if not b else .1;sol=next(s for s in ep['solutions'] if s['lam']==lam)
            anchors=[dict(input_index=i,measured_nH_cm3=float(10**y)) for i,y in zip(ep['anchor_indices'],ep['anchor_labels'])]
            q=dict(preset=task['id'],inputs=task['nodes'],anchors=anchors);a=model.predict(**q)
            e.write(out/(task['id']+'_'+str(b)+'_request.json'),q);e.write(out/(task['id']+'_'+str(b)+'_response.json'),a)
            for i,(p,r) in enumerate(zip(a['predictions'],sol['records'])):
                check(task['id']+'/'+str(b)+'/'+str(i)+'/domain',float(p['valid']!=r['valid']))
                if p['valid']:
                    check(task['id']+'/'+str(b)+'/'+str(i)+'/properties',np.max(abs(np.array([p['log10_nH_cm3'],p['log10_muH_cm2Vs'],p['log10_sigma_S_cm']])-
                        np.array([r['predicted'],r['predicted_log_mu'],r['predicted_log_sigma']]))),2e-6);points+=1
            json.dumps(a,allow_nan=False)
    grow=next(task for task in tasks if task['fit']['method']=='FeasibleProcess')
    good=copy.deepcopy(grow['nodes'][0]);bad=dict(good,incorporated_cm3=1e30)
    a=model.predict(preset=grow['id'],inputs=[good,bad])
    check('partial_state/good_preserved',float(not a['predictions'][0]['numeric']))
    check('partial_state/bad_retained',float(a['predictions'][1]['numeric'] or a['predictions'][1]['valid']))
    check('partial_state/coverage_false',float(a['full_valid_coverage']))
    e.write(out/'partial_failure_response.json',a)
    q=dict(preset='legacy_reference',inputs=[good]);a=model.predict(**q)
    b=ContextPropertyPredictorV2(old).predict(preset='DOI::full::FixedPhysical',inputs=[good])
    check('legacy_factory/nH',abs(a['predictions'][0]['log10_nH_cm3']-b['predictions'][0]['log10_nH_cm3']),2e-8)
    grad=model.zero_anchor_gradient(preset=grow['id'],input=good)
    check('gradient/actual_parameter_dimension',abs(len(grad['parameter_jacobian'][0])-len(grow['fit']['parameters'])))
    check('gradient/learned_chain',float(not grad['includes_learned_input_chain']))
    e.write(out/'gradient.json',grad)
    paths=[Path(__file__),ROOT/'src/sandbox/feasible_property_predictor.py',ROOT/'src/sandbox/charge_feasible_saved_model.py',
        ROOT/'scripts/trainable_sandbox_20261017/phase27_api_runtime.py']
    result=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,replayed_points=points,
        partial_failure_retained=True,old_factory_checked=True,final33model_or_predictive_accuracy_validation=False,
        goal_complete=False,code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths})
    e.write(out/'audit.json',result);print('phase27 mixed public interface smoke',result['passed'],len(checks),points,flush=True)
    if not result['passed']:raise RuntimeError('Preserve mixed-interface failure before repair')


if __name__=='__main__':main()
