"""Test the public saved interface against actual isolated source-operator outputs."""
from pathlib import Path
import sys,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.context_property_predictor import ContextPropertyPredictor
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.kroger_db import NPZ
import phase2_energy as e


def main():
    out=e.BASE/'phase25_shared_process/interface_smoke';source=e.BASE/'phase25_shared_process/source_smoke_v3'
    tasks=json.loads((source/'tasks.json').read_text())['tasks'];observed={r['id']:r for r in json.loads((source/'outputs.json').read_text())['records']}
    presets={};files=[]
    for task in tasks:
        f=task['fit'];p=out/'bundle/models'/(f['key']+'.json')
        if not p.exists():e.write(p,f)
        if not any(r['path']==str(p.relative_to(out/'bundle')) for r in files):files.append(dict(path=str(p.relative_to(out/'bundle')),sha256=e.sha(p)))
        presets[task['id']]=dict(model_key=f['key'],source_kind=task['kind'],source_lambdas={'0':None,'1':.1,'2':.1},
            training_support={s:dict(dose_cm3=[1e16,5e20],growth_T_C=[450,850]) for s in ('Si','Sn')},
            scope='interface smoke only;not final full33 model or selected source policy')
    e.write(out/'bundle/manifest.json',dict(version=1,presets=presets,files=files,profile_sha256=published_coherent_profile().fingerprint,physical_reference_sha256=e.sha(NPZ)))
    predictor=ContextPropertyPredictor(out/'bundle');checks=[];replays=0
    def check(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
    for task in tasks:
        for ep in observed[task['id']]['episodes']:
            b=ep['budget'];lam=None if b==0 else .1
            sol=next(s for s in ep['solutions'] if s['lam']==lam)
            anchors=[dict(input_index=i,measured_nH_cm3=float(10**y)) for i,y in zip(ep['anchor_indices'],ep['anchor_labels'])]
            request=dict(preset=task['id'],inputs=task['nodes'],anchors=anchors)
            p=predictor.predict(**request)
            e.write(out/f"request_{task['id']}_{b}.json",request);e.write(out/f"response_{task['id']}_{b}.json",p)
            for i,(a,c) in enumerate(zip(p['predictions'],sol['records'])):
                check(f"{task['id']}/{b}/{i}/domain",float(a['valid']!=c['valid']),0.)
                if a['valid']:
                    expected=[c['predicted'],c['predicted_log_mu'],c['predicted_log_sigma']]
                    actual=[a['log10_nH_cm3'],a['log10_muH_cm2Vs'],a['log10_sigma_S_cm']]
                    check(f"{task['id']}/{b}/{i}/replay",np.max(np.abs(np.array(expected)-actual)),2e-6);replays+=1
                    check(f"{task['id']}/{b}/{i}/derived_sigma",abs(np.log10(a['sigma_S_cm'])-np.log10(1.602176634e-19*a['nH_cm3']*a['muH_cm2Vs'])),1e-12)
    # Public zero-anchor gradients use the same actual state mapper; numerical
    # parameter/input validation is recorded in the161/131 earlier gates.
    task=tasks[1]
    grad=predictor.zero_anchor_gradient(preset=task['id'],input=task['nodes'][0])
    check('gradient_dimension',abs(len(grad['parameter_jacobian'][0])-6),0.)
    check('added_dose_chain_exported',float(not grad['includes_dg_dh_times_dh_dlogC']),0.)
    e.write(out/'zero_anchor_gradient.json',grad)
    paths=[Path(__file__),ROOT/'src/sandbox/context_property_predictor.py',ROOT/'scripts/trainable_sandbox_20261017/predict_context_properties_v1.py']
    value=dict(passed=all(r['passed'] for r in checks),checks=len(checks),rows=checks,replayed_points=replays,
        query_label_metrics_used=False,final_33model_delivery=False,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths})
    e.write(out/'audit.json',value)
    print('context public interface smoke',value['passed'],len(checks),'replayed',replays,flush=True)
    if not value['passed']:raise RuntimeError('Preserve failed interface replay before a repair')


if __name__=='__main__':main()
