"""Synthetic-label, input-design and cache checks of the new deployment contract."""
from pathlib import Path
import sys,json,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.functional_predictor import DopedPropertyPredictor
from sandbox.kroger_db import load
from sandbox.relative_predictor_v2 import SavedModel,DopedPropertyPredictor as ParentPredictor
from sandbox.coherent_fermi import published_coherent_profile
import phase21_export as export
s=export.s;OUT=s.OUT/'api_gates'


def main():
    plans=json.loads((s.OUT/'request_plans.json').read_text());selection={}
    for variant in ('FieldCurveSafe','OriginalHall','DirectRidge'):
        p=next(p for p in plans if p['variant']==variant and p['held'] is None)
        selection[variant]=[dict(p,budget=b,selected_source=dict(prior='quadratic',lam=None if b==0 else 0.)) for b in (0,1,2)]
    bundle=OUT/'synthetic_bundle';export.build(bundle,selection,grade='syntheticAPIgate;forcedlambda0;notselecteddeployment')
    db=load();predictor=DopedPropertyPredictor(db,bundle);checks=[]
    nodes=[dict(dopant='Si',incorporated_cm3=C,growth_T_C=550.) for C in (1e18,1e19,1e20)]
    def check(name,error,tolerance):checks.append(dict(test=name,error=float(error),tolerance=tolerance,passed=bool(error<tolerance)))
    manifest=json.loads((bundle/'manifest.json').read_text())
    for method in selection:
        key=manifest['variants'][method]['model_key'];record=json.loads((bundle/'models'/f'{key}.json').read_text());model=SavedModel(record,profile=published_coherent_profile(),db=db)
        kind=manifest['variants'][method]['source_kind'];u=.02 if kind=='field' else .2
        physical=[predictor.parse('Si',n) for n in nodes];observed=[float(10**model.point(r,source_kind=kind,shift=u).values[0]) for r in physical]
        for budget in (0,1,2):
            anchors=[dict(input_index=i,measured_nH_cm3=observed[i]) for i in range(budget)]
            args=dict(dopant='Si',incorporated_cm3=1e19,growth_T_C=550.,method=method,source_inputs=nodes,anchors=anchors)
            p=predictor.predict(**args);expected=model.point(physical[1],source_kind=kind,shift=u if budget else 0.)
            check('synthetic_prediction_recovery_'+method+str(budget),abs(p['log10_nH_cm3']-expected.values[0]),2e-5)
            if budget:
                args['incorporated_cm3']=1e18;args['observation_T_K']=250.;q=predictor.predict(**args)
                check('query_temperature_independent_calibration_'+method+str(budget),abs(q['raw_source_coordinate']-p['raw_source_coordinate']),1e-12)
                check('cache_reused_for_other_query_'+method+str(budget),float(not q['calibration_cache_hit']),.5)
                perm=[2,0,1];moved=[nodes[i] for i in perm];movedanchors=[dict(input_index=perm.index(a['input_index']),measured_nH_cm3=a['measured_nH_cm3']) for a in anchors]
                r=predictor.predict('Si',1e19,growth_T_C=550.,method=method,source_inputs=moved,anchors=movedanchors)
                check('source_input_permutation_'+method+str(budget),abs(r['log10_nH_cm3']-p['log10_nH_cm3']),2e-5)
    parent=ParentPredictor(db,export.PARENT)
    for method in manifest['energy_methods']:
        p=predictor.predict('Si',1e19,method=method);q=parent.predict('Si',1e19,method=method)
        check('legacy_route_'+method,abs(p['log10_nH_cm3']-q['log10_nH_cm3']),2e-6)
    bad=[dict(dopant='Eu',incorporated_cm3=1e19),dict(dopant='Si',incorporated_cm3=-1.),
        dict(dopant='Si',incorporated_cm3=1e19,H_known=True,H_cm3=0.),
        dict(dopant='Si',incorporated_cm3=1e19,anchors=[dict(input_index=0,measured_nH_cm3=1e18)]),
        dict(dopant='Si',incorporated_cm3=1e19,source_inputs=nodes,anchors=[dict(input_index=9,measured_nH_cm3=1e18)]),
        dict(dopant='Si',incorporated_cm3=1e19,source_inputs=nodes,anchors=[dict(input_index=0,measured_nH_cm3=1e18)]*2),
        dict(dopant='Si',incorporated_cm3=2e19,source_inputs=nodes,anchors=[dict(input_index=0,measured_nH_cm3=1e18)]),
        dict(dopant='Si',incorporated_cm3=1e19,method='DirectRidge'),
        dict(dopant='Si',incorporated_cm3=1e19,source_inputs=[dict(dopant='Si',incorporated_cm3=1e19,observation_T_K=200.)])]
    for i,args in enumerate(bad):
        refused=False
        try:predictor.predict(**args)
        except ValueError:refused=True
        check('explicit_contract_refusal_'+str(i),float(not refused),.5)
    result=dict(passed=all(r['passed'] for r in checks),checks=len(checks),rows=checks,maximum_error=max(r['error'] for r in checks),
        code_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in (Path(__file__),ROOT/'src/sandbox/functional_predictor.py',ROOT/'src/sandbox/functional_source.py')},
        synthetic_labels_only=True,material_accuracy_claim=False,goal_complete=False)
    s.energy.write(OUT/'verification.json',result);print('API gates',result['passed'],len(checks),'max',result['maximum_error'],flush=True);assert result['passed'],[r for r in checks if not r['passed']]


if __name__=='__main__':main()
