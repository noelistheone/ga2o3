"""Check unsupported dopant/input rejection in both physical and direct APIs."""
from pathlib import Path
import sys,json,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.logodds_predictor import LogOddsResearchPredictor
from sandbox.logodds_predictor_v2 import CheckedLogOddsResearchPredictor
import phase2_energy as energy
OUT=energy.BASE/'phase18_logodds_prediction'


def main():
    folder=OUT/'research_runtime_v2';tasks=json.loads((folder/'tasks.json').read_text());checks=[]
    for method in ('Logit_inventory_original_RT','Logit_add_RT'):
        task=next(t for t in tasks if t['task_id'].startswith(method+'::') and t['task_id'].endswith('::0'))
        b=json.loads(Path(task['bundle']).read_text());p=CheckedLogOddsResearchPredictor(b);raw=LogOddsResearchPredictor(b);r=task['points'][0]['input'];cal=dict(kind='offset',shift=0.)
        a=p.predict(r,cal);q=raw.predict(r,cal);err=max(abs(a['values'][0]-q['values'][0]),max(abs(np.array(a['transport_values'])-q['transport_values'])))
        checks.append(dict(name='unchanged_supported_predictor',method=method,error=err,passed=err<2e-6))
        mutations=[('element','Ce'),('incorporated_cm3',0.),('incorporated_cm3',np.nan),('Tobs_K',0.),('thermal_cap_K',500.),('H_known',True)]
        for field,val in mutations:
            x=dict(r);x[field]=val;rejected=False
            try:p.predict(x,cal)
            except ValueError:rejected=True
            checks.append(dict(name='unsupported_input_rejected',method=method,field=field,passed=rejected))
        for c in (dict(kind='typo',shift=0.),dict(kind='logodds',shift=7.)):
            rejected=False
            try:p.predict(r,c)
            except ValueError:rejected=True
            checks.append(dict(name='invalid_source_rejected',method=method,passed=rejected))
        for lam in (-1.,-np.inf,np.nan):
            rejected=False
            try:p.calibrate([],kind='offset',lam=lam)
            except ValueError:rejected=True
            checks.append(dict(name='invalid_regularization_rejected',method=method,passed=rejected))
    result=dict(passed=all(c['passed'] for c in checks),checks=checks,code_sha256=energy.sha(Path(__file__)),
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (ROOT/'src/sandbox/logodds_predictor.py',ROOT/'src/sandbox/logodds_predictor_v2.py')},
        scope='domain checks,notnew material accuracy;delegate unchangednumericalmapping verifiedin9660row replay',goal_complete=False)
    energy.write(OUT/'predictor_scope_gates.json',result);print(json.dumps(result,indent=2));assert result['passed']


if __name__=='__main__':main()
