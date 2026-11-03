"""Audit actual isolated source outputs without scoring any query labels."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.context_saved_model import SavedContextModel
import phase2_energy as e


def main():
    out=e.BASE/'phase25_shared_process/source_smoke_v3'
    provenance=json.loads((out/'provenance.json').read_text())
    for name,h in provenance['code_hashes'].items():assert e.sha(ROOT/name)==h,name
    tasks=json.loads((out/'tasks.json').read_text());actual=json.loads((out/'outputs.json').read_text())
    assert not actual['denied_reads'] and not actual['runtime_query_labels']
    checks=[]
    def check(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
    index={r['id']:r for r in actual['records']};replays=0;objectives=0
    for task in tasks['tasks']:
        record=index[task['id']];model=SavedContextModel(task['fit']);nodes=[SimpleNamespace(**n) for n in task['nodes']]
        assert all('target' not in n and 'observed' not in n and 'family' not in n for n in task['nodes'])
        bases=[model.point(r,require_observation=False) for r in nodes]
        for ep in record['episodes']:
            check(task['id']+f"/budget{ep['budget']}/anchor_count",abs(len(ep['anchor_labels'])-ep['budget']),0.)
            for sol in ep['solutions']:
                if not sol['valid']:continue
                delta=sol.get('response_delta',[sol['raw_shift']]*len(nodes))
                for i,(row,pred,shift) in enumerate(zip(nodes,sol['records'],delta)):
                    kind='offset' if task['kind']=='affine' else task['kind']
                    q=model.point(row,source_kind=kind,shift=shift,require_observation=False)
                    expected=np.array([q.log_nH,q.log_muH,q.log_sigma,q.log_mu_CT,q.log_sigma_CT])
                    saved=np.array([pred[k] for k in ('predicted','predicted_log_mu','predicted_log_sigma','predicted_log_mu_CT','predicted_log_sigma_CT')])
                    check(task['id']+f"/{ep['budget']}/{sol['lam']}/{i}/replay",np.max(np.abs(expected-saved)),2e-9)
                    check(task['id']+f"/{ep['budget']}/{sol['lam']}/{i}/domain",float(q.valid!=pred['valid']),0.)
                    replays+=1
                if ep['budget'] and 'objective' in sol:
                    values=np.array([p['predicted'] for p in sol['records']]);base=np.array([p.log_nH for p in bases])
                    residual=values[ep['anchor_indices']]-ep['anchor_labels']
                    expected=float(np.mean(residual**2)+(0. if sol['lam'] is None else sol['lam']*np.mean((values-base)**2)))
                    check(task['id']+f"/{ep['budget']}/{sol['lam']}/objective",abs(sol['objective']-expected),1e-9)
                    objectives+=1
    result=dict(passed=all(r['passed'] for r in checks),checks=len(checks),rows=checks,replayed_points=replays,
        verified_objectives=objectives,query_metrics_used=False,blocked_reads=0,
        physical_reference_sha256=actual['physical_reference_sha256'],
        code_hashes=dict(provenance['code_hashes'],**{str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))}),goal_complete=False)
    e.write(out/'audit.json',result)
    print('phase25 isolated smoke audit',result['passed'],len(checks),'replays',replays,'objectives',objectives,flush=True)
    if not result['passed']:raise RuntimeError('Preserve failed actual source audit before any repair')


if __name__=='__main__':main()
