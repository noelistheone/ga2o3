"""Reproduce the recorded in-training adapter smoke on exact saved snapshots."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.pool_share_saved_model import SavedPoolModel
from sandbox.context_saved_model import INPUT_NAMES
import phase26_training as t
import phase2_energy as e


def main():
    out=t.OUT/'saved_model_smoke';first=json.loads((out/'audit.json').read_text());checks=[];points=0
    for spec in first['fit_snapshots']:
        path=ROOT/spec['fit_path'];assert e.sha(path)==spec['fit_sha256'];r=json.loads(path.read_text())
        table=t.parent.subset(r['scheme'],r['train_units']);m=SavedPoolModel(r)
        for i,x in enumerate(table.itertuples()):
            row=SimpleNamespace(**{k:getattr(x,k) for k in INPUT_NAMES});p=m.point(row,require_observation=False)
            error=abs(p.log_nH-r['selected_training_values'][i]);points+=1
            checks.append(dict(name=spec['method']+'/'+x.sample_id+'/saved_training_replay',error=error,tolerance=2e-6,passed=bool(error<=2e-6)))
            if i==0:
                a=m.offset_point(row,p,.05,require_observation=False);b=m.point(row,source_kind='offset',shift=.05,require_observation=False)
                error=float(np.max(abs(np.array([a.log_nH,a.log_muH,a.log_sigma])-np.array([b.log_nH,b.log_muH,b.log_sigma]))))
                checks.append(dict(name=spec['method']+'/offset_full_chain',error=error,tolerance=2e-8,passed=bool(error<=2e-8)))
    assert points==first['replayed_points']==70
    result=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,replayed_points=points,
        inherits_invalidfit_refusal_check=bool(first['rows'][-1]['passed']),initial_audit_sha256=e.sha(out/'audit.json'),
        scope=first['scope'],goal_complete=False,code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in [Path(__file__),ROOT/'src/sandbox/pool_share_saved_model.py']})
    e.write(out/'reproduced.json',result);print('Exact saved-smoke snapshots replayed',result['passed'],points,flush=True)
    if not result['passed']:raise RuntimeError('Preserve failed saved-adapter replay before repair')


if __name__=='__main__':main()
