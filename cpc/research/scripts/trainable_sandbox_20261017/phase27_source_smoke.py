"""Fresh source-worker integration on exact completed fits; no accuracy claim."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.charge_feasible_saved_model import SavedFeasibleModel as SavedPoolModel
from sandbox.kroger_db import NPZ
import phase27_prediction as p
import phase2_energy as e

OUT=p.t.OUT/'source_smoke'


def main():
    assert not (OUT/'audit.json').exists(),'Keep prior smoke results before repair'
    snapshots=json.loads((p.t.OUT/'saved_model_smoke/audit.json').read_text())['fit_snapshots'];tasks=[];refs=[]
    for i,spec in enumerate(snapshots):
        path=ROOT/spec['fit_path'];assert e.sha(path)==spec['fit_sha256'];f=json.loads(path.read_text())
        # All conditions of this training family are valid by construction.
        # This is source-inference software validation, not held-source MAE.
        family=sorted(f['train_families'])[0];nodes,mapping,episodes=p.design(family)
        key='source_smoke_'+str(i);kind='field'
        tasks.append(dict(id=key,fit={k:f[k] for k in p.MINIMAL if k in f},kind=kind,nodes=nodes,episodes=episodes))
        refs.append(dict(id=key,family=family,fit_path=spec['fit_path'],fit_sha256=spec['fit_sha256']))
    path=ROOT/snapshots[0]['fit_path'];f=json.loads(path.read_text())
    f=dict(f,key='intentional_invalid_training_fixture',training_valid=False)
    nodes,mapping,episodes=p.design(sorted(f['train_families'])[0])
    tasks.append(dict(id='invalid_reference',fit={k:f[k] for k in p.MINIMAL if k in f},kind='offset',nodes=nodes,episodes=episodes))
    packet=OUT/'tasks.json';output=OUT/'outputs.json'
    e.write(packet,dict(tasks=tasks,lambdas=p.LAM,physical_reference_sha256=e.sha(NPZ)))
    e.write(OUT/'snapshot_lineage.json',dict(snapshots=refs,invalid_reference_path=str(path.relative_to(ROOT)),invalid_reference_sha256=e.sha(path),
        invalid_fixture_is_synthetic_flag_change=True,scope='completed4unit training models and allowed source anchors;software integration only',goal_complete=False))
    run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase27_source_runtime.py'),str(packet),str(output)],cwd=ROOT,capture_output=True,text=True)
    (OUT/'execution.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode()
    data=json.loads(output.read_text());observed={r['id']:r for r in data['records']};checks=[];replays=0;objectives=0
    def check(name,x,tol=0.):checks.append(dict(name=name,error=float(x),tolerance=tol,passed=bool(x<=tol)))
    check('forbidden_result_reads',len(data['denied_reads']));check('worker_query_labels',float(data['runtime_query_labels']))
    for task in tasks:
        actual=observed[task['id']]
        if task['id']=='invalid_reference':
            check('invalid/no_point_evaluations',actual['actual_physical_point_calls'])
            check('invalid/not_servable',sum(r['numeric'] or r['valid'] for ep in actual['episodes'] for s in ep['solutions'] for r in s['records']))
            continue
        m=SavedPoolModel(task['fit']);rows=[SimpleNamespace(**v) for v in task['nodes']]
        base=[r['predicted'] for r in actual['episodes'][0]['solutions'][0]['records']]
        for ep in actual['episodes']:
            expected=task['episodes'][ep['budget']]
            check(task['id']+'/'+str(ep['budget'])+'/anchor_indices',float(ep['anchor_indices']!=expected['anchor_indices']))
            check(task['id']+'/'+str(ep['budget'])+'/anchor_labels',max([0.]+[abs(a-b) for a,b in zip(ep['anchor_labels'],expected['anchor_labels'])]))
            for solution in ep['solutions']:
                if not solution['valid']:continue
                z=solution['raw_shift'];values=[]
                for i,(row,record) in enumerate(zip(rows,solution['records'])):
                    q=m.point(row,source_kind=task['kind'],shift=z,require_observation=False)
                    check(task['id']+'/'+str(ep['budget'])+'/'+str(i)+'/domain',float(record['valid']!=q.valid))
                    if record['numeric']:
                        err=np.max(abs(np.array([record['predicted'],record['predicted_log_mu'],record['predicted_log_sigma']])-
                            np.array([q.log_nH,q.log_muH,q.log_sigma])))
                        check(task['id']+'/'+str(ep['budget'])+'/'+str(i)+'/point_replay',err,2e-6);replays+=1
                        values.append(record['predicted'])
                if ep['budget'] and len(values)==len(rows):
                    mean=float(np.mean((np.array(values)[ep['anchor_indices']]-ep['anchor_labels'])**2))
                    penalty=float(np.mean((np.array(values)-base)**2));lam=solution['lam']
                    expected=mean+(0. if lam is None else lam*penalty)
                    check(task['id']+'/'+str(ep['budget'])+'/'+str(lam)+'/mean_objective',abs(solution['objective']-expected),2e-8);objectives+=1
    check('actual_valid_replay_points',float(replays==0))
    paths=[Path(__file__),Path(__file__).parent/'phase27_source_runtime.py',ROOT/'src/sandbox/charge_feasible_saved_model.py']
    result=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,replayed_points=replays,mean_objectives_checked=objectives,
        actual_training_snapshots=True,source_excluded_accuracy_test=False,final33_model_delivery=False,
        query_label_or_training_table_files_read=False,goal_complete=False,
        code_hashes={str(q.relative_to(ROOT)):e.sha(q) for q in paths})
    e.write(OUT/'audit.json',result);print('phase27 fresh source integration',result['passed'],len(checks),replays,objectives,flush=True)
    if not result['passed']:raise RuntimeError('Retain isolated source-integration failure before repair')


if __name__=='__main__':main()
