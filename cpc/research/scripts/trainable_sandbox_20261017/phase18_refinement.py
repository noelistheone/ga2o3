"""Uniform failed-start refinement and final source-excluded model selection."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,copy,time,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase18_prediction as study
import phase2_energy as energy
OUT=study.OUT


def fit(families,method,config):
    original=study.fit(families,method,config);p=OUT/'refined_fits'/(original['key']+'.json')
    return json.loads(p.read_text()) if p.exists() else original


study.core.fit=fit


def composite():
    result=study.all_fits()
    for path in (OUT/'refined_fits').glob('*.json'):
        f=json.loads(path.read_text());f['composite_origin']=dict(path=str(path.relative_to(ROOT)),sha256=energy.sha(path));result[f['key']]=f
    return result


def worker(old):
    path=OUT/'refined_fits'/(old['key']+'.json')
    if path.exists():return old['key']
    method=old['method'];starts=list(study.specification(method)[2])+([old['parameters']] if old['parameters'] is not None else [])
    clock=time.perf_counter();parameters,trials,failures=study.optimize(method,old['train_families'],starts,360)
    f=copy.deepcopy(old);f.pop('composite_origin',None)
    if parameters is None:parameters=old['parameters']
    f.update(parameters=parameters,trials=trials,state_failures=failures,
        refinement=dict(parent=old['composite_origin'],original_trials=old['trials'],additional_starts=len(starts),seconds=time.perf_counter()-clock))
    f['predictions'],f['property_predictions']=study.saved_predictions(method,parameters)
    energy.write(path,f);print('refinedlogit',method,old['key'],'stop',sum(t['success'] for t in trials),'starts',len(starts),flush=True)
    return old['key']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--select',action='store_true');args=ap.parse_args()
    study.freeze();raw=study.all_fits();assert len(raw)==120
    jobs=[f for f in raw.values() if not f.get('inherited') and any(not t['success'] for t in f['trials'])]
    protocol=dict(code_sha256=energy.sha(Path(__file__)),prediction_protocol_sha256=energy.sha(OUT/'prediction_protocol.json'),
        rule='allfailed-startfits;2original+savedbest,360sameobjective/tolerances;no outerfilter',affected=[dict(key=f['key'],**f['composite_origin']) for f in jobs])
    p=OUT/'refinement_protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:energy.write(p,protocol)
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for _ in pool.map(worker,jobs):pass
    final=composite();energy.write(OUT/'refinement_complete.json',dict(created_utc=energy.now(),affected_fits=len(jobs),
        additional_starts=sum(f['refinement']['additional_starts'] for f in final.values() if f.get('refinement')),goal_complete=False))
    if args.select:
        for method in study.STRUCTURES:study.core.select(method)
        energy.write(OUT/'selection_complete.json',dict(created_utc=energy.now(),methods=8,goal_complete=False))


if __name__=='__main__':main()
