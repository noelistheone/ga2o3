"""Uniform numerical follow-up for every phase16 failed-start fit, if any."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,argparse,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.optimize import least_squares
import phase16_prediction as study
import phase2_energy as energy

OUT=study.OUT;BASE_FIT=study.fit


def affected():
    records=study.all_fits();assert len(records)==180
    return [f for f in records.values() if not f['inherited'] and any(not t.get('success') for t in f['trials'])]


def fit(families,method,config):
    f=BASE_FIT(families,method,config)
    path=OUT/'refined_fits'/(f['key']+'.json')
    return json.loads(path.read_text()) if path.exists() else f


study.core.fit=fit


def composite():
    records=study.all_fits()
    for p in (OUT/'refined_fits').glob('*.json'):
        f=json.loads(p.read_text());f['composite_origin']=dict(path=str(p.relative_to(ROOT)),sha256=energy.sha(p));records[f['key']]=f
    return records


def worker(old):
    dest=OUT/'refined_fits'/(old['key']+'.json')
    if dest.exists():return old['key']
    m=study.mapper(old['method']);tr=study.training_data(old['train_families'],old['method'])
    y=tr.target.to_numpy(float);sw=np.sqrt(tr.loss_weight.to_numpy(float));lo,hi,starts=study.core.specification(old['method'])
    if old['parameters'] is not None:starts=list(starts)+[old['parameters']]
    trials=[];failures=[];cache={};clock=time.perf_counter();before=m.counters()
    def calc(w):
        k=tuple(w)
        if k not in cache:
            try:result=study.forward_targets(m,w,tr)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failures.append(dict(parameters=w,error=repr(exc)));raise
            cache.clear();cache[k]=result
        return cache[k]
    for initial in starts:
        try:
            op=least_squares(lambda w:(calc(w)[0]-y)*sw,initial,jac=lambda w:calc(w)[1]*sw[:,None],bounds=(lo,hi),
                max_nfev=360,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            trials.append(dict(initial=initial,parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                nfev=int(op.nfev),njev=int(op.njev),optimality=float(op.optimality),active_mask=op.active_mask,message=op.message))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            trials.append(dict(initial=initial,parameters=initial,loss=None,success=False,error=repr(exc)))
    finite=[t for t in trials if t['loss'] is not None]
    parameters=min(finite,key=lambda t:t['loss'])['parameters'] if finite else old['parameters']
    f=copy.deepcopy(old);f.pop('composite_origin',None)
    f.update(parameters=parameters,trials=trials,state_failures=failures,
        refinement=dict(parent=old['composite_origin'],old_trials=old['trials'],additional_starts=len(starts),
            numerical_budget_only=True,seconds=time.perf_counter()-clock))
    f['predictions'],f['property_predictions']=study.saved_predictions(m,parameters)
    f['selected_loss_blocks']=[]
    if parameters is not None:
        residual=study.forward_targets(m,parameters,tr)[0]-y
        for b in tr.loss_block.unique():
            mask=tr.loss_block.eq(b).to_numpy();f['selected_loss_blocks'].append(dict(block=b,weighted_SSE=float(tr.loss_weight.to_numpy()[mask]@(residual[mask]**2))))
    f['refinement_counters']={k:m.counters()[k]-before[k] for k in before}
    energy.write(dest,f)
    print('refined',old['method'],old['key'],'starts',len(starts),'stop',sum(bool(t.get('success')) for t in trials),flush=True)
    return old['key']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--select',action='store_true');a=ap.parse_args()
    study.freeze();jobs=affected();protocol=dict(code_sha256=energy.sha(Path(__file__)),
        prediction_protocol_sha256=energy.sha(OUT/'prediction_protocol.json'),
        affected=[dict(key=f['key'],**f['composite_origin']) for f in jobs],
        rule='ALLnewfitswithanyfailedstart,2original+savedbest ifavailable,360sameobjective/tolerances;no outer selection filter')
    path=OUT/'refinement_protocol.json'
    if path.exists():assert json.loads(path.read_text())==energy.clean(protocol)
    else:energy.write(path,protocol)
    if jobs:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for _ in pool.map(worker,jobs):pass
    energy.write(OUT/'refinement_complete.json',dict(created_utc=energy.now(),affected_fits=len(jobs),
        additional_starts=sum(len(f.get('trials',[])) for f in composite().values() if f.get('refinement')),goal_complete=False))
    if a.select:
        for method in study.STRUCTURES:study.core.select(method)
        energy.write(OUT/'selection_execution_complete.json',dict(created_utc=energy.now(),methods=len(study.STRUCTURES),goal_complete=False))


if __name__=='__main__':main()
