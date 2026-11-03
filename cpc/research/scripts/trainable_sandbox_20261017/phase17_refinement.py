"""Uniform numerical refinement and selection for base and matched controls."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,argparse,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.optimize import least_squares
import phase17_controls as controls
import phase2_energy as energy
study=controls.study;OUT=study.OUT
ALL_METHODS={**study.STRUCTURES,**controls.METHODS}


def raw_fits():return {**study.all_fits(),**controls.all_fits()}


def fit(families,method,config):
    f=controls.fit(families,method,config)
    p=OUT/'refined_fits'/(f['key']+'.json')
    return json.loads(p.read_text()) if p.exists() else f


study.core.fit=fit


def composite():
    result=raw_fits()
    for p in (OUT/'refined_fits').glob('*.json'):
        f=json.loads(p.read_text());f['composite_origin']=dict(path=str(p.relative_to(ROOT)),sha256=energy.sha(p));result[f['key']]=f
    return result


def worker(old):
    p=OUT/'refined_fits'/(old['key']+'.json')
    if p.exists():return old['key']
    method=old['method'];control=method in controls.METHODS
    tr=controls.training_data(old['train_families'],method) if control else study.training_data(old['train_families'],method)
    y=tr.target.to_numpy(float);sw=np.sqrt(tr.loss_weight.to_numpy(float));spec=controls.specification if control else study.specification
    lo,hi,starts=spec(method);starts=list(starts)+([old['parameters']] if old['parameters'] is not None else [])
    trials=[];failures=[];cache={};clock=time.perf_counter()
    def calc(w):
        key=tuple(w)
        if key not in cache:
            try:value=controls.activation(w,tr,controls.METHODS[method]['relation']) if control else study.forward_targets(method,w,tr)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failures.append(dict(parameters=w,error=repr(exc)));raise
            cache.clear();cache[key]=value
        return cache[key]
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
        refinement=dict(parent=old['composite_origin'],original_trials=old['trials'],additional_starts=len(starts),seconds=time.perf_counter()-clock))
    if control:
        values,_=controls.activation(parameters,study.TARGETS,controls.METHODS[method]['relation']);f['predictions'],f['property_predictions']=controls.predictions(values)
    else:f['predictions'],f['property_predictions']=study.saved_predictions(method,parameters)
    energy.write(p,f);print('refinedinventory',method,old['key'],'starts',len(starts),'stop',sum(t['success'] for t in trials),flush=True)
    return old['key']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--select',action='store_true');a=ap.parse_args()
    study.freeze();controls.freeze();records=raw_fits();assert len(records)==525
    jobs=[f for f in records.values() if not f.get('inherited') and any(not t.get('success') for t in f['trials'])]
    protocol=dict(code_sha256=energy.sha(Path(__file__)),prediction_protocol_sha256=energy.sha(OUT/'prediction_protocol.json'),
        controls_protocol_sha256=energy.sha(controls.OUT/'protocol.json'),affected=[dict(key=f['key'],**f['composite_origin']) for f in jobs],
        rule='allfailedstartfits;2original+savedbest,360sameobjective/tolerances;no outer selection filter')
    pp=OUT/'refinement_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:energy.write(pp,protocol)
    if jobs:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for _ in pool.map(worker,jobs):pass
    energy.write(OUT/'refinement_complete.json',dict(created_utc=energy.now(),affected_fits=len(jobs),
        additional_starts=sum(f['refinement']['additional_starts'] for f in composite().values() if f.get('refinement')),goal_complete=False))
    if a.select:
        for method in ALL_METHODS:study.core.select(method)
        energy.write(OUT/'selection_complete.json',dict(created_utc=energy.now(),methods=len(ALL_METHODS),goal_complete=False))


if __name__=='__main__':main()
