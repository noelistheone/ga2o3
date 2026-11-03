"""Actual all-training headroom/domain pilot, explicitly no held-out claim."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os,sys,time,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase33_learning_v1 as l

OUT=l.OUT/'all_training_pilot_v1'


def fit(task):
    method,config=task;key=c.digest(dict(method=method,config=config,protocol=c.e.sha(OUT/'protocol_before.json')))
    target=OUT/'fits'/(key+'.json');assert not target.exists()
    rows=l.subset(c.units());initial,warm=l.initializer(c.units(),method,config);lo,hi=l.bounds()
    clock=time.perf_counter();failures=[];calls=0;cache={};before=l.model(method).counters()
    def calc(w):
        nonlocal calls
        k=tuple(w)
        if k not in cache:
            calls+=1
            try:residual,J,q=l.loss_arrays(method,w,rows,config)
            except (ValueError,RuntimeError,ArithmeticError) as exc:
                failures.append(dict(parameters=np.asarray(w).tolist(),error=repr(exc)));raise
            cache.clear();cache[k]=(residual,J,q)
        return cache[k]
    def optimize(start,cap,stage):
        try:
            r=least_squares(lambda w:calc(w)[0],start,jac=lambda w:calc(w)[1],bounds=(lo,hi),
                x_scale='jac',max_nfev=cap,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            return dict(initial=np.asarray(start).tolist(),parameters=r.x.tolist(),loss=float(r.fun@r.fun),
                success=bool(r.success),nfev=int(r.nfev),optimality=float(r.optimality),message=r.message,
                active_bounds=r.active_mask.tolist(),stage=stage,cap=cap)
        except (ValueError,RuntimeError,ArithmeticError) as exc:
            return dict(initial=np.asarray(start).tolist(),parameters=None,loss=None,success=False,error=repr(exc),stage=stage,cap=cap)
    trials=[optimize(w,80,'initial') for w in initial]
    if any(not v['success'] for v in trials):
        finite=[v for v in trials if v['loss'] is not None]
        best=[min(finite,key=lambda v:v['loss'])['parameters']] if finite else []
        trials.extend(optimize(w,160,'uniform_refinement') for w in [*initial,*best])
    finite=[v for v in trials if v['loss'] is not None];selected=min(finite,key=lambda v:v['loss']) if finite else None
    result=dict(created_utc=c.e.now(),key=key,method=method,config=config,train_units=list(c.units()),
        training_case_ids=rows.case_id.tolist(),initializer_reference=warm,trials=trials,state_failures=failures,
        parameters=None if selected is None else selected['parameters'],selected_success=bool(selected and selected['success']),
        seconds=time.perf_counter()-clock,objective_forward_calls=calls,training_valid=False,
        no_heldout_or_predictive_gain_claim=True,retrospective_development=True,goal_complete=False)
    if selected is not None:
        residual,J,q=calc(np.asarray(selected['parameters']))
        result.update(selected_loss=float(residual@residual),training_valid=bool(q['valid'].all()),
            training_valid_rows=int(q['valid'].sum()),training_rows=len(rows),training_values=q['values'].tolist(),
            parameter_names=l.model(method).parameter_names,regularized_jacobian_rank=int(np.linalg.matrix_rank(J)))
        values=q['values'];wn=c.learning.weights(rows);wp=c.learning.weights(rows,paired_only=True);paired,rho=c.learning.observed_pair(rows)
        result.update(training_nH_MAE=float(wn@abs(values[:,0]-rows.target.to_numpy(float))),
            training_rho_MAE=float(wp[paired]@abs(-values[paired,2]-rho[paired])))
    after=l.model(method).counters();result['solver_counter_changes']={k:after[k]-before[k] for k in before}
    c.write(target,result);print('Spatial pilot',method,config,'success',result['selected_success'],
        'valid',result['training_valid'],'seconds',round(result['seconds'],1),flush=True)
    return {k:result[k] for k in ['key','method','config','selected_success','training_valid','seconds']}


def main():
    assert not (OUT/'protocol_before.json').exists()
    code=[Path(__file__),Path(l.__file__),Path(c.__file__),
        *[Path(__file__).with_name(v) for v in ['spatial_inventory_model_v1.py','spatial_poisson_v1.py',
            'spatial_local_charge_v1.py','spatial_parent_profile_v1.py','spatial_transport_v1.py']],
        *list((c.ROOT/'src/sandbox').glob('*.py'))]
    gates=c.BASE/'spatial_predictive_candidate/model_gates_v3/gates.json';assert c.document(gates)['passed']
    tasks=[(m,dict(gamma=1.,alpha=a)) for m in l.METHODS for a in [0.,.1]]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),pid=os.getpid(),workers=4,
        tasks=tasks,all_training_groups=list(c.units()),parameter_bounds=[x.tolist() for x in l.bounds()],grid_size=l.GRID_SIZE,
        optimizer='2starts80;anynonstop triggers both+bestfinite160;minimumfinite retained including nonstopped',
        all_current_training_data_used=True,heldout_data_used=False,external_cohort_used=False,
        purpose='headroom,physical domain,numerical robustness;not nested predictive validation',
        geometry_not_measured=True,GPU_used=False,GPU_authorized='GPU1only',
        code_hashes=c.file_hashes(code),input_hashes=c.file_hashes([gates,c.OLD/'data/scenarios49.csv']),goal_complete=False))
    start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(fit,tasks):
            records.append(r);c.write(OUT/'status.json',dict(created_utc=c.e.now(),returned=len(records),planned=len(tasks),records=records,goal_complete=False))
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'complete.json',dict(created_utc=c.e.now(),records=records,seconds=time.perf_counter()-start,
        no_heldout_or_predictive_gain_claim=True,goal_complete=False))


if __name__=='__main__':main()
