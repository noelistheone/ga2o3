"""Matched state-rate controls using the frozen phase31 optimizer schedule.

The optimizer body is copied unchanged except method-specific bounds/initializers,
rate-closure counter naming, and the printed study label. The parent SHA is bound.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,os,sys,time
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase32_learning as l
from sandbox.kroger_db import NPZ
OUT=l.OUT

def fit(task):
    train,method,config=task;train=tuple(sorted(train))
    key,path=l.fit_path(train,method,config)
    assert not path.exists(),'Inspect prior fit;never silently overwrite'
    rows=l.subset(train);wn=c.learning.weights(rows)
    paired,rho=c.learning.observed_pair(rows)
    initial,warm=l.initializer(train,method,config)
    before=l.model(method).counters()
    reference_before=l.reference_model().counters()
    clock=time.perf_counter();calls=0;model_calls=0;reference_calls=0
    continuations=0;failures=[];cache={}
    lo,hi=l.bounds(method)
    record=dict(key=key,method=method,config=config,train_units=list(train),
        train_families=sorted(rows.family.unique()),
        training_condition_ids=sorted(rows.sample_id.unique()),training_case_ids=rows.case_id.tolist(),
        reported_conditions=rows.sample_id.nunique(),scenario_rows=len(rows),
        paired_conditions=rows.loc[paired,'sample_id'].nunique(),
        training_nH_weights=wn.tolist(),
        training_rho_weights=c.learning.weights(rows,paired_only=True).tolist(),
        initializer_reference=warm,parameters=None,selected_success=False,training_valid=False,
        parameter_names=l.model(method).parameter_names,trials=[],state_failures=[],
        loss_observables='one Hall direction plus paired rho;two response priors;no third sigma label',
        scenario_rows_not_independent_samples=True,goal_complete=False)
    def calc(w):
        nonlocal calls,model_calls,reference_calls,continuations
        current=tuple(w)
        if current not in cache:
            calls+=1
            try:
                q=l.forward(method,w,rows,penalty=config['alpha']>0)
                model_calls+=q['actual_state_rate_model_calls']
                reference_calls+=q['actual_reference_model_calls']
                continuations+=int((~q['valid']).sum())
                residual,J,_=l.loss_arrays(method,w,rows,config,calculated=q)
            except (ValueError,RuntimeError,ArithmeticError) as exc:
                failures.append(dict(parameters=np.asarray(w).tolist(),error=repr(exc)));raise
            cache.clear();cache[current]=(residual,J,q)
        return cache[current]
    def optimize(start,cap,stage):
        try:
            result=least_squares(lambda w:calc(w)[0],start,jac=lambda w:calc(w)[1],
                bounds=(lo,hi),x_scale='jac',max_nfev=cap,
                ftol=1e-7,xtol=1e-7,gtol=1e-7)
            return dict(initial=np.asarray(start).tolist(),parameters=result.x.tolist(),
                loss=float(result.fun@result.fun),success=bool(result.success),
                nfev=int(result.nfev),optimality=float(result.optimality),
                message=result.message,active_bounds=result.active_mask.tolist(),
                stage=stage,cap=cap)
        except (ValueError,RuntimeError,ArithmeticError) as exc:
            return dict(initial=np.asarray(start).tolist(),parameters=None,
                loss=None,success=False,error=repr(exc),stage=stage,cap=cap)
    trials=[optimize(w,120,'initial') for w in initial]
    if any(not trial['success'] for trial in trials):
        finite=[trial for trial in trials if trial['loss'] is not None]
        best=[min(finite,key=lambda v:v['loss'])['parameters']] if finite else []
        trials += [optimize(w,360,'uniform_refinement') for w in [*initial,*best]]
    finite=[trial for trial in trials if trial['loss'] is not None]
    chosen=min(finite,key=lambda v:v['loss']) if finite else None
    record.update(trials=trials,state_failures=failures,new_optimizer_starts=len(trials),
        parameters=chosen['parameters'] if chosen else None,
        selected_loss=chosen['loss'] if chosen else None,
        selected_success=bool(chosen and chosen['success']))
    if chosen:
        residual,J,q=calc(np.asarray(chosen['parameters']))
        v=q['values']
        record.update(training_valid=bool(q['valid'].all()),
            selected_training_values=v.tolist(),selected_training_valid=q['valid'].tolist(),
            training_nH_MAE=float(wn@abs(v[:,0]-rows.target.to_numpy(float))),
            training_rho_MAE=float(c.learning.weights(rows,paired_only=True)[paired]@abs(-v[paired,2]-rho[paired])),
            observation_surrogate_jacobian_rank=int(np.linalg.matrix_rank(
                J[:len(rows)+(int(paired.sum()) if config['gamma']>0 else 0)])),
            regularized_surrogate_jacobian_rank=int(np.linalg.matrix_rank(J)),
            surrogate_rank_is_not_measured_information_rank=True)
    after=l.model(method).counters();reference_after=l.reference_model().counters()
    record.update(seconds=time.perf_counter()-clock,objective_forward_calls=calls,
        actual_state_rate_model_calls=model_calls,actual_reference_model_calls=reference_calls,
        observation_continuation_points=continuations,
        actual_transport_integral_calls=2*model_calls+reference_calls,
        solver_counter_changes={k:after[k]-before[k] for k in before},
        reference_solver_counter_changes={k:reference_after[k]-reference_before[k] for k in reference_before})
    c.write(path,record)
    print('fit32',key,method,config,'groups',len(train),'sec',round(record['seconds'],2),
        'valid',record['training_valid'],'stopped',record['selected_success'],flush=True)
    return dict(key=key,method=method,selected_success=record['selected_success'],
        training_valid=record['training_valid'],seconds=record['seconds'],
        optimizer_starts=record['new_optimizer_starts'])


def declare():
    path=OUT/'training_protocol.json'
    assert not path.exists(),'Inspect prior protocol;repair by version'
    gate=OUT/'gates_v1/gates.json';g=c.document(gate);assert g['passed']
    c.verify_hashes(g)
    parent=c.BASE/'phase31_rate_redesign/training_protocol.json'
    c.verify_hashes(c.document(parent))
    warmfiles=[]
    for train in c.learning.contexts():
        for config in l.configs():
            f,p=c.fit(train,'ShareProcess205',config)
            assert f['training_valid'] and f['selected_success'];warmfiles.append(p)
    code=[Path(__file__),Path(l.__file__),Path(c.__file__),
          Path(__file__).with_name('phase31_learning.py'),
          Path(__file__).with_name('phase31_training.py'),
          Path(__file__).with_name('phase32_source_runtime.py')]
    code+=list((c.ROOT/'src/sandbox').glob('*.py'))
    files=[gate,parent,NPZ,c.OLD/'data/conditions35.csv',c.OLD/'data/scenarios49.csv',
           c.OLD/'data/episodes35.csv',c.ROOT/'docs/ML_SANDBOX_STATE_RATE_CONTROL_20261020.md',*warmfiles]
    value=dict(created_utc=c.e.now(),purpose='strong state-rate controls for spectral-shape and response-regularization attribution',
        methods=l.METHODS,configs=l.configs(),training_contexts=[list(v) for v in c.learning.contexts()],
        planned_fit_records=576,nonlinear_fit_records=576,
        primary_comparison='ChargeRatioTau against ChargeRatioSpectrum;all four controls and all0/1/2budgets reported',
        state_feature='same predicted absolute-charge/free-electron ratio,chemical dose,or complex-center density',
        parameters={m:len(l.model(m).parameter_names) for m in l.METHODS},
        global_control_has7_actual_parameters_other_controls8=True,
        parameter_bounds={m:[v.tolist() for v in l.bounds(m)] for m in l.METHODS},
        parameter_ranges_are_numerical_hypotheses_not_measured_limits=True,
        optimizer='identical phase31two starts120;anynonstop triggers original2+bestfinite360;minimum finite retained;nonstopped not served',
        objective='identical nH+gamma*rho+alpha*(reference-nH response and same-stock mu response) SSE and weights',
        warm_start='same frozen ShareProcess205 stock/context/gamma/alpha;[-3,0]or[0,1] for state controls,[-3]or[0] global',
        inherited_warm_start_cost_linked_not_zero=True,outer_labels_for_selection=False,
        reported_conditions=35,DOIs=7,related_groups=5,scenario_rows=49,paired_conditions=11,
        retrospective=True,independent_blind_validation=False,GPU_used=False,new_DFT_MD=0,new_material_measurements=0,
        no_material_mechanism_identification=True,code_hashes=c.file_hashes(set(code)),
        input_hashes=c.file_hashes(set(files)),goal_complete=False)
    c.write(path,value);c.verify_hashes(value);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();assert args.workers==4
    assert not (OUT/'training_started.json').exists(),'Inspect original PID;never duplicate on timeout'
    protocol=declare()
    tasks=[(train,method,config) for train in c.learning.contexts() for method in l.METHODS for config in l.configs()]
    assert len(tasks)==576
    c.write(OUT/'training_started.json',dict(created_utc=c.e.now(),pid=os.getpid(),workers=args.workers,planned_fit_records=576))
    clock=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(fit,tasks):
            records.append(record)
            c.write(OUT/'training_status_current.json',dict(created_utc=c.e.now(),returned_records=len(records),planned_fit_records=576,
                selected_nonstopped=sum(not r['selected_success'] for r in records),invalid_training_fits=sum(not r['training_valid'] for r in records),
                actual_optimizer_starts=sum(r['optimizer_starts'] for r in records),goal_complete=False))
    c.verify_hashes(protocol)
    c.write(OUT/'training_complete.json',dict(created_utc=c.e.now(),fit_records=len(records),
        selected_nonstopped=sum(not r['selected_success'] for r in records),invalid_training_fits=sum(not r['training_valid'] for r in records),
        actual_optimizer_starts=sum(r['optimizer_starts'] for r in records),wall_seconds=time.perf_counter()-clock,
        worker_fit_seconds=sum(r['seconds'] for r in records),new_material_samples=0,goal_complete=False))
    print('State-rate control training complete;held-out validation still required',flush=True)

if __name__=='__main__':main()
