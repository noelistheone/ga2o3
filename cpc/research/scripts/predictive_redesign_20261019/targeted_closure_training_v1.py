"""Six hinge fits and ten label-isolated inner inventory fits; no outer tuning."""
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,time,os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase34_learning_v1 as overlay
import targeted_closure_model_v1 as m

OUT=m.OUT/'inventory_training_v1'
PREVIOUS=c.BASE/'proposal_followup_20261021_v1'


def fit_key(method,train):return c.digest(dict(method=method,train=sorted(train)))
def fit_path(method,train):return OUT/'fits'/(fit_key(method,train)+'.json')


def fit_task(task):
    method,train=task;rows=overlay.subset(train);starts,warm=m.initializers(train,method)
    lo,hi=overlay.bounds('OverlayShareProcess');cache={};calls=0;failures=[];clock=time.perf_counter()
    def calc(w):
        nonlocal calls
        key=tuple(w)
        if key not in cache:
            calls+=1
            try:q=m.loss_arrays(method,w,rows)
            except (ValueError,RuntimeError,ArithmeticError) as exc:failures.append(repr(exc));raise
            cache.clear();cache[key]=q
        return cache[key]
    def run(w,cap):
        try:
            q=least_squares(lambda v:calc(v)[0],w,jac=lambda v:calc(v)[1],bounds=(lo,hi),x_scale='jac',
                max_nfev=cap,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            return dict(parameters=q.x.tolist(),loss=float(q.fun@q.fun),success=bool(q.success),
                nfev=int(q.nfev),message=q.message,cap=cap)
        except (ValueError,RuntimeError,ArithmeticError) as exc:
            return dict(parameters=None,loss=None,success=False,error=repr(exc),cap=cap)
    trials=[run(w,80) for w in starts]
    if any(not q['success'] for q in trials):
        finite=[q for q in trials if q['loss'] is not None]
        more=[min(finite,key=lambda q:q['loss'])['parameters']] if finite else []
        trials.extend(run(w,160) for w in [*starts,*more])
    finite=[q for q in trials if q['loss'] is not None];best=min(finite,key=lambda q:q['loss']) if finite else None
    record=dict(created_utc=c.e.now(),method=method,train_units=list(train),initializer=warm,
        parameters=None if best is None else best['parameters'],selected_success=bool(best and best['success']),
        training_valid=False,trials=trials,objective_calls=calls,failures=failures,seconds=time.perf_counter()-clock,
        config=dict(gamma=0.,alpha=m.ALPHA),parameter_count=6)
    if best:
        _,_,q=calc(np.array(best['parameters']));record.update(training_valid=bool(q['valid'].all()),
            training_values=q['values'].tolist(),training_case_ids=rows.case_id.tolist())
    c.write(fit_path(method,train),record)
    print('Closing inventory',method,len(train),'groups',record['selected_success'],record['training_valid'],
        'seconds',round(record['seconds'],1),flush=True)
    return record


def predictions(rows,fit,variant,cohort):
    out=[]
    for row in rows.itertuples():
        item=dict(case_id=row.case_id,sample_id=row.sample_id,family=row.family,related_cluster=row.related_cluster,
            variant=variant,cohort=cohort,primary_common20=bool(getattr(row,'primary_common20',False)),budget=0,role='query',
            mobility_eligible=bool(row.mobility_eligible),observed=float(row.target),scenario_index=float(row.scenario_index),
            observed_log_mu=float(np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None,
            observed_log_sigma=float(np.log10(c.learning.E_CHARGE)+row.target+np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None,
            logdose=float(np.log10(row.incorporated_cm3)))
        try:
            if not fit['training_valid'] or not fit['selected_success']:raise ValueError('Unqualified minimum-finite-loss fit')
            p=m.model(fit['method']).point(m.clean(row),fit['parameters'],require_observation=False);v=m.point_values(p)
            item.update(predicted=float(v[0]),predicted_log_mu=float(v[1]),predicted_log_sigma=float(v[2]),
                numeric=bool(np.isfinite(v).all()),valid=bool(p.diagnostics['observation_domain_valid']),
                local_logdose_slope=float(p.input_jacobian[0,p.input_names.index('log10_dose_cm3')]),
                share_dose_coefficient=float(fit['parameters'][3]))
        except (ValueError,RuntimeError,ArithmeticError) as exc:
            item.update(numeric=False,valid=False,predicted=None,predicted_log_mu=None,predicted_log_sigma=None,error=repr(exc))
        out.append(item)
    return out


def main():
    assert not (OUT/'protocol_before.json').exists()
    gate=m.OUT/'gates_v2/gates.json';assert c.document(gate)['passed'];c.verify_hashes(c.document(gate.parent/'protocol_before.json'))
    units=list(c.units());trains=[tuple(u for u in units if u!=held) for held in units]+[tuple(units)]
    tasks=[('DiluteHinge',t) for t in trains]+[('LinearShared',t) for t in combinations(units,3)]
    inputs=[gate,c.OLD/'data/scenarios49.csv',c.BASE/'external_rate_cohort_v2/scenarios24.csv',
        PREVIOUS/'overlay_nH_controls_v1/predictions.csv',PREVIOUS/'overlay_nH_controls_v1/external_schema_replay_v2/predictions.csv']
    for method,train in tasks:
        fit,path=c.fit(train,'ShareProcess205',dict(gamma=0.,alpha=m.ALPHA))
        assert fit['training_valid'] and fit['selected_success'];inputs.append(path)
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),pid=os.getpid(),workers=4,tasks=tasks,
        primary='fixed threshold18.4,width.25,center18,high-dose slope1/2;6parameters;gamma0alpha.1',
        threshold_not_selected_on_outer_or_external_labels=True,external_retrospective=True,
        inner_linear_fits='ten three-group inventories for fully upstream-isolated rate-scale hyperparameter selection',
        optimizer='two starts80;anynonstop adds two starts+minimumfinite160;retain minimumfinite,refuse if selectednonstopped',
        no_broader_grid_or_tuning_after_outer_results=True,
        code_hashes=c.file_hashes([Path(__file__),Path(m.__file__),Path(c.__file__),Path(overlay.__file__),
            *list((c.ROOT/'src/sandbox').glob('*.py'))]),input_hashes=c.file_hashes(set(inputs))))
    start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fit_task,t) for t in tasks]):
            records.append(future.result());c.write(OUT/'status.json',dict(created_utc=c.e.now(),finished=len(records),planned=len(tasks),seconds=time.perf_counter()-start))
    rows=[];table=c.cases()
    for held in units:
        train=tuple(u for u in units if u!=held);fit=c.document(fit_path('DiluteHinge',train))
        rows.extend(predictions(table[table.related_cluster.eq(held)],fit,'DiluteHinge','old_cohort_heldout'))
    full=c.document(fit_path('DiluteHinge',tuple(units)))
    rows.extend(predictions(c.read(c.BASE/'external_rate_cohort_v2/scenarios24.csv'),full,'DiluteHinge','external3_conditional'))
    old=c.read(PREVIOUS/'overlay_nH_controls_v1/predictions.csv');old=old[old.cohort.eq('old_cohort_heldout')].copy()
    old['variant']=old.variant.replace({'LearnedNHInventory':'LinearShared'})
    ext=c.read(PREVIOUS/'overlay_nH_controls_v1/external_schema_replay_v2/predictions.csv').copy()
    ext['variant']=ext.variant.replace({'LearnedNHInventory':'LinearShared'})
    frame=pd.concat([pd.DataFrame(rows),old,ext],ignore_index=True);frame.to_csv(OUT/'predictions.csv',index=False)
    summaries=[]
    for (cohort,variant),p in frame.groupby(['cohort','variant'],sort=True):
        for scope in (('all_query','common20') if cohort=='old_cohort_heldout' else ('query',)):
            q=p[p.primary_common20.eq(True)] if scope=='common20' else p
            expected=20 if scope=='common20' else 49 if cohort=='old_cohort_heldout' else 24
            complete=len(q)==expected
            j=dict(cohort=cohort,variant=variant,scope=scope,rows=len(q),conditions=q.sample_id.nunique(),
                complete=complete,numeric=int(q.numeric.eq(True).sum()),valid=int(q.valid.eq(True).sum()))
            for endpoint in ('nH','mu','rho'):j[endpoint+'_MAE']=c.group_score(q.to_dict('records'),endpoint,paired=endpoint!='nH',group_equal=False) if complete else None
            summaries.append(j)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    result=dict(created_utc=c.e.now(),new_fit_records=len(records),optimizer_starts=sum(len(r['trials']) for r in records),
        selected_nonstopped=sum(not r['selected_success'] for r in records),invalid_training_fits=sum(not r['training_valid'] for r in records),
        prediction_rows=len(frame),seconds=time.perf_counter()-start,no_new_material_conditions=True)
    c.write(OUT/'complete.json',result);print(result,flush=True)


if __name__=='__main__':main()
