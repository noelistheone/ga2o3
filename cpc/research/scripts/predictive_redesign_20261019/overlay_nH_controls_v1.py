"""Matched nH-only physical reference and external frozen-model controls."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,time,itertools,numpy as np,pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase34_learning_v1 as l

OUT=c.BASE/'proposal_followup_20261021_v1/overlay_nH_controls_v1'
BLOCK=c.BASE/'proposal_followup_20261021_v1/constrained_block_v1'


def fit_task(task):
    held,method=task;train=list(c.units()) if held is None else [u for u in c.units() if u!=held]
    cfg=dict(gamma=0.,alpha=.1 if method=='OverlayShareProcess' else 0.)
    rows=l.subset(train);starts,warm=l.initializer(train,method,cfg);lo,hi=l.bounds(method);cache={};calls=0
    clock=time.perf_counter()
    def calc(w):
        nonlocal calls
        key=tuple(w)
        if key not in cache:calls+=1;cache.clear();cache[key]=l.loss_arrays(method,w,rows,cfg)
        return cache[key]
    def run(w,cap):
        try:
            r=least_squares(lambda x:calc(x)[0],w,jac=lambda x:calc(x)[1],bounds=(lo,hi),x_scale='jac',max_nfev=cap,
                ftol=1e-7,xtol=1e-7,gtol=1e-7)
            return dict(parameters=r.x.tolist(),loss=float(r.fun@r.fun),success=bool(r.success),nfev=int(r.nfev),cap=cap,message=r.message)
        except (ValueError,RuntimeError,ArithmeticError) as exc:return dict(parameters=None,loss=None,success=False,error=repr(exc),cap=cap)
    trials=[run(w,80) for w in starts]
    if any(not t['success'] for t in trials):
        finite=[t for t in trials if t['loss'] is not None];extra=[min(finite,key=lambda t:t['loss'])['parameters']] if finite else []
        trials.extend(run(w,160) for w in [*starts,*extra])
    finite=[t for t in trials if t['loss'] is not None];best=min(finite,key=lambda t:t['loss']) if finite else None
    record=dict(created_utc=c.e.now(),held=held,method=method,train_units=train,trials=trials,initializer=warm,
        parameters=best['parameters'] if best else None,selected_success=bool(best and best['success']),training_valid=False,
        objective_calls=calls,seconds=time.perf_counter()-clock,config=cfg)
    if best:record['training_valid']=bool(calc(np.array(best['parameters']))[2]['valid'].all())
    c.write(OUT/'fits'/(c.digest(dict(held=held,method=method))+'.json'),record)
    print('Matched nH control',method,held,record['selected_success'],record['training_valid'],flush=True)
    return record


def predict(rows,fit,variant,cohort):
    out=[];mapper=l.model(fit['method'])
    for row in rows.itertuples():
        item=dict(case_id=row.case_id,sample_id=row.sample_id,family=row.family,related_cluster=row.related_cluster,
            variant=variant,cohort=cohort,budget=0,role='query',primary_common20=bool(getattr(row,'primary_common20',False)),
            mobility_eligible=bool(row.mobility_eligible),observed=float(row.target),scenario_index=row.scenario_index,
            observed_log_mu=float(np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None,
            observed_log_sigma=float(np.log10(c.learning.E_CHARGE)+row.target+np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None)
        try:
            clean=row._replace(target=np.nan,mu_cm2Vs=np.nan,carrier_cm3=np.nan,observed_log_nH=np.nan)
            p=mapper.point(clean,fit['parameters'],require_observation=False);v=np.r_[p.values[0],p.transport_values]
            item.update(predicted=float(v[0]),predicted_log_mu=float(v[1]),predicted_log_sigma=float(v[2]),
                numeric=bool(np.isfinite(v).all()),valid=bool(p.diagnostics['observation_domain_valid']))
        except (ValueError,RuntimeError,ArithmeticError) as exc:item.update(numeric=False,valid=False,predicted=None,predicted_log_mu=None,predicted_log_sigma=None,error=repr(exc))
        out.append(item)
    return out


def main():
    assert not (OUT/'protocol_before.json').exists()
    external=c.BASE/'external_rate_cohort_v2/scenarios24.csv'
    tasks=[(u,'OverlayReference') for u in c.units()]+[(None,'OverlayReference'),(None,'OverlayShareProcess')]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),tasks=tasks,workers=4,
        comparison='sameoverlay sameNHlabels fixedreference2 vs sharedinventory6;gamma0;no outerselectedconfig',
        inventory_outer_fits='reuse five completed block primaryfits,not retrain',
        fulltraining_models='fit original35/49 only,externalcohortnottrained;external3conditions/24scenarios alreadyseenretrospectively',
        new_reference_reason='match labelbudget/library of the new block primary before attributing gains',
        code_hashes=c.file_hashes([Path(__file__),Path(l.__file__),Path(c.__file__)]),
        input_hashes=c.file_hashes([external,c.OLD/'data/scenarios49.csv',BLOCK/'complete.json'])))
    with ProcessPoolExecutor(max_workers=4) as pool:records=list(pool.map(fit_task,tasks))
    predictions=[]
    for fit in records:
        if not fit['selected_success'] or not fit['training_valid']:continue
        if fit['held'] is not None:
            table=c.cases();table=table[table.related_cluster.eq(fit['held'])]
            predictions.extend(predict(table,fit,'FixedNHReference','old_cohort_heldout'))
        else:
            table=c.read(external)
            predictions.extend(predict(table,fit,'FixedNHReference' if fit['method']=='OverlayReference' else 'LearnedNHInventory','external3_conditional'))
    for f in (BLOCK/'primary_fits').glob('*.json'):
        fit=c.document(f)
        if not fit['selected_success'] or not fit['training_valid']:continue
        fit=dict(fit,method='OverlayShareProcess');table=c.cases();table=table[table.related_cluster.eq(fit['held'])]
        predictions.extend(predict(table,fit,'LearnedNHInventory','old_cohort_heldout'))
    frame=pd.DataFrame(predictions);frame.to_csv(OUT/'predictions.csv',index=False);summary=[]
    for cohort,variant in itertools.product(frame.cohort.unique(),frame.variant.unique()):
        for scope in (('all_query','common20') if cohort=='old_cohort_heldout' else ('query',)):
            q=frame[frame.cohort.eq(cohort)&frame.variant.eq(variant)]
            if scope=='common20':q=q[q.primary_common20.eq(True)]
            expected=20 if scope=='common20' else 49 if cohort=='old_cohort_heldout' else 24
            complete=len(q)==expected
            row=dict(cohort=cohort,variant=variant,scope=scope,rows=len(q),conditions=q.sample_id.nunique(),
                complete=complete,numeric=int(q.numeric.eq(True).sum()),valid=int(q.valid.eq(True).sum()))
            for endpoint in ('nH','mu','rho'):row[endpoint+'_MAE']=c.group_score(q.to_dict('records'),endpoint,paired=endpoint!='nH',group_equal=False) if complete else None
            summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'complete.json',dict(created_utc=c.e.now(),new_fit_records=len(records),new_optimizer_starts=sum(len(r['trials']) for r in records),
        new_external_physical_conditions=0,rows=len(frame),all_fits_qualified=all(r['selected_success'] and r['training_valid'] for r in records),
        no_blind_or_microscopic_identification_claim=True))


if __name__=='__main__':main()
