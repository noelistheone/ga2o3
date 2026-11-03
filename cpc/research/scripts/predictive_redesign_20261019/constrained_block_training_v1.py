"""Actual two-stage training with full derivatives and training-nH constraints.

Stage1: nH-only shared inventory. Stage2: fixed stage1 weights, train only
association/texture with genuine objective and constraint derivatives. This
does not claim to hold unseen nH errors fixed or prove charge-moment autonomy.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,time,os,itertools,numpy as np,pandas as pd
from scipy.optimize import least_squares,minimize
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase34_learning_v1 as overlay
import phase35_learning_v1 as spatial

OUT=c.BASE/'proposal_followup_20261021_v1/constrained_block_v1'
DRIFT=.03;LOSS_SLACK=.0001;FEAS_TOL=1e-7


def obj_and_constraints(method,prefix,beta,rows,baseline):
    w=np.r_[prefix,beta];q=spatial.forward(method,w,rows,penalty=True)
    wn=c.learning.weights(rows);wp=c.learning.weights(rows,paired_only=True);paired,rho=c.learning.observed_pair(rows)
    v=q['values'];J=q['jacobian'][:,:,6:];rn=v[:,0]-rows.target.to_numpy(float)
    rr=-v[paired,2]-rho[paired];obj=float(wp[paired]@(rr*rr)+.1*(wn@(q['prior_mu']**2)))
    grad=-2*(wp[paired]*rr)@J[paired,2]+.2*(wn*q['prior_mu'])@q['prior_mu_jacobian'][:,6:]
    drift=v[:,0]-baseline['values'][:,0]
    nl=float(wn@(rn*rn));limit=baseline['nH_loss']+LOSS_SLACK
    constraints=np.r_[limit-nl,DRIFT-drift,DRIFT+drift]
    CJ=np.r_[(-2*(wn*rn)@J[:,0])[None,:],-J[:,0],J[:,0]]
    return obj,np.asarray(grad),constraints,CJ,q


def primary_fit(held):
    train=tuple(u for u in c.units() if u!=held);rows=overlay.subset(train);cfg=dict(gamma=0.,alpha=.1)
    initial,warm=overlay.initializer(train,'OverlayShareProcess',cfg);lo,hi=overlay.bounds('OverlayShareProcess')
    cache={};calls=0;failures=[];start=time.perf_counter()
    def calc(w):
        nonlocal calls
        key=tuple(w)
        if key not in cache:
            calls+=1
            try:res=overlay.loss_arrays('OverlayShareProcess',w,rows,cfg)
            except (ValueError,RuntimeError,ArithmeticError) as exc:failures.append(repr(exc));raise
            cache.clear();cache[key]=res
        return cache[key]
    trials=[]
    def run(w,cap):
        try:
            r=least_squares(lambda x:calc(x)[0],w,jac=lambda x:calc(x)[1],bounds=(lo,hi),x_scale='jac',
                max_nfev=cap,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            return dict(parameters=r.x.tolist(),loss=float(r.fun@r.fun),success=bool(r.success),nfev=int(r.nfev),cap=cap,message=r.message)
        except (ValueError,RuntimeError,ArithmeticError) as exc:return dict(parameters=None,loss=None,success=False,cap=cap,error=repr(exc))
    trials.extend(run(w,80) for w in initial)
    if any(not q['success'] for q in trials):
        ff=[r for r in trials if r['loss'] is not None];more=[min(ff,key=lambda r:r['loss'])['parameters']] if ff else []
        trials.extend(run(w,160) for w in [*initial,*more])
    ff=[r for r in trials if r['loss'] is not None];best=min(ff,key=lambda r:r['loss']) if ff else None
    fit=dict(held=held,train_units=list(train),initializer=warm,config=cfg,trials=trials,failures=failures,
        parameters=best['parameters'] if best else None,selected_success=bool(best and best['success']),
        training_valid=False,seconds=time.perf_counter()-start,objective_calls=calls)
    if best:
        _,_,q=calc(np.array(best['parameters']));fit['training_valid']=bool(q['valid'].all())
    c.write(OUT/'primary_fits'/(c.digest(held)+'.json'),fit)
    print('Constrained primary',held,fit['selected_success'],fit['training_valid'],round(fit['seconds'],1),flush=True)
    return fit


def predict(method,w,held,variant):
    rows=c.cases();rows=rows[rows.related_cluster.eq(held)];mapper=spatial.model(method);out=[]
    for row in rows.itertuples():
        clean=row._replace(target=np.nan,mu_cm2Vs=np.nan,carrier_cm3=np.nan,observed_log_nH=np.nan)
        item=dict(case_id=row.case_id,sample_id=row.sample_id,family=row.family,related_cluster=row.related_cluster,
            variant=variant,role='query',budget=0,primary_common20=bool(row.primary_common20),
            scenario_index=row.scenario_index,mobility_eligible=bool(row.mobility_eligible),observed=float(row.target),
            observed_log_mu=float(np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None,
            observed_log_sigma=float(np.log10(c.learning.E_CHARGE)+row.target+np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None)
        try:
            p=mapper.point(clean,w,require_observation=False);v=np.r_[p.values[0],p.transport_values];d=p.diagnostics
            item.update(predicted=float(v[0]),predicted_log_mu=float(v[1]),predicted_log_sigma=float(v[2]),
                numeric=bool(np.isfinite(v).all()),valid=bool(d['observation_domain_valid']),
                texture_amplitude=d['actual_texture_amplitude'],triple_fraction=d['triple_complex_dopant_fraction'],
                conductivity_weighted_kell_bad=d['conductivity_weighted_kell_bad'],resistivity_weighted_kell_bad=d['resistivity_weighted_kell_bad'])
        except (ValueError,RuntimeError,ArithmeticError) as exc:item.update(numeric=False,valid=False,predicted=None,predicted_log_mu=None,predicted_log_sigma=None,error=repr(exc))
        out.append(item)
    return out


def secondary_fit(task):
    held,method=task;primary=c.document(OUT/'primary_fits'/(c.digest(held)+'.json'))
    key=c.digest(dict(held=held,method=method,protocol=c.e.sha(OUT/'protocol_before.json')));clock=time.perf_counter()
    if not primary['selected_success'] or not primary['training_valid']:
        result=dict(held=held,method=method,refused='unqualified primary fit');c.write(OUT/'secondary_fits'/(key+'.json'),result);return result
    prefix=np.array(primary['parameters']);rows=overlay.subset(primary['train_units']);b0=np.array([-12.,0.])
    q0=spatial.forward(method,np.r_[prefix,b0],rows,penalty=True);wn=c.learning.weights(rows)
    baseline=dict(values=q0['values'],nH_loss=float(wn@((q0['values'][:,0]-rows.target.to_numpy(float))**2)))
    cache={};calls=0;exceptions=[]
    def calc(beta):
        nonlocal calls
        kk=tuple(beta)
        if kk not in cache:
            calls+=1
            try:r=obj_and_constraints(method,prefix,beta,rows,baseline)
            except (ValueError,RuntimeError,ArithmeticError) as exc:exceptions.append(dict(beta=list(beta),error=repr(exc)));raise
            cache.clear();cache[kk]=r
        return cache[kk]
    value,grad,con,CJ,q=calc(b0)
    baseline_feasible=bool(q['valid'].all() and min(con)>=-FEAS_TOL)
    initial=[b0];target=np.array([4.,24.]);seed_probes=[]
    for t in [1.,.1,.01,.001]:
        beta=b0+t*(target-b0)
        try:rr=calc(beta);feasible=bool(rr[4]['valid'].all() and min(rr[2])>=-FEAS_TOL)
        except (ValueError,RuntimeError,ArithmeticError):feasible=False
        seed_probes.append(dict(fraction=t,feasible=feasible))
        if feasible:initial.append(beta);break
    if len(initial)==1:initial.append(b0.copy())
    trials=[]
    for beta in initial:
        try:
            opt=minimize(lambda x:calc(x)[0],beta,jac=lambda x:calc(x)[1],method='SLSQP',bounds=[(-12.,12.),(0.,30.)],
                constraints=[dict(type='ineq',fun=lambda x:calc(x)[2],jac=lambda x:calc(x)[3])],options=dict(maxiter=60,ftol=1e-9))
            rr=calc(opt.x);feasible=bool(rr[4]['valid'].all() and min(rr[2])>=-FEAS_TOL)
            trials.append(dict(beta=opt.x.tolist(),objective=float(rr[0]),success=bool(opt.success),feasible=feasible,
                iterations=int(opt.nit),message=opt.message,min_constraint=float(min(rr[2]))))
        except (ValueError,RuntimeError,ArithmeticError) as exc:trials.append(dict(beta=None,objective=None,success=False,feasible=False,error=repr(exc)))
    fallback=dict(beta=b0.tolist(),objective=float(value),success=baseline_feasible,feasible=baseline_feasible,
        fallback=True,min_constraint=float(min(con)),not_an_optimizer_convergence_claim=True)
    eligible=[r for r in trials if r['success'] and r['feasible']]+([fallback] if baseline_feasible else [])
    best=min(eligible,key=lambda r:r['objective']) if eligible else None
    result=dict(held=held,method=method,key=key,primary_path=str((OUT/'primary_fits'/(c.digest(held)+'.json')).relative_to(c.ROOT)),
        prefix_frozen=True,primary_parameters=prefix.tolist(),baseline_nH_loss=baseline['nH_loss'],baseline_rho_objective=float(value),
        seed_probes=seed_probes,trials=trials,baseline_candidate=fallback,selected=best,objective_calls=calls,exceptions=exceptions,
        seconds=time.perf_counter()-clock,protects_training_not_unseen_scores=True)
    if best:
        rr=calc(np.array(best['beta']));result.update(training_max_nH_drift=float(max(abs(rr[4]['values'][:,0]-baseline['values'][:,0]))),
            training_nH_loss_change=float(c.learning.weights(rows)@((rr[4]['values'][:,0]-rows.target.to_numpy(float))**2)-baseline['nH_loss']),
            training_full_constraints_valid=bool(rr[4]['valid'].all() and min(rr[2])>=-FEAS_TOL))
        predictions=predict(method,np.r_[prefix,best['beta']],held,method)
        if method=='ClusterCoupled':predictions.extend(predict(method,np.r_[prefix,b0],held,'FrozenPrimary'))
    else:predictions=[]
    c.write(OUT/'secondary_fits'/(key+'.json'),result);c.write(OUT/'predictions'/(key+'.json'),predictions)
    print('Constrained secondary',method,held,'fallback',None if best is None else best.get('fallback',False),
        'rhoobj',None if best is None else round(best['objective'],5),'sec',round(result['seconds'],1),flush=True)
    return result


def gates():
    target=OUT/'loss_gates_v1';assert not (target/'protocol_before.json').exists()
    c.write(target/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes([Path(__file__),Path(spatial.__file__)]),
        purpose='objective and actual inequality derivatives,not masked Jacobian'))
    held=c.units()[0];train=[u for u in c.units() if u!=held];rows=overlay.subset(train)
    f,_=c.fit(train,'ShareProcess205',dict(gamma=0.,alpha=.1));prefix=np.array(f['parameters']);beta=np.array([-4.,8.])
    q=spatial.forward('ClusterCoupled',np.r_[prefix,[-12.,0.]],rows,penalty=True);wn=c.learning.weights(rows)
    baseline=dict(values=q['values'],nH_loss=float(wn@((q['values'][:,0]-rows.target.to_numpy(float))**2)))
    results=[]
    for method in spatial.METHODS:
        a=obj_and_constraints(method,prefix,beta,rows,baseline)
        for j in range(2):
            h=2e-4;e=np.zeros(2);e[j]=h
            b=obj_and_constraints(method,prefix,beta+e,rows,baseline);d=obj_and_constraints(method,prefix,beta-e,rows,baseline)
            error=float(abs((b[0]-d[0])/(2*h)-a[1][j])/max(1.,abs(a[1][j])))
            ce=float(np.max(abs((b[2]-d[2])/(2*h)-a[3][:,j])/np.maximum(1.,abs(a[3][:,j]))))
            results.append(dict(method=method,coordinate=j,objective_FD_error=error,constraint_FD_error=ce,passed=max(error,ce)<3e-6))
    c.verify_hashes(c.document(target/'protocol_before.json'))
    c.write(target/'gates.json',dict(created_utc=c.e.now(),passed=all(r['passed'] for r in results),checks=len(results),records=results,new_fits=0))
    print('Constrained loss gates',all(r['passed'] for r in results),results,flush=True)


def main():
    if '--gates' in sys.argv:gates();return
    assert not (OUT/'protocol_before.json').exists()
    assert c.document(OUT/'loss_gates_v1/gates.json')['passed'];c.verify_hashes(c.document(OUT/'loss_gates_v1/protocol_before.json'))
    files=[Path(__file__),Path(spatial.__file__),Path(overlay.__file__),Path(c.__file__),
        *[Path(__file__).with_name(n) for n in ['association_spatial_model_v1.py','neutral_association_model_v1.py','neutral_association_inventory_v1.py','si_multidonor_overlay_v1.py','spatial_poisson_v2.py','spatial_local_charge_v2.py','spatial_parent_profile_v1.py','spatial_transport_v1.py']],
        *list((c.ROOT/'src/sandbox').glob('*.py'))]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),pid=os.getpid(),workers=4,
        primary_stage='sameoverlay sixparameter nHonly alpha.1;five relatedgroupout fits;two starts80/refine160',
        secondary_stage='frozen prefix;actual two-coordinate association/texture constrained SLSQP;3models x5folds;two starts60',
        constraints=dict(maximum_training_prediction_drift_dex=DRIFT,training_nH_squared_loss_slack=LOSS_SLACK,feasibility_tolerance=FEAS_TOL),
        loss='paired rho squared+0.1 totalmu-response penalty;no derivedthirdlabel',
        baseline='sameprefix beta[-12,0];tinyfiniteassociationexplicit,not exact zeroassociation theorem',
        fallback='feasible baseline retained;nonstopped/infeasible optimizerresults not selected',
        source_budget=0,primary_endpoint='common20nH;pairedmu/rho secondary',not_blind=True,
        no_masked_jacobian=True,no_claim_unseen_nH_preserved=True,
        code_hashes=c.file_hashes(files),input_hashes=c.file_hashes([c.OLD/'data/scenarios49.csv',OUT/'loss_gates_v1/gates.json'])))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(primary_fit,c.units()):pass
    tasks=list(itertools.product(c.units(),spatial.METHODS));finished=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(secondary_fit,t) for t in tasks]):
            finished.append(f.result());c.write(OUT/'status.json',dict(created_utc=c.e.now(),finished=len(finished),planned=15))
    predictions=[]
    for p in (OUT/'predictions').glob('*.json'):predictions.extend(c.document(p))
    frame=pd.DataFrame(predictions);frame.to_csv(OUT/'predictions.csv',index=False);summaries=[]
    for variant,scope in itertools.product(('FrozenPrimary',*spatial.METHODS),('all_query','common20')):
        q=frame[frame.variant.eq(variant)]
        if scope=='common20':q=q[q.primary_common20.eq(True)]
        expected=c.cases();expected=expected[expected.primary_common20.eq(True)] if scope=='common20' else expected
        complete=set(q.case_id)==set(expected.case_id) and len(q)==len(expected)
        item=dict(variant=variant,scope=scope,complete=complete,rows=len(q),numeric=int(q.numeric.eq(True).sum()),valid=int(q.valid.eq(True).sum()))
        for endpoint in ('nH','mu','rho'):item[endpoint+'_MAE']=c.group_score(q.to_dict('records'),endpoint,paired=endpoint!='nH',group_equal=False) if complete else None
        summaries.append(item)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False);c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'complete.json',dict(created_utc=c.e.now(),primary_fits=5,secondary_fit_records=len(finished),seconds=time.perf_counter()-start,
        prediction_rows=len(frame),fallback_selected=sum(bool(r.get('selected',{}).get('fallback',False)) for r in finished),
        actual_new_material_samples=0,stable_prediction_gain_established=False))


if __name__=='__main__':main()
