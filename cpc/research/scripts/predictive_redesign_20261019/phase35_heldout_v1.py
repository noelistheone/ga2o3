"""Fixed-hyperparameter, related-group held-out spatial/association comparison.

No query-selected regimes, no masked Jacobians, no claims of monotone benefit.
Existing phase35 pilot and all earlier code/results remain unchanged.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import os,sys,time,itertools,csv,numpy as np,pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase35_learning_v1 as spatial
import phase34_learning_v1 as overlay

OUT=c.BASE/'phase35_association_spatial/related_group_heldout_v1'
METHODS=('OverlayReference','OverlayShareProcess','ConstantAssociation',*spatial.METHODS)
CONFIGS=(dict(gamma=1.,alpha=.1),dict(gamma=1.,alpha=0.))
ADAPT=('none','offset_fixed','share_step_fixed')


def backend(method):return spatial if method in spatial.METHODS else overlay


def source_predictions(method,parameters,rows,budget,adaptation):
    """Only anchor targets are passed into adaptation; queries carry no labels."""
    mapper=backend(method).model(method);roles=c.roles()
    result=[];actual_calls=0;cache={}
    def point(row,z=0.):
        nonlocal actual_calls
        key=(c.learning.input_key(row,'process'),z)
        if key not in cache:
            p=mapper.point(row,parameters,source_kind='share' if z else 'none',source_shift=z,require_observation=False)
            v=np.r_[p.values[0],p.transport_values]
            cache[key]=(v,float(p.source_jacobian[0]),bool(p.diagnostics['observation_domain_valid']),p.diagnostics)
            actual_calls+=1
        return cache[key]
    for family,source in rows.groupby('family',sort=True):
        episode=roles[roles.family.eq(family)&roles.budget.eq(budget)]
        rolemap=dict(zip(episode.sample_id,episode.role))
        anchor_names=set(episode.loc[episode.role.eq('anchor'),'sample_id'])
        assert len(anchor_names)==budget
        anchors=source[source.sample_id.isin(anchor_names)]
        residuals=[];gradients=[];delta=0.;z=0.;failed=[]
        for sample,aa in anchors.groupby('sample_id',sort=True):
            vals=[];jj=[]
            for row in aa.itertuples():
                try:
                    v,j,valid,d=point(row)
                    if not valid:failed.append('anchor outside declared observation domain')
                    vals.append(v[0]);jj.append(j)
                except (ValueError,RuntimeError,ArithmeticError) as exc:failed.append(repr(exc))
            if len(vals)!=len(aa):continue
            assert aa.target.nunique()==1
            residuals.append(float(aa.target.iloc[0])-float(np.mean(vals)))
            gradients.append(float(np.mean(jj)))
        if budget and adaptation!='none' and len(residuals)==budget:
            if adaptation=='offset_fixed':delta=float(np.sum(residuals)/(budget+1.))
            else:
                # One declared regularized IFT step, not a converged nonlinear MAP.
                # Averaging scenarios within each physical anchor avoids pseudo-replication.
                g=np.asarray(gradients)/.3;r=np.asarray(residuals)/.3
                z=float(np.clip(g@r/(g@g+1.),-1.,1.))
        ok_source=adaptation=='none' or (not failed and len(residuals)==budget)
        # Labels are not consulted during the actual query forward call.
        for row in source.itertuples():
            item=dict(case_id=row.case_id,sample_id=row.sample_id,family=family,related_cluster=row.related_cluster,
                scenario_index=row.scenario_index,scenario_weight=row.scenario_weight,role=rolemap[row.sample_id],
                primary_common20=bool(row.primary_common20),mobility_eligible=bool(row.mobility_eligible),
                observed=float(row.target),observed_log_mu=float(np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None,
                incorporated_cm3=float(row.incorporated_cm3),budget=budget,adaptation=adaptation,
                delta_dex=delta,source_coordinate=z,source_anchor_ids=sorted(anchor_names),
                source_valid=ok_source,source_failures=failed,regime=('low' if row.incorporated_cm3<1e18 else 'middle' if row.incorporated_cm3<1e19 else 'high'),
                observed_log_sigma=float(np.log10(c.learning.E_CHARGE)+row.target+np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None)
            try:
                # Remove target labels from the object passed to the physical predictor.
                clean=row._replace(target=np.nan,mu_cm2Vs=np.nan,carrier_cm3=np.nan,observed_log_nH=np.nan)
                v,j,valid,dg=point(clean,z)
                v=v.copy();v[[0,2]]+=delta
                item.update(predicted=float(v[0]),predicted_log_mu=float(v[1]),predicted_log_sigma=float(v[2]),
                    numeric=bool(np.isfinite(v).all()),valid=bool(valid and ok_source),source_derivative=j,
                    same_physical_state=bool(delta==0.),empirical_offset_not_recovered_physical_state=bool(delta!=0.))
                for key in ['actual_texture_amplitude','triple_complex_dopant_fraction','conductivity_weighted_kell_bad',
                    'resistivity_weighted_kell_bad','maximum_moment1_ell_grad_logn']:
                    item[key]=dg.get(key)
                item['potential_span_eV']=float(np.ptp(dg['local_fermi_eV'])) if 'local_fermi_eV' in dg else None
            except (ValueError,RuntimeError,ArithmeticError) as exc:
                item.update(predicted=None,predicted_log_mu=None,predicted_log_sigma=None,numeric=False,valid=False,error=repr(exc))
            result.append(item)
    return result,actual_calls


def fit_task(task):
    held,method,config=task;train=tuple(u for u in c.units() if u!=held)
    key=c.digest(dict(held=held,method=method,config=config,protocol=c.e.sha(OUT/'protocol_before.json')))
    fitpath=OUT/'fits'/(key+'.json');predpath=OUT/'predictions'/(key+'.json')
    if fitpath.exists():raise RuntimeError('Do not duplicate an existing fit')
    b=backend(method);rows=b.subset(train);clock=time.perf_counter();fails=[];cache={};calls=0
    lower,upper=spatial.bounds() if b is spatial else overlay.bounds(method)
    initial,warm=b.initializer(train,method,config)
    def calc(w):
        nonlocal calls
        kk=tuple(w)
        if kk not in cache:
            calls+=1
            try:q=b.loss_arrays(method,w,rows,config)
            except (ValueError,RuntimeError,ArithmeticError) as exc:
                fails.append(dict(parameters=np.asarray(w).tolist(),error=repr(exc)));raise
            cache.clear();cache[kk]=q
        return cache[kk]
    def optimize(start,cap,stage):
        try:
            r=least_squares(lambda w:calc(w)[0],start,jac=lambda w:calc(w)[1],bounds=(lower,upper),
                x_scale='jac',max_nfev=cap,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            return dict(parameters=r.x.tolist(),loss=float(r.fun@r.fun),success=bool(r.success),
                nfev=int(r.nfev),optimality=float(r.optimality),active_bounds=r.active_mask.tolist(),message=r.message,stage=stage,cap=cap)
        except (ValueError,RuntimeError,ArithmeticError) as exc:
            return dict(parameters=None,loss=None,success=False,error=repr(exc),stage=stage,cap=cap)
    trials=[optimize(w,80,'initial') for w in initial]
    if any(not q['success'] for q in trials):
        ff=[q for q in trials if q['loss'] is not None];best=[min(ff,key=lambda q:q['loss'])['parameters']] if ff else []
        trials.extend(optimize(w,160,'uniform_refinement') for w in [*initial,*best])
    ff=[q for q in trials if q['loss'] is not None];selected=min(ff,key=lambda q:q['loss']) if ff else None
    record=dict(created_utc=c.e.now(),key=key,held_group=held,train_units=list(train),method=method,config=config,
        initializer_reference=warm,trials=trials,state_failures=fails,parameters=selected['parameters'] if selected else None,
        selected_success=bool(selected and selected['success']),training_valid=False,objective_forward_calls=calls)
    if selected:
        res,J,q=calc(np.asarray(selected['parameters']))
        singular=np.linalg.svd(q['jacobian'][:,0],compute_uv=False)
        record.update(training_valid=bool(q['valid'].all()),selected_loss=float(res@res),
            training_nH_jacobian_singular_values=singular.tolist(),training_nH_jacobian_rank=int(np.linalg.matrix_rank(q['jacobian'][:,0])),
            training_case_ids=rows.case_id.tolist())
    record['fit_seconds']=time.perf_counter()-clock
    c.write(fitpath,record)
    heldrows=c.cases()[c.cases().related_cluster.eq(held)].copy();predictions=[];ncalls=0
    if record['selected_success'] and record['training_valid']:
        for budget in (0,1,2):
            for adaptation in ADAPT:
                pp,nn=source_predictions(method,np.asarray(record['parameters']),heldrows,budget,adaptation)
                predictions.extend(pp);ncalls+=nn
    else:
        # No favorable subset metric is computed for an unqualified fit.
        record['prediction_refused']='unqualified training fit'
    for r in predictions:r.update(method=method,alpha=config['alpha'],gamma=config['gamma'],fit_key=key,outer_held=held)
    c.write(predpath,dict(created_utc=c.e.now(),fit_key=key,rows=predictions,actual_forward_calls=ncalls,
        refusal=record.get('prediction_refused'),held_condition_ids=sorted(heldrows.sample_id.unique())))
    end=dict(key=key,method=method,held_group=held,config=config,selected_success=record['selected_success'],
        training_valid=record['training_valid'],fit_seconds=record['fit_seconds'],seconds=time.perf_counter()-clock,
        prediction_rows=len(predictions),physical_calls=ncalls)
    c.write(OUT/'finished_tasks'/(key+'.json'),end)
    print('Held-out phase35',method,config['alpha'],held,'stopped',end['selected_success'],
        'valid',end['training_valid'],'sec',round(end['seconds'],1),flush=True)
    return end


def summarize():
    rows=[]
    for p in (OUT/'predictions').glob('*.json'):rows.extend(c.document(p)['rows'])
    if not rows:raise RuntimeError('No predictions')
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'predictions.csv',index=False)
    summary=[]
    for method,config,adapt,budget,scope in itertools.product(METHODS,CONFIGS,ADAPT,(0,1,2),('all_query','common20')):
        q=frame[frame.method.eq(method)&frame.alpha.eq(config['alpha'])&frame.adaptation.eq(adapt)&frame.budget.eq(budget)&frame.role.eq('query')]
        if scope=='common20':q=q[q.primary_common20.eq(True)]
        expected=c.cases();roles=c.roles();roleids=set(roles.loc[roles.budget.eq(budget)&roles.role.eq('query'),'sample_id'])
        expected=expected[expected.sample_id.isin(roleids)]
        if scope=='common20':expected=expected[expected.primary_common20.eq(True)]
        complete=len(q)==len(expected) and set(q.case_id)==set(expected.case_id)
        item=dict(method=method,alpha=config['alpha'],adaptation=adapt,budget=budget,scope=scope,
            expected_rows=len(expected),returned_rows=len(q),complete_coverage=complete,conditions=q.sample_id.nunique(),
            numeric=int(q.numeric.eq(True).sum()),valid=int(q.valid.eq(True).sum()))
        for endpoint,paired in [('nH',False),('mu',True),('rho',True)]:
            value=c.group_score(q.to_dict('records'),endpoint,paired=paired,group_equal=False) if complete else None
            item[endpoint+'_MAE']=value
        summary.append(item)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    records=[c.document(p) for p in (OUT/'finished_tasks').glob('*.json')]
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'complete.json',dict(created_utc=c.e.now(),tasks=len(records),planned_tasks=60,
        training_fits=len(list((OUT/'fits').glob('*.json'))),actual_optimizer_starts=sum(len(c.document(p)['trials']) for p in (OUT/'fits').glob('*.json')),
        selected_nonstopped=sum(not r['selected_success'] for r in records),invalid_training_fits=sum(not r['training_valid'] for r in records),
        prediction_rows=len(frame),physical_prediction_calls=sum(r['physical_calls'] for r in records),
        no_hyperparameter_selection_by_outer_scores=True,retrospective=True,new_material_samples=0,
        stable_prediction_gain_established=False))


def main():
    if '--summarize' in sys.argv:summarize();return
    if (OUT/'protocol_before.json').exists():raise RuntimeError('Existing declared run: inspect it, do not duplicate')
    assert c.document(spatial.OUT/'all_training_pilot_v1/complete.json')['records']
    tasks=[(held,m,config) for held in c.units() for m in METHODS for config in CONFIGS]
    assert len(tasks)==60
    code=[Path(__file__),Path(spatial.__file__),Path(overlay.__file__),Path(c.__file__),
        *[Path(__file__).with_name(n) for n in ['association_spatial_model_v1.py','neutral_association_model_v1.py','neutral_association_inventory_v1.py',
            'si_multidonor_overlay_v1.py','spatial_local_charge_v2.py','spatial_poisson_v2.py','spatial_parent_profile_v1.py','spatial_transport_v1.py']],
        *list((c.ROOT/'src/sandbox').glob('*.py'))]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),pid=os.getpid(),workers=4,tasks=tasks,
        primary='ClusterCoupled alpha.1;zero-anchor common20 nH,paired mu/rho secondary',
        hyperparameters='gamma1;primaryalpha.1,alpha0 prespecified sensitivity;neither selected by outer scores',
        same_overlay_controls=['OverlayReference','OverlayShareProcess','ConstantAssociation'],
        source_adaptations='none primary;empirical offset mean residual/(k+1);one regularized IFT share-coordinate step',
        source_working_scale_dex=.3,source_step_bound=[-1.,1.],source_metric_not_measured_noise=True,
        scenario_information='one physical anchor target,average scenario prediction/Jacobian before adaptation',
        source_selection='fixed rules,not a nested optimized new-source gate',
        optimizer='two starts80;anynonstop => bothstarts+bestfinite160;bestfinitepreserved;nonstoppednotserved',
        regimes='doseonly exploratory low<1e18,middle<1e19,high>=1e19;not selected by observedmu or used to choosemodels',
        no_detached_false_least_squares_jacobian=True,not_blind=True,no_new_material_or_DFT_MD=True,
        code_hashes=c.file_hashes(code),input_hashes=c.file_hashes([c.OLD/'data/scenarios49.csv',c.OLD/'data/episodes35.csv',
            spatial.OUT/'loss_gates_v1/gates.json',spatial.OUT/'all_training_pilot_v1/complete.json'])))
    start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fit_task,t) for t in tasks]
        for f in as_completed(futures):
            r=f.result();records.append(r)
            c.write(OUT/'status.json',dict(created_utc=c.e.now(),finished=len(records),planned=len(tasks),records=records,seconds=time.perf_counter()-start))
    summarize()


if __name__=='__main__':main()
