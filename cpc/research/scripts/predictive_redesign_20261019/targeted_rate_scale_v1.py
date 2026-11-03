"""Positive common-rate scaling with properly isolated upstream inner fits."""
from pathlib import Path
import sys,itertools,time
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase35_learning_v1 as spatial
import targeted_closure_model_v1 as m
import targeted_closure_training_v1 as training
from sandbox.coherent_transport import fermi_transport

OUT=m.OUT/'rate_scale_v1'
PREVIOUS=c.BASE/'proposal_followup_20261021_v1'
ALPHAS=(.001,.01,.1,1.,10.,100.,1e6)
VARIANTS=('FrozenPrimary','RateConstantMean','RateConstantMedian','RateFeatureRidge')
CACHE={};PRIMARY={};FITS={};CV={};MOMENTS=[];CALLS=0;SOLVES=0
SAVED=c.read(PREVIOUS/'constrained_block_v1/predictions.csv')
SAVED=SAVED[SAVED.variant.eq('FrozenPrimary')].set_index('case_id')


def primary(train):
    train=tuple(sorted(train))
    if train not in PRIMARY:
        if len(train)==3:path=training.fit_path('LinearShared',train)
        elif len(train)==4:
            held=next(u for u in c.units() if u not in train)
            path=PREVIOUS/'constrained_block_v1/primary_fits'/(c.digest(held)+'.json')
        elif len(train)==5:
            paths=[p for p in (PREVIOUS/'overlay_nH_controls_v1/fits').glob('*.json')
                if c.document(p)['held'] is None and c.document(p)['method']=='OverlayShareProcess']
            assert len(paths)==1;path=paths[0]
        else:raise ValueError('Only declared upstream contexts supported')
        fit=c.document(path);assert fit['training_valid'] and fit['selected_success']
        assert set(fit['train_units'])==set(train)
        PRIMARY[train]=(fit,path)
    return PRIMARY[train]


def base(train,rows):
    global CALLS
    train=tuple(sorted(train));fit,path=primary(train);result=[];valid=[]
    for row in rows.itertuples():
        key=(train,str(row.case_id))
        if key not in CACHE:
            p=spatial.model('ClusterCoupled').point(m.clean(row),np.r_[fit['parameters'],-12.,0.],require_observation=False)
            v=m.point_values(p);CALLS+=1;replay_difference=None
            # Copy the exact prior output only after checking a physical replay.
            if len(train)==4 and row.related_cluster not in train and row.case_id in SAVED.index:
                s=SAVED.loc[row.case_id];old=np.array([s.predicted,s.predicted_log_mu,s.predicted_log_sigma])
                replay_difference=float(max(abs(v-old)));assert replay_difference<5e-10;v=old
            CACHE[key]=dict(values=v,valid=bool(p.diagnostics['observation_domain_valid']),
                primary_path=str(path.relative_to(c.ROOT)),case_id=row.case_id,train_units=list(train),
                replay_difference=replay_difference,coherent=p.diagnostics['coherent_transport'])
            # One real state per upstream context; test the actual Coulomb+constant channels.
            if not any(q['train_units']==list(train) for q in MOMENTS):
                state=p.diagnostics['coherent_transport'];profile=spatial.model('ClusterCoupled').profile
                for d in (-2.,0.,2.):
                    kw=dict(Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,mass_ratio=profile.electron_mass_ratio)
                    a=fermi_transport(np.log10(state['free_n_cm3']),np.log10(state['Nq2_cm3']),300.,extra_mobility_cm2Vs=205.,**kw)
                    b=fermi_transport(np.log10(state['free_n_cm3']),np.log10(state['Nq2_cm3'])+d,300.,extra_mobility_cm2Vs=205./10**d,**kw)
                    error=float(max(abs(b.values-m.scale_observables(a.values,d))))
                    MOMENTS.append(dict(train_units=list(train),case_id=row.case_id,log10_rate_scale=d,error_dex=error,passed=error<1e-10,
                        interpretation='rate coefficient scaling only;not a modified atomic q2 inventory'))
        q=CACHE[key];result.append(q['values']);valid.append(q['valid'])
    return np.asarray(result),np.asarray(valid,bool)


def rate_fit(train,kind,alpha=None):
    global SOLVES
    train=tuple(sorted(train));key=(train,kind,alpha)
    if key in FITS:return FITS[key]
    rows=c.cases();rows=rows[rows.related_cluster.isin(train)];rows=rows[rows.mobility_eligible.eq(True)].copy()
    assert len(rows)>0
    v,valid=base(train,rows);assert valid.all()
    weight=c.learning.weights(rows,paired_only=True);X=m.rate_features(rows)
    response=v[:,1]-np.log10(rows.mu_cm2Vs.to_numpy(float));coefs=np.zeros(4)
    if kind=='RateConstantMean':coefs[0]=float(weight@response)
    elif kind=='RateConstantMedian':
        order=np.argsort(response,kind='stable');coefs[0]=float(response[order[np.searchsorted(np.cumsum(weight[order]),.5)]])
    elif kind=='RateFeatureRidge':
        assert alpha in ALPHAS
        meanx=weight@X[:,1:];meanr=float(weight@response);xc=X[:,1:]-meanx;rr=response-meanr
        coefs[1:]=np.linalg.solve(xc.T@(weight[:,None]*xc)+alpha*np.eye(3),xc.T@(weight*rr))
        coefs[0]=meanr-meanx@coefs[1:];SOLVES+=1
    else:raise ValueError('Explicit rate model required')
    j=dict(train_units=list(train),variant=kind,alpha=alpha,coefficients=coefs.tolist(),
        training_case_ids=rows.case_id.tolist(),training_physical_pairs=int(rows.sample_id.nunique()),
        label='one paired logmu response;not rho or a third conductivity label',
        unpenalized_intercept=True,rate_scale_parameterization='a=10^(X beta),equivalent to exp(ln10 X beta)',
        primary_path=str(primary(train)[1].relative_to(c.ROOT)))
    FITS[key]=j
    return j


def prediction_rows(train,rows,kind,alpha=None,cohort='old_cohort_heldout'):
    v,ok=base(train,rows);fit=rate_fit(train,kind,alpha) if kind!='FrozenPrimary' else None
    d=np.zeros(len(rows)) if fit is None else m.rate_features(rows)@np.asarray(fit['coefficients'])
    new=m.scale_observables(v,d);assert np.array_equal(new[:,0],v[:,0])
    out=[]
    for i,row in enumerate(rows.itertuples()):
        out.append(dict(case_id=row.case_id,sample_id=row.sample_id,family=row.family,related_cluster=row.related_cluster,
            cohort=cohort,variant=kind,budget=0,role='query',primary_common20=bool(getattr(row,'primary_common20',False)),
            mobility_eligible=bool(row.mobility_eligible),scenario_index=float(row.scenario_index),observed=float(row.target),
            observed_log_mu=float(np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None,
            observed_log_sigma=float(np.log10(c.learning.E_CHARGE)+row.target+np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None,
            predicted=float(new[i,0]),predicted_log_mu=float(new[i,1]),predicted_log_sigma=float(new[i,2]),
            base_log_nH=float(v[i,0]),base_log_mu=float(v[i,1]),base_log_sigma=float(v[i,2]),log10_rate_scale=float(d[i]),
            positive_rate_scale=float(10**d[i]),numeric=bool(np.isfinite(new[i]).all() and np.isfinite(10**d[i])),valid=bool(ok[i]),
            selected_alpha=alpha,primary_train_units='|'.join(sorted(train)),
            same_frozen_electronic_state=True,phenomenological_common_scale_not_microscopic_species=True))
    return out


def choose(train):
    train=tuple(sorted(train))
    if train in CV:return CV[train]
    scores=[]
    for alpha in ALPHAS:
        bygroup=[]
        for held in train:
            rows=c.cases();rows=rows[rows.related_cluster.eq(held)]
            if not rows.mobility_eligible.eq(True).any():continue
            inner=tuple(u for u in train if u!=held)
            assert len(inner) in (3,4)
            rr=prediction_rows(inner,rows,'RateFeatureRidge',alpha)
            value=c.group_score(rr,'mu',paired=True,group_equal=False)
            bygroup.append(dict(held=held,score=value,train_units=list(inner)))
        assert bygroup and all(s['score'] is not None for s in bygroup)
        scores.append(dict(alpha=alpha,score=float(np.mean([s['score'] for s in bygroup])),folds=bygroup))
    best=min(scores,key=lambda j:(round(j['score'],10),-j['alpha']))
    result=dict(train_units=list(train),selected_alpha=best['alpha'],candidate_scores=scores,
        no_outer_or_external_labels_in_selection=True,upstream_inventories_fit_without_inner_held_source=True)
    CV[train]=result;return result


def main():
    assert not (OUT/'protocol_before.json').exists()
    assert (training.OUT/'complete.json').exists();c.verify_hashes(c.document(training.OUT/'protocol_before.json'))
    inputs=[training.OUT/'complete.json',PREVIOUS/'constrained_block_v1/predictions.csv',
        c.BASE/'external_rate_cohort_v2/scenarios24.csv',c.OLD/'data/scenarios49.csv']
    for p in (training.OUT/'fits').glob('*.json'):inputs.append(p)
    inputs.extend((PREVIOUS/'constrained_block_v1/primary_fits').glob('*.json'))
    inputs.extend((PREVIOUS/'overlay_nH_controls_v1/fits').glob('*.json'))
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),variants=VARIANTS,alpha_grid=ALPHAS,
        primary_endpoint='paired mobility;carrier must be bitwise unchanged relative to saved baseline;derived sigma secondary',
        feature_scales='fixed intercept,(log10C-18)/2,(growthC-700)/300,Sn indicator;no test normalization',
        fitting='weighted mean/median constants;Ridge with unpenalized intercept;inner related-group MAE selection,ties largeralpha',
        all_upstream_inner_exclusions_respected=True,outer_primary_fits_reused_unchanged=True,
        physical_map='all energy-dependent and constant rates multiplied by10^d;moment ratio cancels exactly',
        no_claim_scaling_restores_Born_or_RTA_validity=True,retrospective=True,
        code_hashes=c.file_hashes([Path(__file__),Path(m.__file__),Path(training.__file__),Path(spatial.__file__),
            *list((c.ROOT/'src/sandbox').glob('*.py'))]),input_hashes=c.file_hashes(inputs)))
    clock=time.perf_counter();pred=[]
    for held in c.units():
        train=tuple(u for u in c.units() if u!=held);selection=choose(train)
        rows=c.cases();rows=rows[rows.related_cluster.eq(held)]
        for kind in VARIANTS:pred.extend(prediction_rows(train,rows,kind,selection['selected_alpha'] if kind=='RateFeatureRidge' else None))
        print('Rate scale outer',held,'alpha',selection['selected_alpha'],'elapsed',round(time.perf_counter()-clock,1),flush=True)
    train=tuple(c.units());selection=choose(train);external=c.read(c.BASE/'external_rate_cohort_v2/scenarios24.csv')
    for kind in VARIANTS:pred.extend(prediction_rows(train,external,kind,selection['selected_alpha'] if kind=='RateFeatureRidge' else None,'external3_conditional'))
    frame=pd.DataFrame(pred);frame.to_csv(OUT/'predictions.csv',index=False);summary=[]
    for (cohort,variant),p in frame.groupby(['cohort','variant']):
        for scope in (('all_query','common20') if cohort=='old_cohort_heldout' else ('query',)):
            q=p[p.primary_common20.eq(True)] if scope=='common20' else p
            expected=20 if scope=='common20' else 49 if cohort=='old_cohort_heldout' else 24
            complete=len(q)==expected
            j=dict(cohort=cohort,variant=variant,scope=scope,rows=len(q),conditions=q.sample_id.nunique(),
                paired_conditions=q[q.mobility_eligible.eq(True)].sample_id.nunique(),complete=complete,
                numeric=int(q.numeric.eq(True).sum()),valid=int(q.valid.eq(True).sum()))
            for endpoint in ('nH','mu','rho'):j[endpoint+'_MAE']=c.group_score(q.to_dict('records'),endpoint,paired=endpoint!='nH',group_equal=False) if complete else None
            summary.append(j)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    c.write(OUT/'scale_fits.json',list(FITS.values()));c.write(OUT/'selections.json',list(CV.values()))
    c.write(OUT/'real_state_moment_checks.json',dict(passed=all(r['passed'] for r in MOMENTS),checks=len(MOMENTS),records=MOMENTS))
    cache=[]
    for (train,case),j in CACHE.items():cache.append(dict(train_units='|'.join(train),case_id=case,values=j['values'].tolist(),
        valid=j['valid'],primary_path=j['primary_path'],prior_output_replay_difference=j['replay_difference']))
    c.write(OUT/'base_prediction_cache.json',cache)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    complete=dict(created_utc=c.e.now(),prediction_rows=len(frame),actual_primary_point_calls=CALLS,actual_ridge_linear_solves=SOLVES,
        scale_fit_records=len(FITS),new_upstream_inventory_fits_in_this_script=0,seconds=time.perf_counter()-clock,
        maximum_nH_change_dex=float(max(abs(frame.predicted-frame.base_log_nH))),
        exact_nH_preserved=bool(np.array_equal(frame.predicted.to_numpy(),frame.base_log_nH.to_numpy())),
        physical_moment_checks_pass=all(r['passed'] for r in MOMENTS))
    c.write(OUT/'complete.json',complete);print(complete,flush=True)


if __name__=='__main__':main()
