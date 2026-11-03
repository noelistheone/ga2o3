"""V2 nuisance-profiled gate with the original normalized shift-penalty grid.

This is a versioned, retrospective carrier-task experiment, not a guarantee
of monotonic improvement or a microscopic interpretation of output offsets.
"""
from pathlib import Path
import sys,itertools,time,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from proposal_theory_checks_v1 import projected_design

OUT=c.BASE/'proposal_followup_20261021_v1/profiled_adaptation_v2'
METHOD='JointProcess';SCALES=np.array([.2,.5,1.,1.,1.,1.,.5]);SIGMA=.3
SHIFT_LAMBDAS=(None,10.,1.,.1,.01,0.)
TAUS=(None,10.,1.,.1)
VARIANTS=('FrozenShared','NestedShift','NestedProfiledGate','NestedPhysicsOrDose','NestedUnitDose')
CACHE={};UPDATES=[];CALLS=0


def physical(source,theta,need_jac=False):
    global CALLS
    values=[];jac=[];valid=[]
    for row in source.itertuples():
        key=(c.learning.input_key(row,'process'),tuple(theta))
        if key not in CACHE:
            clean=row._replace(target=np.nan,mu_cm2Vs=np.nan,carrier_cm3=np.nan,observed_log_nH=np.nan)
            try:
                p=c.learning.model(METHOD).point(clean,theta,require_observation=False);CALLS+=1
                CACHE[key]=(float(p.values[0]),p.jacobian[0].copy(),bool(p.diagnostics['observation_domain_valid']))
            except (ValueError,RuntimeError,ArithmeticError) as exc:
                CALLS+=1;CACHE[key]=(np.nan,np.full(len(theta),np.nan),False)
        y,J,ok=CACHE[key];values.append(y);jac.append(J);valid.append(ok)
    return np.array(values),np.array(jac),np.array(valid,bool)


def simulate(source,fit,budget,shift_lambda,tau,base='shared'):
    """No query values enter any update. Every physical anchor counts once."""
    roles=c.roles();theta=np.array(fit['parameters'])
    if base=='dose':
        y0=np.log10(source.incorporated_cm3.to_numpy(float));J=np.zeros((len(source),len(theta)));valid=np.isfinite(y0)&(y0>=13.)
    else:y0,J,valid=physical(source,theta)
    new=y0.copy()
    records=[];vmax=0.;rank=0;step=np.zeros_like(theta);delta=0.
    assert source.family.nunique()==1
    family=str(source.family.iloc[0]);ep=roles[roles.family.eq(family)&roles.budget.eq(budget)]
    anchor_ids=sorted(ep.loc[ep.role.eq('anchor'),'sample_id']);assert len(anchor_ids)==budget
    rolemap=dict(zip(ep.sample_id,ep.role));am=source.sample_id.isin(anchor_ids).to_numpy(bool)
    # Explicitly erase every non-anchor target before the update.
    labels=source.target.to_numpy(float).copy();labels[~am]=np.nan
    means=[];jmeans=[];targets=[];indices=[]
    for name in anchor_ids:
        ix=np.flatnonzero(source.sample_id.eq(name).to_numpy());indices.append(ix)
        assert np.unique(labels[ix]).size==1
        means.append(float(np.mean(y0[ix])));jmeans.append(np.mean(J[ix],axis=0));targets.append(float(labels[ix[0]]))
    if base=='dose':new=np.log10(source.incorporated_cm3.to_numpy(float))
    elif budget>=2 and tau is not None and valid[am].all():
        Janchor=np.asarray(jmeans);residual=np.array(targets)-np.array(means)
        K,F,P=projected_design(Janchor,np.repeat(SIGMA**-2,budget),SCALES)
        eig,V=np.linalg.eigh(F);vmax=max(float(eig[-1]),0.);rank=int(np.linalg.matrix_rank(K,tol=1e-10))
        if vmax>0:
            v=V[:,-1];gate=vmax/(vmax+tau)
            score=K.T@(P@(residual/SIGMA))
            z=float(np.clip(gate*(v@score)/(vmax+1.),-1.,1.))
            step=SCALES*v*z
            lo,hi=c.learning.bounds(METHOD);theta=np.clip(theta+step,lo,hi)
            step=theta-np.asarray(fit['parameters'])
            new,_,valid=physical(source,theta)
        records.append(dict(anchors=budget,effective_rank=rank,max_scaled_information=vmax,
            info_type='working design,not calibrated Fisher',tau=tau,parameter_step=step.tolist(),
            coordinate_projection_to_bounds=True))
    if budget:
        er=[targets[i]-float(np.mean(new[ix])) for i,ix in enumerate(indices)]
        if np.isfinite(er).all() and shift_lambda is not None:delta=float(np.mean(er)/(1.+shift_lambda))
        new=new+delta
    info=dict(family=family,budget=budget,shift_lambda=shift_lambda,tau=tau,base=base,anchor_ids=anchor_ids,
        anchors_count=budget,rank_after_free_offset=rank,max_scaled_information=vmax,
        updated_parameters=theta.tolist(),delta_dex=delta,updates=records,
        conditional_one_step_not_nonlinear_MAP=True,same_physical_state=bool(base=='shared' and delta==0.))
    UPDATES.append(info)
    for i,row in enumerate(source.itertuples()):
        records.append(dict(case_id=row.case_id,sample_id=row.sample_id,family=row.family,related_cluster=row.related_cluster,
            primary_common20=bool(row.primary_common20),mobility_eligible=False,role=rolemap[row.sample_id],
            budget=budget,predicted=float(new[i]) if np.isfinite(new[i]) else None,numeric=bool(np.isfinite(new[i])),
            valid=bool(valid[i] and np.isfinite(new[i])),observed=float(row.target),
            scenario_index=row.scenario_index,scenario_weight=row.scenario_weight,
            delta_dex=delta,information_rank=rank,information_max=vmax,internal_step_norm=float(np.linalg.norm(step/SCALES)),
            output_calibration_is_empirical=bool(delta!=0. or base=='dose')))
    # The first diagnostic record is not a prediction row.
    return [r for r in records if 'case_id' in r]


def policy_tie(p):
    k,t,b=p
    return (b!='shared',t is not None,-(t or 0.),k is not None,-(k or 0.))


def score_candidate(choice,budget,policy):
    k,t,base=policy;perheld=[]
    for item in choice['selected']['fits']:
        fit=c.document(c.ROOT/item['path']);assert fit['training_valid'] and fit['selected_success']
        assert item['held'] not in fit['train_units']
        table=c.cases();held=table[table.related_cluster.eq(item['held'])];rows=[]
        for _,source in held.groupby('family',sort=True):rows.extend(simulate(source,fit,budget,k,t,base))
        q=[r for r in rows if r['role']=='query']
        val=c.group_score(q,'nH',group_equal=False)
        perheld.append(np.inf if val is None else val)
    return float(np.mean(perheld))


def main():
    assert not (OUT/'protocol_before.json').exists()
    choices=[]
    for p in (c.OLD/'prediction/shared_choices').glob('*.json'):
        j=c.document(p)
        if (j['method']==METHOD and j['selection']=='nHonly'
                and j.get('outer_held') in c.units() and len(j['train_units'])==4):choices.append((p,j))
    assert len(choices)==5
    choices.sort(key=lambda q:q[1]['outer_held'])
    inputs=[p for p,j in choices]
    for p,j in choices:
        inputs.extend(c.ROOT/f['path'] for f in j['selected']['fits'])
        f,path=c.fit(j['train_units'],METHOD,j['selected']['config']);inputs.append(path)
    code=[Path(__file__),Path(c.__file__),Path(c.data.__file__),Path(c.learning.__file__),Path(c.design.__file__),Path(__file__).with_name('proposal_theory_checks_v1.py'),
        *list((c.ROOT/'src/sandbox').glob('*.py'))]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),primary='NestedProfiledGate',variants=VARIANTS,
        shared_model='frozen originalJointProcess nHonly;old innerzero selected config reused exactly for everyanchorbudget',
        source_selection='training innerheldrelatedgroup query nH macroMAE;original fixed sharedconfig;no outerqueries',
        revision='v2 matches original normalized-anchor-loss shift regularization grid;v1 preserved;retrospective correction,not independentconfirmation',
        shift_penalty_convention='delta=mean_anchor_residual/(1+shift_lambda);normalized anchor metric,not sum metric',
        shift_lambda_grid=SHIFT_LAMBDAS,tau_grid=TAUS,parameter_prior_scales=SCALES.tolist(),working_sigma_dex=SIGMA,
        update='free-offset nuisanceprojection;leading prior-scaled information direction;one regularizedstep;boxprojection;exactphysicalrecompute;profileempiricalshift',
        single_anchor_has_no_internal_shape_update=True,one_observation_per_physical_anchor=True,
        scenarios='fixed average predictions/Jacobians within each physicalanchor;not independent likelihood observations',
        source_step_norm_bound=1.,strong_baseline='NestedUnitDose;NestedPhysicsOrDose explicitlyallowsdosefallback',
        ties='prefer shared,closedinternalgate,strongershiftshrinkage',
        carrier_only_experiment=True,no_full_transport_or_monotonicity_claim=True,not_blind=True,
        actual_new_shared_fits=0,code_hashes=c.file_hashes(code),input_hashes=c.file_hashes(set(inputs))))
    clock=time.perf_counter();outer=[];selections=[]
    shift_policies=[(k,None,'shared') for k in SHIFT_LAMBDAS]
    gate_policies=[(k,t,'shared') for k,t in itertools.product(SHIFT_LAMBDAS,TAUS)]
    dose_policies=[(k,None,'dose') for k in SHIFT_LAMBDAS]
    for path,choice in choices:
        held=choice['outer_held'];fit,fp=c.fit(choice['train_units'],METHOD,choice['selected']['config'])
        assert fit['training_valid'] and fit['selected_success']
        rows=c.cases();rows=rows[rows.related_cluster.eq(held)]
        for budget in (0,1,2):
            scores={}
            if budget:
                for p in gate_policies+dose_policies:scores[p]=score_candidate(choice,budget,p)
            sets={'FrozenShared':[(None,None,'shared')], 'NestedShift':shift_policies,
                'NestedProfiledGate':gate_policies,'NestedPhysicsOrDose':gate_policies+dose_policies,
                'NestedUnitDose':dose_policies}
            for variant,candidates in sets.items():
                policy=(None,None,'dose' if variant=='NestedUnitDose' else 'shared') if budget==0 else min(candidates,key=lambda p:(round(scores.get(p,np.inf),10),policy_tie(p)))
                selections.append(dict(outer_held=held,budget=budget,variant=variant,policy=policy,
                    inner_score=scores.get(policy),shared_config=choice['selected']['config'],shared_fit_key=fit['key'],
                    candidate_scores=[dict(policy=p,score=scores.get(p)) for p in candidates]))
                for _,source in rows.groupby('family',sort=True):
                    rr=simulate(source,fit,budget,*policy)
                    for r in rr:r.update(variant=variant,outer_held=held,global_fit_key=fit['key'])
                    outer.extend(rr)
            print('Profiled adaptation',held,'budget',budget,'complete',flush=True)
        c.write(OUT/'status.json',dict(created_utc=c.e.now(),held_finished=held,outer_rows=len(outer),physical_calls=CALLS,seconds=time.perf_counter()-clock))
    frame=pd.DataFrame(outer);frame.to_csv(OUT/'predictions.csv',index=False)
    sums=[]
    for variant,budget,scope in itertools.product(VARIANTS,(0,1,2),('all_query','common20')):
        q=frame[frame.variant.eq(variant)&frame.budget.eq(budget)&frame.role.eq('query')]
        if scope=='common20':q=q[q.primary_common20.eq(True)]
        value=c.group_score(q.to_dict('records'),'nH',group_equal=False)
        sums.append(dict(variant=variant,budget=budget,scope=scope,rows=len(q),conditions=q.sample_id.nunique(),
            numeric=int(q.numeric.eq(True).sum()),valid=int(q.valid.eq(True).sum()),nH_MAE=value))
    pd.DataFrame(sums).to_csv(OUT/'summary.csv',index=False)
    c.write(OUT/'selections.json',selections);c.write(OUT/'updates.json',UPDATES)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'complete.json',dict(created_utc=c.e.now(),rows=len(frame),physical_calls=CALLS,working_updates=len(UPDATES),
        new_shared_fits=0,seconds=time.perf_counter()-clock,retrospective=True,stable_prediction_gain_established=False))


if __name__=='__main__':main()
