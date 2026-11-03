"""Train actual paired-observable models on whole-source-excluded contexts."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os,sys,json,time,argparse
import numpy as np
from scipy.optimize import least_squares
from scipy.special import logit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase29_learning as l
import phase29_data as data
import phase2_energy as e

OUT=l.OUT
TABLE=None


def table():
    global TABLE
    if TABLE is None:TABLE=data.load_cases()
    return TABLE


def subset(units):return table()[table().related_cluster.isin(units)].copy()


def fit_path(units,method,config):
    key=l.digest(dict(protocol=e.sha(OUT/'training_protocol.json'),units=sorted(units),method=method,config=config))
    return key,OUT/'fits'/(key+'.json')


def base_fit(units):
    key,path=fit_path(units,'Reference205',dict(gamma=0.,alpha=0.))
    assert path.exists(),'Reference nH-only initializer prerequisite absent'
    value=json.loads(path.read_text());assert value['train_units']==sorted(units)
    if value['parameters'] is None:raise RuntimeError('Both reference initializers failed;preserve before a versioned initialization rule')
    return value,path


def starts(units,method):
    if method=='Reference205':
        second=logit((np.log10(1750.)-np.log10(600.))/(np.log10(1950.)-np.log10(600.)))
        return [np.array([.5,16.]),np.array([second,15.5])],None
    spec=l.SPECS[method];base,path=base_fit(units);w=np.asarray(base['parameters'])
    if spec['kind']=='reference':values=[np.r_[w,0.],np.r_[w,-1.]]
    elif spec['kind']=='activation':
        values=[np.array([15.,0.,-.7,0.,0.,0.,0.]),np.array([16.,0.,1.4,0.,0.,0.,-1.])]
    else:
        nfield={'constant':1,'dose':2,'process':4}[spec['basis']]
        values=[np.r_[w,c,np.zeros(nfield-1),extra] if spec['learn_scale'] else np.r_[w,c,np.zeros(nfield-1)]
            for c,extra in ((-.7,0.),(1.4,-1.))]
    return values,dict(path=str(path.relative_to(ROOT)),sha256=e.sha(path),key=base['key'],
        scope='new same-training-context nH-only reference initializer;not reused outer labels')


def fit(units,method,config):
    units=tuple(sorted(units));key,path=fit_path(units,method,config)
    if path.exists():
        value=json.loads(path.read_text())
        assert value['method']==method and value['train_units']==list(units) and value['config']==config
        return value
    tr=subset(units);clock=time.perf_counter();wn=l.weights(tr);paired,rho=l.observed_pair(tr)
    record=dict(key=key,method=method,config=config,train_units=units,train_families=sorted(tr.family.unique()),
        training_condition_ids=sorted(tr.sample_id.unique()),training_case_ids=tr.case_id.tolist(),
        physical_reported_condition_count=tr.sample_id.nunique(),scenario_rows=len(tr),
        paired_reported_conditions=tr.loc[paired,'sample_id'].nunique(),
        training_nH_case_weights=wn.tolist(),training_rho_case_weights=l.weights(tr,paired_only=True).tolist(),
        loss_observables='one Hall coefficient direction plus paired log-rho;no third independent sigma',
        parameters=None,trials=[],state_failures=[],new_optimizer_starts=0,new_linear_solves=0,
        selected_success=False,training_valid=False,goal_complete=False)
    if method in l.NONLINEAR:
        initial,reference=starts(units,method);record['initializer_reference']=reference
        lo,hi=l.bounds(method);cache={};failures=[];calls=0;point_calls=0;continuations=0
        tracked={method}
        s=l.SPECS[method]
        if s['kind']=='share':tracked.add('ReferenceTau' if s['learn_scale'] else 'Reference205')
        before={m:l.model(m).counters() for m in tracked}
        def calc(w):
            nonlocal calls,point_calls,continuations
            k=tuple(w)
            if k not in cache:
                calls+=1
                try:q=l.forward(method,w,tr,penalty=config['alpha']>0)
                except (ValueError,RuntimeError,ArithmeticError) as exc:
                    failures.append(dict(parameters=np.asarray(w).tolist(),error=repr(exc)));raise
                point_calls+=q['physical_point_calls'];continuations+=int((~q['valid']).sum())
                residual,J,_=l.loss_arrays(method,w,tr,config,calculated=q)
                cache.clear();cache[k]=(residual,J,q)
            return cache[k]
        def run(initial,cap,stage):
            try:
                op=least_squares(lambda w:calc(w)[0],initial,jac=lambda w:calc(w)[1],
                    bounds=(lo,hi),x_scale='jac',max_nfev=cap,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                return dict(initial=np.asarray(initial).tolist(),parameters=op.x.tolist(),loss=float(op.fun@op.fun),
                    success=bool(op.success),nfev=int(op.nfev),optimality=float(op.optimality),
                    message=op.message,stage=stage,cap=cap,active_bounds=op.active_mask.tolist())
            except (ValueError,RuntimeError,ArithmeticError) as exc:
                return dict(initial=np.asarray(initial).tolist(),loss=None,success=False,error=repr(exc),stage=stage,cap=cap)
        trials=[run(x,120,'initial') for x in initial]
        if any(not x['success'] for x in trials):
            finite=[x for x in trials if x['loss'] is not None]
            best=[min(finite,key=lambda x:x['loss'])['parameters']] if finite else []
            trials.extend(run(x,360,'uniform_refinement') for x in initial+best)
        finite=[x for x in trials if x['loss'] is not None]
        chosen=min(finite,key=lambda x:x['loss']) if finite else None
        record.update(trials=trials,state_failures=failures,new_optimizer_starts=len(trials),
            selected_success=bool(chosen and chosen['success']),selected_loss=chosen['loss'] if chosen else None,
            objective_forward_calls=calls,physical_point_calls=point_calls,
            numeric_continuation_trial_case_rows=continuations)
        if chosen:
            optimization_calls=calls;optimization_points=point_calls
            w=np.asarray(chosen['parameters']);r,J,q=calc(w);v=q['values']
            record.update(optimization_objective_forward_calls=optimization_calls,
                optimization_physical_point_calls=optimization_points,
                selection_replay_forward_calls=calls-optimization_calls,
                selection_replay_physical_point_calls=point_calls-optimization_points,
                objective_forward_calls=calls,physical_point_calls=point_calls,
                numeric_continuation_trial_case_rows=continuations)
            data_rows=len(tr)+(int(paired.sum()) if config['gamma']>0 else 0)
            dataJ=J[:data_rows]
            sv=np.linalg.svd(dataJ,compute_uv=False)
            record.update(parameters=w.tolist(),training_valid=bool(q['valid'].all()),
                training_nH_MAE=float(wn@abs(v[:,0]-tr.target.to_numpy(float))),
                training_rho_MAE=float(l.weights(tr,paired_only=True)[paired]@abs(-v[paired,2]-rho[paired])) if paired.any() else None,
                selected_training_values=v.tolist(),selected_training_valid=q['valid'].tolist(),
                observation_jacobian_singular_values=sv.tolist(),
                observation_jacobian_rank=int(np.linalg.matrix_rank(dataJ)),
                regularized_jacobian_rank=int(np.linalg.matrix_rank(J)),
                parameter_names=l.model(method).parameter_names,
                solver_counters={m:{k:l.model(m).counters()[k]-x for k,x in old.items()} for m,old in before.items()})
    elif method=='DirectMultiRidge':
        direct=l.fit_direct(tr,float(config['ridge_alpha']));v=l.direct_values(direct,tr)
        record.update(direct,new_linear_solves=direct['new_linear_solves'],training_valid=bool(np.isfinite(v).all() and (v[:,0]>=13).all()),
            selected_success=True,selected_training_values=v.tolist(),
            selected_training_valid=(np.isfinite(v).all(axis=1)&(v[:,0]>=13)).tolist())
    else:
        empirical=l.fit_empirical(method,tr);v=l.empirical_values(method,empirical,tr)
        record.update(empirical,training_valid=bool(np.isfinite(v).all() and (v[:,0]>=13).all()),
            selected_success=True,new_linear_solves=int(method=='DoseLine'),
            selected_training_values=v.tolist(),selected_training_valid=(np.isfinite(v).all(axis=1)&(v[:,0]>=13)).tolist())
    record['seconds']=time.perf_counter()-clock;e.write(path,record)
    print('fit29',method,config,'groups',len(units),'sec',round(record['seconds'],2),
        'valid',record['training_valid'],'stopped',record['selected_success'],flush=True)
    return e.clean(record)


def declare():
    assert json.loads((OUT/'data/qualification.json').read_text())['passed']
    assert json.loads((OUT/'numerical_gates_v2/gates.json').read_text())['passed']
    methods={m:(l.SPECS[m] if m in l.SPECS else dict(kind='direct_multi' if m=='DirectMultiRidge' else 'empirical')) for m in l.METHODS}
    code=[Path(__file__),Path(__file__).with_name('phase29_learning.py'),Path(__file__).with_name('phase29_data.py'),
        ROOT/'src/sandbox/reference_scattering.py',ROOT/'src/sandbox/activation_transport_control.py',
        ROOT/'src/sandbox/joint_inventory_scattering.py',ROOT/'src/sandbox/learnable_constant_tau.py',
        ROOT/'src/sandbox/paired_hall_observations.py',ROOT/'src/sandbox/charge_feasible_model_v2.py',
        ROOT/'src/sandbox/charge_feasible_model.py',ROOT/'src/sandbox/relative_inventory.py',
        ROOT/'src/sandbox/context_saved_model.py',ROOT/'src/sandbox/coherent_transport.py']
    inputs=[ROOT/'docs/ML_SANDBOX_GOAL_CHECK_PHASE29_20261019.md',
        ROOT/'docs/ML_SANDBOX_JOINT_TRAINING_MATH_PHASE29_20261019.md',
        OUT/'system_goal_direction_receipt_v1.json',OUT/'gate_difference_diagnosis_v1.json',
        OUT/'data/qualification.json',OUT/'data/conditions35.csv',OUT/'data/scenarios49.csv',
        OUT/'data/episodes35.csv',OUT/'numerical_gates_v2/gates.json',OUT/'numerical_gates_v2/protocol_before_gates.json',
        e.BASE/'artifact_manifest_phase28.json']
    contexts=l.contexts();assert len(contexts)==16
    total=sum(len(l.configs(m)) for m in l.METHODS)*len(contexts)
    nonlinear=sum(len(l.configs(m)) for m in l.NONLINEAR)*len(contexts)
    assert total==1392 and nonlinear==1200
    value=dict(created_utc=e.now(),methods=methods,configs={m:l.configs(m) for m in l.METHODS},
        primary='JointProcess_internal',primary_global_method='JointProcess',
        reported_conditions=35,scenario_rows=49,paired_reported_conditions=11,paired_scenario_rows=25,
        DOIs=7,conservative_related_groups=5,validation_scheme='cluster',fit_contexts=contexts,
        original33_roles_common20_preserved=True,source_budget_perDOI=[0,1,2],total_anchor_budgets=[0,7,14],
        total_allquery_reported_conditions=[35,27,20],Gogova_2anchor_nH_queries=0,
        scenario_loss='average squared residual over fixed equally weighted8settings pernewcondition;weights total1;not measured process distribution',
        nH_weights='equalcluster/equalDOIwithincluster/equalreportedconditionwithinDOI;times normalizedscenario weight',
        rho_weights='same hierarchy among pairedconditions only;two primitive directions,no duplicateHall or independent sigma',
        response_regularizer='alpha times training-input mean square nH difference from sameTf,bg,mu0 reference plus shared scale shift squared;statistical not thermodynamic',
        empirical_regularizer='activation response versus zero-field activation prior plus output mu-scale squared',
        global_selection='inner held-related-group zero-anchor condition-averaged nH MAE;no outer label or mu in selection metric',
        source_selection='training-inner query nH MAE;perDOI0/1/2Hall labels only;lambda[None,0,.01,.1,1,10]',
        main_evaluation='originalcommon20 DOI-macro nH;related-group macro andallquery/source results also retained',
        transport_evaluation='original/new pairedfullchainqueries separately;muatHallanchors is separate conditional task;sigma derived',
        optimizer='same2starts120;onanynonstop original2+best360;ftol/xtol/gtol1e-7;minimum finite loss selected even if that best remains nonstopped',
        initialization='new same-training-data Reference205 gamma0 refit;fields-.7or1.4 andscale0or-1;activation fixedstarts;no outer-fit initialization',
        total_fit_records=total,nonlinear_fit_records=nonlinear,direct_multi_fit_records=144,
        simple_empirical_fit_records=48,nominal_nonlinear_initial_starts=2400,
        nonlinear_candidate_budget='9 pernonlinear family except parameter-free-regularization Reference2053;DirectMultiRidge9;closedforms reported separately',
        bounds='Tf_logit[-6,6],physicalbg[14,18],inventorycoeff[-12,12],log10mu0[-6,6];numerical ranges,not physical accuracy boundaries',
        new_DFT_MD=0,new_material_measurements=0,GPU_used=False,GPU_authorized='GPU1only',
        actual_new_Sn_inputs_incomplete=True,retrospective=True,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs})
    path=OUT/'training_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        for k in ('code_hashes','input_hashes','methods','configs','fit_contexts'):assert old[k]==e.clean(value[k]),k
        return old
    e.write(path,value);return value


def task(args):return fit(*args)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args();protocol=declare()
    if args.declare_only:
        print('Declared1392fitrecords/1200nonlinear;outer predictions not started',flush=True);return
    for kind in ('code_hashes','input_hashes'):
        for name,digest in protocol[kind].items():assert e.sha(ROOT/name)==digest,name
    assert not (OUT/'training_started.json').exists(),'inspect actual running handle;no blind restart'
    first=[(u,'Reference205',dict(gamma=0.,alpha=0.)) for u in l.contexts()]
    later=[(u,m,c) for u in l.contexts() for m in l.METHODS for c in l.configs(m)
        if not (m=='Reference205' and c==dict(gamma=0.,alpha=0.))]
    assert len(first)==16 and len(later)==1376
    e.write(OUT/'training_started.json',dict(created_utc=e.now(),pid=os.getpid(),workers=args.workers,
        initial_reference_records=len(first),later_records=len(later),planned_records=1392,goal_complete=False))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        complete=list(pool.map(task,first))
        e.write(OUT/'initial_references_complete.json',dict(created_utc=e.now(),records=16,
            not_stopped=sum(not f['selected_success'] for f in complete),
            invalid=sum(not f['training_valid'] for f in complete)))
        complete+=list(pool.map(task,later))
    assert len(complete)==1392
    result=dict(created_utc=e.now(),fit_records=len(complete),nonlinear_fit_records=1200,
        actual_optimizer_starts=sum(f['new_optimizer_starts'] for f in complete),
        new_linear_solves=sum(f['new_linear_solves'] for f in complete),
        selected_nonstopped=sum(not f['selected_success'] for f in complete),
        invalid_training_fits=sum(not f['training_valid'] for f in complete),
        all_failures_retained=True,outer_predictions_started=False,goal_complete=False)
    e.write(OUT/'training_complete.json',result);print('Training29complete',result,flush=True)


if __name__=='__main__':main()
