"""Thermal-input controlled, joint nH/mu training on the original specimen task."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import os,sys,json,time,argparse,copy
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from csv_io_v2 import read_result_csv as read
from sandbox.thermal_coherent_model import ThermalCoherentHallModel
from sandbox.coherent_fermi import published_coherent_profile,TABLE
import phase15_prediction as parent
import phase16_thermal_audit as thermal
import phase2_energy as energy

OUT=thermal.OUT;core=parent.core
REFERENCES=('HallExtra_original_RT','HallExtra_binding_joint')
STRUCTURES={}
for basis,ref in zip(('original_RT','binding_joint'),REFERENCES):
    for mode in ('effective','record_cap'):
        for gamma in (0.,.1,1.):
            name=ref if mode=='effective' and gamma==0 else f'Thermal_{basis}_{mode}_g{gamma:g}'
            STRUCTURES[name]=dict(**parent.NEW_METHODS[ref],thermal_mode=mode,gamma=gamma,reference_method=ref)
NEW_METHODS={m:d for m,d in STRUCTURES.items() if m not in REFERENCES}
core.METHODS.update(NEW_METHODS);core.OUT=OUT
MODELS={};FITS={};BANK={}


def inputs():
    audit=read(OUT/'thermal_audit/specimens.csv').set_index('sample_id')
    pred=core.PRED.copy();pred['endpoint']='nH';pred['thermal_cap_K']=pred.sample_id.map(audit.thermal_cap_K)
    rows=[];rt=pred[pred.kind.eq('RT')].set_index('sample_id')
    for r in core.DATA[core.DATA.mobility_eligible.eq(True)].itertuples():
        x=rt.loc[r.sample_id].to_dict();x.update(sample_id=r.sample_id,obs_id='MU_RT::'+r.sample_id,
            endpoint='muH',kind='RT',Tobs_K=float(r.measurement_T_K),target=float(np.log10(r.mu_cm2Vs)))
        assert x['Tobs_K']==300.;rows.append(x)
    obs=read(core.prior.prior.TEMPDIR/'temperature_observations.csv')
    for r in obs[obs.endpoint.eq('muH_cm2Vs')].itertuples():
        x=rt.loc[r.sample_id].to_dict();x.update(sample_id=r.sample_id,obs_id='MU_T::'+r.read_id,
            endpoint='muH',kind='curve',Tobs_K=float(r.T_K),target=float(np.log10(r.observed)))
        rows.append(x)
    mu=pd.DataFrame(rows)
    assert len(mu)==59 and mu.obs_id.is_unique and mu[mu.kind.eq('RT')].family.nunique()==2
    return pred,pd.concat([pred,mu],ignore_index=True)


PRED,TARGETS=inputs();core.PRED=PRED


def mapper(method):
    if method not in STRUCTURES:return parent.mapper(method)
    if method not in MODELS:
        MODELS[method]=ThermalCoherentHallModel(STRUCTURES[method],profile=published_coherent_profile(),pO2_atm=1e-5)
    return MODELS[method]


core.mapper=mapper


def training_data(families,method):
    d=STRUCTURES[method];data=TARGETS[TARGETS.family.isin(families)];pieces=[]
    for endpoint,kind,factor in (('nH','RT',1.),('nH','curve',.1 if d['training']=='joint' else 0.),
        ('muH','RT',d['gamma']),('muH','curve',.1*d['gamma'])):
        if factor==0:continue
        sub=data[data.endpoint.eq(endpoint)&data.kind.eq(kind)].copy()
        if kind=='curve':sub=sub[sub.Tobs_K.le(275)].copy()
        if sub.empty:continue
        sub['loss_block']=endpoint+'_'+kind
        sub['loss_weight']=factor/(sub.groupby('sample_id').obs_id.transform('count').to_numpy(float)*
            sub.groupby('family').sample_id.transform('nunique').to_numpy(float)*sub.family.nunique())
        pieces.append(sub)
    return pd.concat(pieces,ignore_index=True).sort_values(['family','sample_id','Tobs_K','endpoint','kind'])


def block_metadata(tr):
    return [dict(block=b,observations=len(g),specimens=g.sample_id.nunique(),families=sorted(g.family.unique()),
        sum_weight=float(g.loss_weight.sum()),temperatures_K=sorted(g.Tobs_K.unique())) for b,g in tr.groupby('loss_block')]


def forward_targets(model,w,table):
    cache={};values=[];jac=[]
    for row in table.itertuples():
        key=(row.sample_id,float(row.Tobs_K))
        if key not in cache:cache[key]=model.point(row,w)
        a=cache[key]
        values.append(a.values[0] if row.endpoint=='nH' else a.transport_values[0])
        jac.append(a.jacobian[0] if row.endpoint=='nH' else a.transport_jacobian[0])
    return np.array(values),np.array(jac)


def saved_predictions(model,parameters):
    predictions=[];properties=[];cached={}
    for row in TARGETS.itertuples():
        try:
            if parameters is None:raise RuntimeError('all starts failed')
            key=(row.sample_id,float(row.Tobs_K))
            if key not in cached:cached[key]=model.point(row,parameters)
            a=cached[key];g=a.diagnostics
            dg={k:g[k] for k in ('root_residual_log','jacobian_condition','free_n_cm3','free_p_cm3','Tf_K',
                'measurement_T_K','Nd_cm3','profile_sha256','localized_fraction','donor_subgroups',
                'same_physical_state_for_output','hall_factor')}
            dg['thermal_mode']=model.thermal_mode;dg['thermal_cap_K']=float(row.thermal_cap_K)
            record=dict(obs_id=row.obs_id,predicted=float(a.values[0] if row.endpoint=='nH' else a.transport_values[0]),
                predicted_log_nH=float(a.values[0]),charge_outputs=a.values[1:],transport_values=a.transport_values,
                numeric=True,physical_valid=True,diagnostics=dg)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            record=dict(obs_id=row.obs_id,predicted=None,numeric=False,physical_valid=False,error=repr(exc))
        (predictions if row.endpoint=='nH' else properties).append(record)
    return predictions,properties


def oldbank():
    global BANK
    if not BANK:
        BANK={(f['method'],tuple(sorted(f['train_families'])),json.dumps(f['config'],sort_keys=True)):f for f in parent.composite().values()}
    return BANK


def fit(families,method,config):
    families=tuple(sorted(families))
    if method not in STRUCTURES:return oldbank()[method,families,json.dumps(config,sort_keys=True)]
    key=core.prior.old.digest(dict(protocol=energy.sha(OUT/'prediction_protocol.json'),
        train_families=families,method=method,config=config))
    path=OUT/('inherited' if method in REFERENCES else 'fits')/(key+'.json')
    if key in FITS:return FITS[key]
    if path.exists():FITS[key]=json.loads(path.read_text());return FITS[key]
    tr=training_data(families,method);weights=tr.loss_weight.to_numpy(float);sw=np.sqrt(weights);y=tr.target.to_numpy(float)
    model=mapper(method);before=model.counters();clock=time.perf_counter();failures=[];trials=[];parameters=None
    if method in REFERENCES:
        old=oldbank()[method,families,json.dumps(config,sort_keys=True)]
        parameters=old['parameters'];trials=copy.deepcopy(old['trials'])
        inherited=dict(fit_key=old['key'],origin=old['composite_origin'],training_reused=True,
            new_training_starts=0,new_target_labels_used=False)
    else:
        inherited=None;lo,hi,starts=core.specification(method);cache={}
        def calc(w):
            k=tuple(w)
            if k not in cache:
                try:value=forward_targets(model,w,tr)
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    failures.append(dict(parameters=w,error=repr(exc)));raise
                cache.clear();cache[k]=value
            return cache[k]
        for initial in starts:
            try:
                op=least_squares(lambda w:(calc(w)[0]-y)*sw,initial,jac=lambda w:calc(w)[1]*sw[:,None],
                    bounds=(lo,hi),max_nfev=120,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                trials.append(dict(initial=initial,parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                    nfev=int(op.nfev),njev=int(op.njev),optimality=float(op.optimality),active_mask=op.active_mask,message=op.message))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                trials.append(dict(initial=initial,parameters=initial,loss=None,success=False,error=repr(exc)))
        finite=[t for t in trials if t['loss'] is not None]
        if finite:parameters=min(finite,key=lambda t:t['loss'])['parameters']
    training_counters={k:model.counters()[k]-before[k] for k in before}
    pred,properties=saved_predictions(model,parameters)
    loss_blocks=[]
    if parameters is not None:
        values,_=forward_targets(model,parameters,tr);errors=values-y
        for b in tr.loss_block.unique():
            mask=tr.loss_block.eq(b).to_numpy();loss_blocks.append(dict(block=b,weighted_SSE=float(weights[mask]@(errors[mask]**2))))
    f=dict(key=key,method=method,definition=STRUCTURES[method],config=config,train_families=families,parameters=parameters,
        parameter_names=model.parameter_names,trials=trials,state_failures=failures,inherited=inherited,
        training_obs_ids=tr.obs_id.tolist(),training_specimens=sorted(tr.sample_id.unique()),
        weight_per_observation=weights,training_blocks=block_metadata(tr),selected_loss_blocks=loss_blocks,
        metadata=dict(mu_training_families=sorted(tr[tr.endpoint.eq('muH')].family.unique()),
            objective='composite predictive squared log10 errors;not an independent Gaussian likelihood'),
        predictions=pred,property_predictions=properties,seconds=time.perf_counter()-clock,
        training_counters=training_counters,counters={k:model.counters()[k]-before[k] for k in before})
    energy.write(path,f);FITS[key]=energy.clean(f)
    print('thermaljoint',method,len(families),'seconds',round(f['seconds'],2),
        'starts',0 if inherited else len(trials),'stop',sum(bool(t.get('success')) for t in trials),flush=True)
    return FITS[key]


core.fit=fit


def freeze():
    gate=json.loads((OUT/'numerical_gates.json').read_text());assert gate['passed']
    for path,digest in gate['code_hashes'].items():assert energy.sha(ROOT/path)==digest,path
    sources=(Path(__file__),ROOT/'src/sandbox/thermal_coherent_model.py',Path(thermal.__file__),
        ROOT/'scripts/trainable_sandbox_20261017/phase16_gates.py',ROOT/'docs/ML_SANDBOX_THERMAL_JOINT_20261018.md',
        ROOT/'src/sandbox/coherent_hall_model.py',ROOT/'src/sandbox/donor_mixture_model.py',
        ROOT/'src/sandbox/coherent_fermi.py',ROOT/'src/sandbox/coherent_transport.py')
    inputs=(OUT/'thermal_audit/audit.json',OUT/'thermal_audit/specimens.csv',OUT/'numerical_gates.json',
        energy.BASE/'dataset_v1.csv',energy.BASE/'episodes_v1.csv',core.prior.prior.TEMPDIR/'paired_specimens.csv',
        core.prior.prior.TEMPDIR/'temperature_observations.csv',TABLE,energy.BASE/'artifact_manifest_phase15.json')
    jobs=[dict(method=m,train_families=f) for n in (3,4) for f in combinations(core.FAMILIES,n) for m in NEW_METHODS]
    payload=dict(version=1,created_utc=energy.now(),structures=STRUCTURES,new_methods=NEW_METHODS,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in sources},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        new_jobs=jobs,new_fit_count=150,new_starts=300,inherited_fit_count=30,
        primary='original29Si/Sn5DOI,0/1/2sample nHanchors,29/23/18queries,common18',
        loss='nRT + beta_n*nT + gamma*(muRT+.1muT);beta_n0 originalRT,.1 E(C);allblocks DOI/specimen/read equal',
        temperature_training='curveT<=275K;each target at ownT;no observed nH feature for mu',
        cap='conditional reported film growth/intervalupper andpostfilm anneal;no prep substrateT asfilmTmax',
        optimizer='same2coordinates120max_nfev,1e-7stopping;ALLfailed-startfits uniformly eligible360refinement',
        specifications={m:dict(names=mapper(m).parameter_names,bounds=core.specification(m)[:2],starts=core.specification(m)[2]) for m in STRUCTURES},
        adaptation='nHoutputoffset only;same7lambda innerDOIselection;mu not used in anchors;gamma fixed ablation,not outerselected',
        inherited=[dict(method=m,train_families=f,fit_key=oldbank()[m,tuple(f),'{}']['key'],
            **oldbank()[m,tuple(f),'{}']['composite_origin']) for n in (3,4) for f in combinations(core.FAMILIES,n) for m in REFERENCES],
        evidence='retrospective,muRT5/2DOI,muT54reads/3oldspecimens/1DOI;unknown covariance;205 empirical prior;not a likelihood or physical freeze identification',
        new_sigma_labels=0,new_DFT_MD=0,new_material_measurements=0,goal_complete=False)
    path=OUT/'prediction_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        for key in ('structures','code_hashes','input_hashes','inherited','specifications'):assert old[key]==energy.clean(payload[key]),key
    else:energy.write(path,payload)


def all_fits():
    result={}
    for folder in ('fits','inherited'):
        for path in (OUT/folder).glob('*.json'):
            f=json.loads(path.read_text());f['composite_origin']=dict(path=str(path.relative_to(ROOT)),sha256=energy.sha(path));result[f['key']]=f
    return result


def worker(job):return fit(*job)['key']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--fit-only',action='store_true');a=ap.parse_args()
    freeze();jobs=[(families,m,{}) for n in (3,4) for families in combinations(core.FAMILIES,n) for m in STRUCTURES]
    assert len(jobs)==180
    energy.write(OUT/'training_execution_started.json',dict(created_utc=energy.now(),workers=a.workers,new_fits=150,new_starts=300,inherited=30))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(worker,jobs):pass
    energy.write(OUT/'training_execution_complete.json',dict(created_utc=energy.now(),records=len(all_fits()),goal_complete=False))
    if not a.fit_only:
        for method in STRUCTURES:core.select(method)
        energy.write(OUT/'selection_execution_complete.json',dict(created_utc=energy.now(),methods=len(STRUCTURES),goal_complete=False))


if __name__=='__main__':main()
