"""Train conserved parent allocation on the unchanged actual-dose task."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import os,sys,json,time,argparse
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit,logit
from sandbox.inventory_coherent_model import InventoryCoherentModel
from sandbox.coherent_fermi import published_coherent_profile,TABLE
from sandbox import transport
import phase16_refinement as prior
import phase2_energy as energy

OUT=energy.BASE/'phase17_inventory_learning';core=prior.study.core;PRED=prior.study.PRED;TARGETS=prior.study.TARGETS
FAMILIES=core.FAMILIES;MODELS={};FITS={};BANK={}
STRUCTURES={};REFERENCES={}
for basis,definition in [('original_RT',dict(fraction='zero',training='RT')),
    ('binding_joint',dict(fraction='one',binding='screen',training='joint'))]:
    common=dict(**definition,readout='ionized_extra')
    ref='Inventory_reference_'+basis
    STRUCTURES[ref]=dict(**common,inventory_kind='none',relation='constant')
    REFERENCES[ref]='Thermal_'+basis+'_record_cap_g0'
    for kind,relations in [('native',('constant',)),('complex',('constant','dose_plus','dose_minus')),('output',('constant','dose_plus','dose_minus'))]:
        for relation in relations:STRUCTURES['Stock_'+kind+'_'+relation+'_'+basis]=dict(**common,inventory_kind=kind,relation=relation)
STRUCTURES['activation_sigmoid_RT']=dict(empirical='activation_sigmoid',training='RT')
NEW_METHODS={m:d for m,d in STRUCTURES.items() if m not in REFERENCES}
OLD_MAPPER=core.mapper;core.METHODS.update(STRUCTURES);core.OUT=OUT


def mapper(method):
    if method not in STRUCTURES:return OLD_MAPPER(method)
    if 'empirical' in STRUCTURES[method]:raise ValueError('Direct activation baseline has no defect mapper')
    if method not in MODELS:MODELS[method]=InventoryCoherentModel(STRUCTURES[method],profile=published_coherent_profile(),pO2_atm=1e-5)
    return MODELS[method]


core.mapper=mapper


def specification(method):
    if 'empirical' in STRUCTURES[method]:
        return np.array([14.,-12.,-12.]),np.array([18.,12.,12.]),[[16.,logit(.2),-1.],[15.5,logit(.7),-.1]]
    m=mapper(method);lo=[14.];hi=[18.];starts=[]
    if m.iE is not None:lo.extend([np.log10(.003),0.]);hi.extend([np.log10(.2),5.])
    if m.ic0 is not None:lo.append(-6. if m.kind=='output' else 0.);hi.append(6. if m.kind=='output' else 12.)
    if m.ic1 is not None:lo.append(-6. if m.kind=='output' else 0.);hi.append(6. if m.kind=='output' else 12.)
    for B,E,screen,c0,c1,output in [(16.,.03,.2,.03,.1,0.),(15.5,.06,1.,.3,1.,3.)]:
        w=[B]
        if m.iE is not None:w.extend([np.log10(E),screen])
        if m.ic0 is not None:w.append(output if m.kind=='output' else c0)
        if m.ic1 is not None:w.append(0. if m.kind=='output' else c1)
        starts.append(w)
    return np.array(lo),np.array(hi),starts


def training_data(families,method):
    data=PRED[PRED.family.isin(families)];rt=data[data.kind.eq('RT')].copy()
    rt['loss_weight']=1/(rt.groupby('family').sample_id.transform('nunique').to_numpy(float)*rt.family.nunique())
    rt['loss_block']='nH_RT'
    if STRUCTURES[method]['training']=='RT':return rt.sort_values(['family','sample_id','Tobs_K','kind'])
    c=data[data.kind.eq('curve')&data.Tobs_K.le(275)].copy()
    if len(c):c['loss_weight']=.1/(c.groupby('sample_id').obs_id.transform('count').to_numpy(float)*c.groupby('family').sample_id.transform('nunique').to_numpy(float)*c.family.nunique())
    c['loss_block']='nH_curve'
    return pd.concat([rt,c],ignore_index=True).sort_values(['family','sample_id','Tobs_K','kind'])


def activation(parameters,table):
    B,a,b=np.asarray(parameters,float);C=table.incorporated_cm3.to_numpy(float);H=table.H_cm3.to_numpy(float)
    total=C+H+10**B;x=np.log10(C)-18.;coordinate=a+b*x
    logf=-np.logaddexp(0.,-coordinate);one_minus_f=expit(-coordinate)
    value=np.log10(total)+logf/np.log(10.)
    jac=np.c_[np.full(len(C),10**B)/total,one_minus_f/np.log(10.),x*one_minus_f/np.log(10.)]
    return value,jac


def forward_targets(method,parameters,table):
    if 'empirical' in STRUCTURES[method]:return activation(parameters,table)
    m=mapper(method);values=[];jac=[]
    for r in table.itertuples():
        a=m.point(r,parameters);values.append(a.values[0]);jac.append(a.jacobian[0])
    return np.array(values),np.array(jac)


def saved_predictions(method,parameters):
    n=[];mu=[];physical='empirical' not in STRUCTURES[method];cache={}
    if not physical and parameters is not None:empirical,_=activation(parameters,TARGETS)
    for j,row in enumerate(TARGETS.itertuples()):
        try:
            if parameters is None:raise RuntimeError('all starts failed')
            if physical:
                key=(row.sample_id,float(row.Tobs_K))
                if key not in cache:cache[key]=mapper(method).point(row,parameters)
                a=cache[key];g=a.diagnostics;logn=float(a.values[0]);transport_values=a.transport_values;charges=a.values[1:]
                dg={k:g[k] for k in ('root_residual_log','jacobian_condition','free_n_cm3','free_p_cm3','Tf_K','measurement_T_K',
                    'Nd_cm3','profile_sha256','localized_fraction','donor_subgroups','same_physical_state_for_output','hall_factor',
                    'native_survival','complex_survival','complex_reallocation_fraction','foreign_atom_relative_error',
                    'foreign_atom_derivative_error','host_Ga_O_inventory_change_cm3','thermal_cap_K','hot_polish_steps','initial_hot_root_residual')}
                valid=True if STRUCTURES[method]['inventory_kind']!='output' else None
            else:
                logn=float(empirical[j]);logmu=float(np.log10(transport.mu_caughey_thomas(10**logn,float(row.Tobs_K))))
                transport_values=[logmu,np.log10(transport.E)+logn+logmu];charges=None;valid=None
                dg=dict(same_physical_state_for_output=False,property_scope='direct bounded activation prior with legacy empirical CT(nHargument,T);no defect state',
                    thermal_cap_K=float(row.thermal_cap_K),measurement_T_K=float(row.Tobs_K))
            value=logn if row.endpoint=='nH' else float(transport_values[0])
            p=dict(obs_id=row.obs_id,predicted=value,predicted_log_nH=logn,charge_outputs=charges,
                transport_values=transport_values,numeric=True,physical_valid=valid,diagnostics=dg)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            p=dict(obs_id=row.obs_id,predicted=None,numeric=False,physical_valid=False,error=repr(exc))
        (n if row.endpoint=='nH' else mu).append(p)
    return n,mu


def oldbank():
    global BANK
    if not BANK:
        old=list(prior.study.parent.composite().values())+list(prior.composite().values())
        BANK={(f['method'],tuple(sorted(f['train_families'])),json.dumps(f['config'],sort_keys=True)):f for f in old}
    return BANK


def fit(families,method,config):
    families=tuple(sorted(families))
    if method not in STRUCTURES:return oldbank()[method,families,json.dumps(config,sort_keys=True)]
    key=core.prior.old.digest(dict(protocol=energy.sha(OUT/'prediction_protocol.json'),method=method,train_families=families,config=config))
    path=OUT/('inherited' if method in REFERENCES else 'fits')/(key+'.json')
    if key in FITS:return FITS[key]
    if path.exists():FITS[key]=json.loads(path.read_text());return FITS[key]
    tr=training_data(families,method);weights=tr.loss_weight.to_numpy(float);sw=np.sqrt(weights);y=tr.target.to_numpy(float)
    clock=time.perf_counter();trials=[];failures=[];parameters=None;inherited=None
    physical='empirical' not in STRUCTURES[method];m=mapper(method) if physical else None;before=m.counters() if physical else None
    if method in REFERENCES:
        old=oldbank()[REFERENCES[method],families,'{}'];parameters=old['parameters'][1:]
        values,_=forward_targets(method,parameters,tr);loss=float(weights@((values-y)**2))
        inherited=dict(method=old['method'],key=old['key'],origin=old['composite_origin'],original_trials=old['trials'],
            parameter_conversion='drop logitTf,fix at6;same remainingparameters;new polished-equation forwardaudit,no optimizer')
        trials=[dict(initial=parameters,parameters=parameters,loss=loss,success=True,nfev=0,njev=0,inherited=True)]
    else:
        lo,hi,starts=specification(method);cache={}
        def calc(w):
            k=tuple(w)
            if k not in cache:
                try:out=forward_targets(method,w,tr)
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    failures.append(dict(parameters=w,error=repr(exc)));raise
                cache.clear();cache[k]=out
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
    train_counts=None if before is None else {k:m.counters()[k]-before[k] for k in before}
    pred,properties=saved_predictions(method,parameters)
    f=dict(key=key,method=method,definition=STRUCTURES[method],config=config,train_families=families,parameters=parameters,
        parameter_names=m.parameter_names if m else ['log10_background','activation_intercept','activation_slope'],
        trials=trials,state_failures=failures,inherited=inherited,
        training_obs_ids=tr.obs_id.tolist(),training_specimens=sorted(tr.sample_id.unique()),weight_per_observation=weights,
        training_blocks=[dict(block=b,observations=len(g),sum_weight=float(g.loss_weight.sum())) for b,g in tr.groupby('loss_block')],
        predictions=pred,property_predictions=properties,seconds=time.perf_counter()-clock,training_counters=train_counts,
        counters=None if before is None else {k:m.counters()[k]-before[k] for k in before},
        metadata=dict(new_mu_supervision=False,property_scope='conditional actualdose-to-property forecast;not measured speciation or reactionrates'))
    energy.write(path,f);FITS[key]=energy.clean(f)
    print('inventoryfit',method,len(families),'seconds',round(f['seconds'],2),'newstarts',0 if inherited else 2,
        'stop',sum(bool(t['success']) for t in trials),flush=True)
    return FITS[key]


core.fit=fit


def freeze():
    gate=json.loads((OUT/'numerical_gates.json').read_text());assert gate['passed']
    for p,d in gate['code_hashes'].items():assert energy.sha(ROOT/p)==d,p
    sources=(Path(__file__),ROOT/'src/sandbox/fixed_inventory_charge.py',ROOT/'src/sandbox/inventory_coherent_model.py',
        ROOT/'scripts/trainable_sandbox_20261017/phase17_gates.py',ROOT/'docs/ML_SANDBOX_INVENTORY_LEARNING_20261018.md',
        ROOT/'src/sandbox/donor_mixture.py',ROOT/'src/sandbox/donor_mixture_model.py',ROOT/'src/sandbox/coherent_transport.py')
    inputs=(OUT/'numerical_gates.json',OUT/'inventory_audit/contract_audit.json',OUT/'inventory_audit/state_contributions.csv',
        energy.BASE/'artifact_manifest_phase16.json',energy.BASE/'dataset_v1.csv',energy.BASE/'episodes_v1.csv',
        core.prior.prior.TEMPDIR/'paired_specimens.csv',core.prior.prior.TEMPDIR/'temperature_observations.csv',
        prior.study.OUT/'thermal_audit/specimens.csv',TABLE)
    payload=dict(created_utc=energy.now(),version=1,structures=STRUCTURES,new_methods=NEW_METHODS,references=REFERENCES,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in sources},input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        new_fits=225,new_starts=450,inherited_references=30,primary='unchanged29Si/Sn5DOI/0-1-2nHspecimenanchors/29-23-18queries/common18',
        loss='sourceequalRT orRT+.1source/specimen/read equalcurve<=275K;no mu supervision',
        learner='sharedconstant/softplusdose native survival or foreign-atom-preserving complex allocation;sameparametercount outputcontrols',
        allocator='referenceisolatedreceiverfractions+theirderivatives;Ga/Oopeninventoryendpointnot kineticpathway',
        reference_precision='samehot equations,localNewtonpolish beforeinventoryIFT;oldlibrary/results unchanged',
        specifications={m:dict(bounds=specification(m)[:2],starts=specification(m)[2]) for m in STRUCTURES},
        optimizer='2starts,120max_nfev,ftol/xtol/gtol1e-7;ALLfailedstartfits uniform360refinement',
        adaptation='same7lambda outputnHoffset,innerwholeDOI,0-1-2anchors;no newsourcephysicaladaptation ininitialbatch',
        transport='sameFD/Nq2/free-onlyscreening/205 empiricalextra/Hall;fullchainmuRT5/2,muT54reads/3/1;independentsigma0',
        selection='allfixedmodelsreported;no outerbest family or basis choice',retrospective=True,new_DFT_MD=0,new_physical_measurements=0,goal_complete=False)
    p=OUT/'prediction_protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        for k in ('structures','code_hashes','input_hashes','specifications'):assert old[k]==energy.clean(payload[k]),k
    else:energy.write(p,payload)


def all_fits():
    records={}
    for folder in ('fits','inherited'):
        for p in (OUT/folder).glob('*.json'):
            f=json.loads(p.read_text());f['composite_origin']=dict(path=str(p.relative_to(ROOT)),sha256=energy.sha(p));records[f['key']]=f
    return records


def worker(job):return fit(*job)['key']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--fit-only',action='store_true');a=ap.parse_args();freeze()
    jobs=[(f,m,{}) for n in (3,4) for f in combinations(FAMILIES,n) for m in STRUCTURES];assert len(jobs)==255
    energy.write(OUT/'training_started.json',dict(created_utc=energy.now(),workers=a.workers,new_fits=225,new_starts=450,inherited=30))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(worker,jobs):pass
    energy.write(OUT/'training_complete.json',dict(created_utc=energy.now(),records=len(all_fits()),goal_complete=False))
    if not a.fit_only:
        for m in STRUCTURES:core.select(m)
        energy.write(OUT/'selection_complete.json',dict(created_utc=energy.now(),methods=len(STRUCTURES),goal_complete=False))


if __name__=='__main__':main()
