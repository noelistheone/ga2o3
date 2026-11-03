"""Train log-odds inventory closure on the unchanged doped-property task."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import os,sys,json,time,argparse,copy
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit,logit
from sandbox.logodds_inventory_model import LogOddsInventoryModel
from sandbox.coherent_fermi import published_coherent_profile,TABLE
from sandbox import transport
import phase17_refinement as prior
import phase2_energy as energy

OUT=energy.BASE/'phase18_logodds_prediction';core=prior.study.core
PRED=prior.study.PRED;TARGETS=prior.study.TARGETS;FAMILIES=core.FAMILIES
STRUCTURES={};REFERENCES={'Logit_product_RT':'activation_sigmoid_RT'}
for basis,training in [('original_RT','RT'),('binding_joint','joint')]:
    shared=dict(fraction='zero' if training=='RT' else 'one',training=training,readout='ionized_extra')
    if training=='joint':shared['binding']='screen'
    STRUCTURES['Logit_inventory_'+basis]=dict(**shared,closure='inventory')
    STRUCTURES['Logit_output_'+basis]=dict(**shared,closure='affine_output')
    STRUCTURES['Logit_add_'+training]=dict(empirical='activation_add',training=training)
    STRUCTURES['Logit_product_'+training]=dict(empirical='activation_product',training=training)
MODELS={};FITS={};BANK={}
core.METHODS.update(STRUCTURES);core.OUT=OUT


def mapper(method):
    if 'empirical' in STRUCTURES[method]:raise ValueError('Direct activation has no defect state mapper')
    if method not in MODELS:MODELS[method]=LogOddsInventoryModel(STRUCTURES[method],profile=published_coherent_profile(),pO2_atm=1e-5)
    return MODELS[method]


core.mapper=mapper


def specification(method):
    d=STRUCTURES[method];lo=[14.];hi=[18.]
    if d.get('binding')=='screen':lo += [np.log10(.003),0.];hi += [np.log10(.2),5.]
    output=d.get('closure')=='affine_output';lo += [-6.,-6.] if output else [-12.,-12.]
    hi += [6.,6.] if output else [12.,12.]
    starts=[]
    for B,E,screen,fraction,slope,c0 in [(16.,.03,.2,.2,-1.,0.),(15.5,.06,1.,.7,-.1,3.)]:
        w=[B]
        if d.get('binding')=='screen':w += [np.log10(E),screen]
        w += [c0,0.] if output else [float(logit(fraction)),slope]
        starts.append(w)
    return np.array(lo),np.array(hi),starts


def training_data(families,method):
    old='Inventory_reference_original_RT' if STRUCTURES[method]['training']=='RT' else 'Inventory_reference_binding_joint'
    return prior.study.training_data(families,old)


def activation(w,table,kind,*,source_shift=0.):
    w=np.asarray(w,float);C=table.incorporated_cm3.to_numpy(float);H=table.H_cm3.to_numpy(float)
    x=np.log10(C)-18.;g=w[1]+w[2]*x+np.log(10.)*source_shift
    logfraction=-np.logaddexp(0.,-g);survival=expit(-g);logB=w[0]*np.log(10.)
    if kind=='activation_product':
        total=C+H+10**w[0];values=np.log10(total)+logfraction/np.log(10.)
        dB=np.full(len(C),10**w[0])/total;activation_weight=np.ones(len(C))
    elif kind=='activation_add':
        logactive=np.log(C+H)+logfraction;logn=np.logaddexp(logB,logactive)
        values=logn/np.log(10.);dB=np.exp(logB-logn);activation_weight=np.exp(logactive-logn)
    else:raise ValueError('Explicit add/product activation required')
    da=activation_weight*survival/np.log(10.)
    return values,np.c_[dB,da,x*da],activation_weight*survival


def forward_targets(method,parameters,table):
    d=STRUCTURES[method]
    if 'empirical' in d:return activation(parameters,table,d['empirical'])[:2]
    model=mapper(method);values=[];J=[]
    for row in table.itertuples():
        p=model.point(row,parameters);values.append(p.values[0]);J.append(p.jacobian[0])
    return np.array(values),np.array(J)


def saved_predictions(method,parameters):
    n=[];mu=[];physical='empirical' not in STRUCTURES[method];cache={}
    if not physical and parameters is not None:empirical=activation(parameters,TARGETS,STRUCTURES[method]['empirical'])[0]
    for j,row in enumerate(TARGETS.itertuples()):
        try:
            if parameters is None:raise RuntimeError('All starts failed')
            if physical:
                key=(row.sample_id,float(row.Tobs_K))
                if key not in cache:cache[key]=mapper(method).point(row,parameters)
                p=cache[key];g=p.diagnostics;logn=float(p.values[0]);charges=p.values[1:];tv=p.transport_values
                dg={k:g[k] for k in ('root_residual_log','jacobian_condition','free_n_cm3','free_p_cm3','Tf_K','measurement_T_K',
                    'Nd_cm3','profile_sha256','localized_fraction','donor_subgroups','same_physical_state_for_output','hall_factor',
                    'native_survival','complex_survival','complex_reallocation_fraction','foreign_atom_relative_error',
                    'foreign_atom_derivative_error','host_Ga_O_inventory_change_cm3','thermal_cap_K','hot_polish_steps',
                    'allocation_logit','allocation_positive_coordinate','affine_output_correction_dex')}
                valid=True if STRUCTURES[method]['closure']=='inventory' else None
            else:
                logn=float(empirical[j]);logmu=float(np.log10(transport.mu_caughey_thomas(10**logn,float(row.Tobs_K))))
                tv=[logmu,float(np.log10(transport.E)+logn+logmu)];charges=None;valid=None
                dg=dict(same_physical_state_for_output=False,property_scope='direct logit activation and CT(predictednH,T),no defect stock',
                    thermal_cap_K=float(row.thermal_cap_K),measurement_T_K=float(row.Tobs_K))
            pred=dict(obs_id=row.obs_id,predicted=logn if row.endpoint=='nH' else float(tv[0]),
                predicted_log_nH=logn,charge_outputs=charges,transport_values=tv,numeric=True,physical_valid=valid,diagnostics=dg)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            pred=dict(obs_id=row.obs_id,predicted=None,numeric=False,physical_valid=False,error=repr(exc))
        (n if row.endpoint=='nH' else mu).append(pred)
    return n,mu


def oldbank():
    global BANK
    if not BANK:
        BANK={(f['method'],tuple(sorted(f['train_families'])),json.dumps(f['config'],sort_keys=True)):f for f in prior.composite().values()}
    return BANK


def optimize(method,families,starts,max_nfev):
    tr=training_data(families,method);sw=np.sqrt(tr.loss_weight.to_numpy(float));y=tr.target.to_numpy(float)
    lo,hi,_=specification(method);cache={};trials=[];failures=[]
    def calc(w):
        key=tuple(w)
        if key not in cache:
            try:value=forward_targets(method,w,tr)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failures.append(dict(parameters=w,error=repr(exc)));raise
            cache.clear();cache[key]=value
        return cache[key]
    for initial in starts:
        try:
            op=least_squares(lambda w:(calc(w)[0]-y)*sw,initial,jac=lambda w:calc(w)[1]*sw[:,None],bounds=(lo,hi),
                max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            trials.append(dict(initial=initial,parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                nfev=int(op.nfev),njev=int(op.njev),optimality=float(op.optimality),active_mask=op.active_mask,message=op.message))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            trials.append(dict(initial=initial,parameters=initial,loss=None,success=False,error=repr(exc)))
    finite=[t for t in trials if t['loss'] is not None]
    return (None if not finite else min(finite,key=lambda t:t['loss'])['parameters']),trials,failures


def fit(families,method,config):
    families=tuple(sorted(families));key=core.prior.old.digest(dict(protocol=energy.sha(OUT/'prediction_protocol.json'),method=method,train_families=families,config=config))
    path=OUT/('inherited' if method in REFERENCES else 'fits')/(key+'.json')
    if key in FITS:return FITS[key]
    if path.exists():FITS[key]=json.loads(path.read_text());return FITS[key]
    tr=training_data(families,method);clock=time.perf_counter();d=STRUCTURES[method]
    model=None if 'empirical' in d else mapper(method);before=None if model is None else model.counters()
    inherited=None;failures=[]
    if method in REFERENCES:
        old=oldbank()[REFERENCES[method],families,'{}'];parameters=old['parameters']
        val,_=forward_targets(method,parameters,tr);loss=float(tr.loss_weight.to_numpy(float)@((val-tr.target.to_numpy(float))**2))
        inherited=dict(method=old['method'],key=old['key'],origin=old['composite_origin'],original_trials=old['trials'],
            note='identical product function/bounds/starts/traininglabels;parameters reused,no new optimizer')
        trials=[dict(initial=parameters,parameters=parameters,loss=loss,success=bool(min(old['trials'],key=lambda t:t['loss'])['success']),
            nfev=0,njev=0,inherited=True)]
    else:parameters,trials,failures=optimize(method,families,specification(method)[2],120)
    counts=None if before is None else {k:model.counters()[k]-before[k] for k in before}
    n,mu=saved_predictions(method,parameters)
    f=dict(key=key,method=method,definition=d,config=config,train_families=families,parameters=parameters,
        parameter_names=model.parameter_names if model is not None else ['log10_background','activation_logit_a','activation_logit_b'],
        trials=trials,state_failures=failures,inherited=inherited,training_obs_ids=tr.obs_id.tolist(),
        training_specimens=sorted(tr.sample_id.unique()),weight_per_observation=tr.loss_weight.to_numpy(float),
        training_blocks=[dict(block=b,observations=len(g),sum_weight=float(g.loss_weight.sum())) for b,g in tr.groupby('loss_block')],
        predictions=n,property_predictions=mu,seconds=time.perf_counter()-clock,training_counters=counts,
        counters=None if before is None else {k:model.counters()[k]-before[k] for k in before},
        metadata=dict(new_mu_supervision=False,property_scope='actualdose-to-property prediction,conditional endpoint inventory'))
    energy.write(path,f);FITS[key]=energy.clean(f)
    print('logitfit',method,len(families),'seconds',round(f['seconds'],2),'newstarts',0 if inherited else 2,
        'stop',sum(bool(t['success']) for t in trials),flush=True)
    return FITS[key]


core.fit=fit


def freeze():
    gate=json.loads((OUT/'numerical_gates.json').read_text());assert gate['passed']
    for p,h in gate['code_hashes'].items():assert energy.sha(ROOT/p)==h,p
    sources=(Path(__file__),ROOT/'src/sandbox/logodds_inventory_model.py',ROOT/'scripts/trainable_sandbox_20261017/phase18_gates.py',
        ROOT/'docs/ML_SANDBOX_LOGODDS_PREDICTION_20261018.md',ROOT/'src/sandbox/inventory_coherent_model.py',ROOT/'src/sandbox/fixed_inventory_charge.py')
    inputs=(OUT/'numerical_gates.json',energy.BASE/'artifact_manifest_phase17.json',energy.BASE/'dataset_v1.csv',energy.BASE/'episodes_v1.csv',
        prior.study.prior.study.OUT/'thermal_audit/specimens.csv',TABLE)
    payload=dict(created_utc=energy.now(),version=1,structures=STRUCTURES,references=REFERENCES,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in sources},input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        primary='unchanged29Si/Sn5DOI/0-1-2nHspecimenanchors/29-23-18queries/common18',new_fits=105,new_starts=210,references_reused=15,
        training='RTnH orRT+.1nHcurve<=275K;DOI/specimen/read equal;no mu labels',
        specifications={m:dict(bounds=specification(m)[:2],starts=specification(m)[2]) for m in STRUCTURES},
        optimizer='2starts/120nfev/1e-7stopping;allfailed-start records uniformly2original+savedbest360nfev;preserveV1',
        source_primary='same7lambda outputnHshift selectedinnerDOI;internal-logit adaptation requiresseparateprotocol',
        empirical='bothaddbackgroundoutside andproductbackgroundinside sigmoid;productRTidenticaloldreference',
        transport='sameFD/Nq2/free-onlyscreening/Hall/205 empiricalextra;muRT5/2,muT54/3/1;sigma derived',
        expected_inner_scores=2400,retrospective=True,universal_gain_established=False,new_DFT_MD=0,new_measurements=0,goal_complete=False)
    p=OUT/'prediction_protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        for key in ('structures','references','code_hashes','input_hashes','specifications'):assert old[key]==energy.clean(payload[key]),key
    else:energy.write(p,payload)


def all_fits():
    result={}
    for folder in ('fits','inherited'):
        for path in (OUT/folder).glob('*.json'):
            f=json.loads(path.read_text());f['composite_origin']=dict(path=str(path.relative_to(ROOT)),sha256=energy.sha(path));result[f['key']]=f
    return result


def worker(job):return fit(*job)['key']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--fit-only',action='store_true');a=ap.parse_args();freeze()
    jobs=[(f,m,{}) for n in (3,4) for f in combinations(FAMILIES,n) for m in STRUCTURES];assert len(jobs)==120
    energy.write(OUT/'training_started.json',dict(created_utc=energy.now(),workers=a.workers,new_fits=105,new_starts=210,references=15))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(worker,jobs):pass
    energy.write(OUT/'training_complete.json',dict(created_utc=energy.now(),records=len(all_fits()),goal_complete=False))
    if not a.fit_only:
        for method in STRUCTURES:core.select(method)


if __name__=='__main__':main()
