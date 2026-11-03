"""Train one conservative relative-inventory relation and matched controls.

Old29/5 source task stays fixed. Hyperparameters are chosen within training
families; the later source is development evidence, never an unseen blind set.
"""
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import os,sys,json,time,hashlib,argparse
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import logit
from sklearn.linear_model import Ridge
from csv_io_v2 import read_result_csv as read
from sandbox.relative_inventory_model import RelativeInventoryModel
from sandbox.coherent_fermi import published_coherent_profile
import phase16_prediction as inputs
import phase9_joint_prediction as oldfeatures
import phase2_energy as energy
from transport_chain import median

BASE=energy.BASE;OUT=BASE/'phase20_source_inventory';RT=inputs.PRED[inputs.PRED.kind.eq('RT')].copy()
FAMILIES=tuple(sorted(RT.family.unique()));ROLES=read(BASE/'episodes_v1.csv')
REG=(0.,.001,.01,.1,1.,None);RIDGE=(.01,.1,1.,10.,100.,1e4)
PHYSICS=('inventory','output');MODELS={};REFS={}


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()[:20]


def reference(families):
    global REFS
    if not REFS:
        for path in (BASE/'phase15_coherent_hall/fits').glob('*.json'):
            d=json.loads(path.read_text())
            if d['method']=='HallExtra_original_RT':REFS[tuple(sorted(d['train_families']))]=(d,path)
        path=BASE/'phase19_delivery/fits/HallExtra_original_RT.json';d=json.loads(path.read_text())
        REFS[tuple(FAMILIES)]=(d,path)
    return REFS[tuple(sorted(families))]


def mapper(method):
    if method not in MODELS:MODELS[method]=RelativeInventoryModel(dict(field=method),profile=published_coherent_profile(),pO2_atm=1e-5)
    return MODELS[method]


def weights(d):return 1/(d.family.map(d.family.value_counts()).to_numpy(float)*d.family.nunique())


def forward(method,w,table,*,penalty=False):
    model=mapper(method);values=[];jac=[];delta=[];deltaJ=[];valid=[]
    for row in table.itertuples():
        p=model.point(row,w,require_observation=False);values.append(p.values[0]);jac.append(p.jacobian[0]);valid.append(p.diagnostics['observation_domain_valid'])
        if penalty:
            if method=='output':delta.append(w[2]);deltaJ.append([0.,0.,1.])
            else:
                zero=np.array(w).copy();zero[2]=0.;q=model.point(row,zero,require_observation=False)
                dj=p.jacobian[0].copy();dj[:2]-=q.jacobian[0,:2]
                delta.append(p.values[0]-q.values[0]);deltaJ.append(dj)
    return np.array(values),np.array(jac),np.array(delta),np.array(deltaJ),np.array(valid)


def fit(families,method,reg):
    families=tuple(sorted(families));key=digest(dict(protocol=energy.sha(OUT/'prediction_protocol.json'),families=families,method=method,reg=reg))
    path=OUT/'fits'/f'{key}.json'
    if path.exists():return json.loads(path.read_text())
    tr=RT[RT.family.isin(families)];wgt=weights(tr);y=tr.target.to_numpy(float);clock=time.perf_counter()
    if method in PHYSICS and reg is None:
        base,origin=reference(families);pars=list(base['parameters'])+[0.]
        f=dict(key=key,method=method,reg=reg,parameters=pars,train_families=families,
            trials=[],state_failures=[],training_valid=True,reference_path=str(origin.relative_to(ROOT)),reference_sha256=energy.sha(origin),
            exact_reference_fallback=True,new_optimizer_starts=0)
    elif method in PHYSICS:
        base,_=reference(families);lo=[-6.,14.,-6.];hi=[6.,18.,6.]
        other=[logit((np.log10(1750.)-np.log10(600.))/(np.log10(1950.)-np.log10(600.))),15.5,.1]
        starts=[list(base['parameters'])+[0.],other];sw=np.sqrt(wgt);cache={};trials=[];failures=[];continuations=0
        def calc(w):
            nonlocal continuations
            k=tuple(w)
            if k not in cache:
                try:value=forward(method,w,tr,penalty=reg>0.)
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    failures.append(dict(parameters=np.array(w),error=repr(exc)));raise
                continuations+=int((~value[4]).sum());cache.clear();cache[k]=value
            return cache[k]
        def residual(w):
            v,J,d,dJ,valid=calc(w);a=(v-y)*sw
            return np.r_[a,np.sqrt(reg)*sw*d] if reg>0 else a
        def jacobian(w):
            v,J,d,dJ,valid=calc(w);a=J*sw[:,None]
            return np.r_[a,np.sqrt(reg)*sw[:,None]*dJ] if reg>0 else a
        def run(w,n,stage):
            try:
                op=least_squares(residual,w,jac=jacobian,bounds=(lo,hi),x_scale='jac',max_nfev=n,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                return dict(stage=stage,initial=w,parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                    nfev=int(op.nfev),optimality=float(op.optimality),message=op.message)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:return dict(stage=stage,initial=w,loss=None,success=False,error=repr(exc))
        trials=[run(s,120,'initial') for s in starts]
        if any(not t['success'] for t in trials):
            good=[t for t in trials if t['loss'] is not None];extra=[min(good,key=lambda t:t['loss'])['parameters']] if good else []
            trials += [run(s,360,'uniform_refinement') for s in starts+extra]
        good=[t for t in trials if t['loss'] is not None];best=min(good,key=lambda t:t['loss']) if good else None
        pars=None if best is None else best['parameters'];valid=False
        if pars is not None:valid=bool(forward(method,pars,tr)[4].all())
        f=dict(key=key,method=method,reg=reg,train_families=families,parameters=pars,trials=trials,state_failures=failures,
            training_valid=valid,selected_success=best['success'] if best else False,numerical_continuation_trial_states=continuations,
            exact_reference_fallback=False,new_optimizer_starts=len(trials),training_obs_ids=tr.obs_id.tolist())
    elif method=='ridge':
        X=oldfeatures.features(tr);center=np.average(X,axis=0,weights=wgt);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=wgt));scale[scale<1e-12]=1.
        m=Ridge(alpha=reg).fit((X-center)/scale,y,sample_weight=wgt*len(tr))
        f=dict(key=key,method=method,reg=reg,train_families=families,training_valid=True,
            coefficients=m.coef_,intercept=float(m.intercept_),center=center,scale=scale,trials=[],new_optimizer_starts=0)
    else:
        delta=0.;coef=None
        if method=='median':delta=float(median(y-np.log10(tr.incorporated_cm3.to_numpy(float)),wgt))
        if method=='line':
            X=np.c_[np.ones(len(tr)),np.log10(tr.incorporated_cm3.to_numpy(float))-18.]
            coef=np.linalg.lstsq(X*np.sqrt(wgt[:,None]),y*np.sqrt(wgt),rcond=None)[0]
        f=dict(key=key,method=method,reg=reg,train_families=families,training_valid=True,offset_dex=delta,
            coefficients=coef,trials=[],new_optimizer_starts=0)
    f['seconds']=time.perf_counter()-clock;f=energy.clean(f);energy.write(path,f)
    print('relativefit',method,reg,len(families),'sec',round(f['seconds'],2),'valid',f['training_valid'],flush=True)
    return f


def configs(method):return list(REG) if method in PHYSICS else list(RIDGE) if method=='ridge' else [None]


def predict(f,table,*,source_kind='offset',shift=0.,require_observation=True):
    method=f['method']
    if method in PHYSICS:
        if f.get('parameters') is None or not f['training_valid']:raise ValueError('No valid trained model')
        kind='field' if source_kind=='field' else 'offset'
        p=[mapper(method).point(r,f['parameters'],source_kind=kind,source_shift=shift,require_observation=require_observation) for r in table.itertuples()]
        return np.array([v.values[0] for v in p]),np.array([v.source_jacobian[0] for v in p]),np.array([v.diagnostics['observation_domain_valid'] for v in p])
    if method=='ridge':
        x=(oldfeatures.features(table)-np.asarray(f['center']))/np.asarray(f['scale']);y=float(f['intercept'])+x@np.asarray(f['coefficients'])
    else:
        x=np.log10(table.incorporated_cm3.to_numpy(float)+(table.H_cm3.to_numpy(float) if method=='unitH' else 0.))
        y=f['coefficients'][0]+f['coefficients'][1]*(x-18.) if method=='line' else x+f['offset_dex']
    y=y+shift;valid=y>=13.
    if require_observation and not valid.all():raise ValueError('Empirical readout below detection floor')
    return y,np.ones(len(y)),valid


def zero_score(f,held):
    d=RT[RT.family.eq(held)]
    try:p,_,v=predict(f,d)
    except (ValueError,RuntimeError,np.linalg.LinAlgError):return np.inf
    return float(np.abs(p-d.target.to_numpy(float)).mean()) if v.all() else np.inf


def select_global(train,method):
    rows=[]
    for reg in configs(method):
        scores=[zero_score(fit([f for f in train if f!=held],method,reg),held) for held in train]
        rows.append(dict(reg=reg,score=float(np.mean(scores)),inner_families=list(train),inner_scores=scores))
    def key(row):
        score=np.inf if row['score'] is None else row['score'];r=np.inf if row['reg'] is None else row['reg']
        return round(score,10),-r
    best=min(rows,key=key);return best['reg'],rows


def freeze():
    gates=json.loads((OUT/'numerical_gates.json').read_text());assert gates['passed']
    for p,h in gates['code_hashes'].items():assert energy.sha(ROOT/p)==h,p
    learning=json.loads((OUT/'learning_gates.json').read_text());assert learning['passed']
    for p,h in learning['code_hashes'].items():assert energy.sha(ROOT/p)==h,p
    refs=[dict(families=list(f),path=str(p.relative_to(ROOT)),sha256=energy.sha(p)) for n in (3,4) for f in combinations(FAMILIES,n) for _,p in [reference(f)]]
    paths=(Path(__file__),ROOT/'src/sandbox/relative_inventory.py',ROOT/'src/sandbox/relative_inventory_model.py',
        ROOT/'src/sandbox/observation_adaptation.py')
    payload=dict(created_utc=energy.now(),version=1,code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (BASE/'dataset_v1.csv',BASE/'episodes_v1.csv',BASE/'artifact_manifest_phase19.json',OUT/'numerical_gates.json',OUT/'learning_gates.json',OUT/'probe_audit.json')},
        reference_fits=refs,methods=['inventory','output','ridge','median','line','unit','unitH'],
        regularization=REG,ridge_grid=RIDGE,physical_parameters=['logitTf','log10background','sharedlog10odds'],
        objective='sourceequal squared nH + gamma sourceequal squared departure from sameTf/B h0 readout;outputcontrol has same response penalty',
        shared_selection='withinoutertraining families,LOFOzeroanchor select gamma;freeze it across0/1/2then selectsourceprior on same training innerfolds',
        source_prior='quadratic or log1p(response_effect_dex^2),both same nearzero curvature;7lambda0/.001/.01/.1/1/10/inf;no calibratedposteriorclaim',
        source_coordinate='internal raw log10odds normalized via firstanchor nH response;outputeffect is identical observable dex coordinate',
        budgets='same original29/5,0-1-2roles inclguard,common18;2warmstarts120 cap,allfailedstartfits360bothoriginal+best;sourceanchorvectors<=4096,33nodes80localiterations',
        actual_compute='recordoptimizerstarts,root/vectorcalls andCPUtime;directclosedformcontrols mayuse less,not exact walltime matching',
        field='signed dopantatomshare KL tilt;nativefixed;foreign/Oconserved,Gaopen;not a new microscopicthermodynamic law',
        validation_grade='retrospective development;alloldlabels seen;Leedyused to motivatedesign,not blind;no newsource label in fitting/selection',
        energy_budget='full873 suppliedenergies,nonewlabelrevelation',new_DFT_MD=0,new_material_measurements=0,GPU_used=False,goal_complete=False)
    p=OUT/'prediction_protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        for key in ('code_hashes','input_hashes','reference_fits'):assert old[key]==energy.clean(payload[key]),key
    else:energy.write(p,payload)


def worker(job):return fit(*job)['key']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();freeze()
    if a.train:
        jobs=[(list(f),m,r) for n in (3,4) for f in combinations(FAMILIES,n) for m in ('inventory','output','ridge','median','line','unit','unitH') for r in configs(m)]
        energy.write(OUT/'training_started.json',dict(created_utc=energy.now(),jobs=len(jobs),new_nonlinear_fits=150,expected_initial_nonlinear_starts=300,workers=a.workers))
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for _ in pool.map(worker,jobs):pass
        energy.write(OUT/'training_complete.json',dict(created_utc=energy.now(),jobs=len(jobs),goal_complete=False))


if __name__=='__main__':main()
