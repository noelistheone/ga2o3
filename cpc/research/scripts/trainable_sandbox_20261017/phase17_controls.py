"""Match the activation-function class and newly supplied cap inputs directly."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
import phase17_prediction as study
import phase2_energy as energy
from sandbox import transport

OUT=study.OUT/'matched_controls';CACHE={}
METHODS={}
for training in ('RT','joint'):
    for relation in ('constant','dose_plus','dose_minus'):
        METHODS['Matched_activation_'+relation+'_'+training]=dict(empirical='matched_activation',training=training,relation=relation)
    METHODS['Direct_ridge_cap_'+training]=dict(empirical='ridge',training=training)
study.core.METHODS.update(METHODS)


def training_data(families,method):
    reference='Inventory_reference_original_RT' if METHODS[method]['training']=='RT' else 'Inventory_reference_binding_joint'
    return study.training_data(families,reference)


def specification(method):
    relation=METHODS[method]['relation'];lo=[14.,0.];hi=[18.,12.]
    starts=[[16.,.03],[15.5,.3]]
    if relation!='constant':
        lo.append(0.);hi.append(12.);starts[0].append(.1);starts[1].append(1.)
    return np.array(lo),np.array(hi),starts


def activation(w,table,relation):
    C=table.incorporated_cm3.to_numpy(float);H=table.H_cm3.to_numpy(float);D=C+H
    phi=np.zeros(len(table))
    if relation!='constant':
        sign=1. if relation=='dose_plus' else -1.;phi=np.logaddexp(0.,sign*(np.log10(C)-18.))
    s=w[1]+(w[2]*phi if len(w)>2 else 0.);survival=np.exp(-s);fraction=-np.expm1(-s)
    n=10**w[0]+D*fraction;values=np.log10(n)
    J=np.c_[np.full(len(table),10**w[0])/n,D*survival/(n*np.log(10.))]
    if len(w)>2:J=np.c_[J,J[:,1]*phi]
    return values,J


def features(d):
    c=np.log10(d.incorporated_cm3.to_numpy(float))-18.;T=d.Tobs_K.to_numpy(float)
    hk=d.H_known.to_numpy(float);ck=d.carbon_known.to_numpy(float)
    h=np.log10(np.maximum(d.H_cm3.to_numpy(float),1))-17.;carbon=np.log10(np.maximum(d.carbon_cm3.to_numpy(float),1))-17.
    return np.c_[c,c*c,(d.growth_T_C.to_numpy(float)+273.15-1100)/300,
        (d.thermal_cap_K.to_numpy(float)-1100)/300,d.element.eq('Sn').to_numpy(float),np.log(T/300),300/T-1,
        c*(300/T-1),hk,h*hk,ck,carbon*ck]


def predictions(values):
    n=[];mu=[]
    for r,v in zip(study.TARGETS.itertuples(),values):
        logmu=float(np.log10(transport.mu_caughey_thomas(10**v,float(r.Tobs_K))))
        p=dict(obs_id=r.obs_id,predicted=float(v if r.endpoint=='nH' else logmu),predicted_log_nH=float(v),
            charge_outputs=None,transport_values=[logmu,float(np.log10(transport.E)+v+logmu)],numeric=True,physical_valid=None,
            diagnostics=dict(same_physical_state_for_output=False,property_scope='direct activation/Ridge,legacy CT(nHargument,T),no defect inventory',
                thermal_cap_K=float(r.thermal_cap_K),measurement_T_K=float(r.Tobs_K)))
        (n if r.endpoint=='nH' else mu).append(p)
    return n,mu


def fit(families,method,config):
    if method not in METHODS:return study.fit(families,method,config)
    families=tuple(sorted(families));key=study.core.prior.old.digest(dict(protocol=energy.sha(OUT/'protocol.json'),method=method,train_families=families,config=config))
    path=OUT/'fits'/(key+'.json')
    if key in CACHE:return CACHE[key]
    if path.exists():CACHE[key]=json.loads(path.read_text());return CACHE[key]
    tr=training_data(families,method);weights=tr.loss_weight.to_numpy(float);y=tr.target.to_numpy(float);sw=np.sqrt(weights)
    clock=time.perf_counter();trials=[];meta={};parameters=None
    if METHODS[method]['empirical']=='ridge':
        X=features(tr);center=np.average(X,axis=0,weights=weights);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=weights));scale[scale<1e-12]=1.
        m=Ridge(**config).fit((X-center)/scale,y,sample_weight=weights*len(tr)/weights.sum())
        values=m.predict((features(study.TARGETS)-center)/scale)
        fitted=m.predict((X-center)/scale);meta=dict(center=center,scale=scale,coefficients=m.coef_,intercept=m.intercept_)
        trials=[dict(loss=float(weights@((fitted-y)**2)),success=True,linear_unique_solver=True,nfev=1)]
    else:
        relation=METHODS[method]['relation'];lo,hi,starts=specification(method)
        for initial in starts:
            op=least_squares(lambda w:(activation(w,tr,relation)[0]-y)*sw,initial,
                jac=lambda w:activation(w,tr,relation)[1]*sw[:,None],bounds=(lo,hi),max_nfev=120,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            trials.append(dict(initial=initial,parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                nfev=int(op.nfev),njev=int(op.njev),optimality=float(op.optimality),active_mask=op.active_mask,message=op.message))
        parameters=min(trials,key=lambda t:t['loss'])['parameters'];values,_=activation(parameters,study.TARGETS,relation)
    pred,prop=predictions(values)
    f=dict(key=key,method=method,definition=METHODS[method],config=config,train_families=families,parameters=parameters,
        parameter_names=[] if parameters is None else ['log10_background','s0']+(['s1'] if len(parameters)==3 else []),
        metadata=meta,trials=trials,state_failures=[],inherited=None,training_obs_ids=tr.obs_id.tolist(),
        training_specimens=sorted(tr.sample_id.unique()),weight_per_observation=weights,predictions=pred,property_predictions=prop,
        seconds=time.perf_counter()-clock,new_mu_supervision=False)
    energy.write(path,f);CACHE[key]=energy.clean(f)
    print('matchedcontrol',method,len(families),config,'stop',sum(t['success'] for t in trials),flush=True)
    return CACHE[key]


def freeze():
    study.freeze();checks=[]
    for method,d in METHODS.items():
        if d['empirical']=='ridge':continue
        tr=training_data(study.FAMILIES,method);w=np.array(specification(method)[2][1]);a,J=activation(w,tr,d['relation'])
        for j in range(len(w)):
            h=1e-4;p=w.copy();n=w.copy();p[j]+=h;n[j]-=h
            fd=(activation(p,tr,d['relation'])[0]-activation(n,tr,d['relation'])[0])/(2*h)
            error=float(max(abs(fd-J[:,j])/(1+abs(fd))));assert error<1e-6
            checks.append(dict(method=method,j=j,error=error))
    protocol=dict(created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),methods=METHODS,
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (study.OUT/'prediction_protocol.json',energy.BASE/'dataset_v1.csv',energy.BASE/'episodes_v1.csv')},
        rationale='samealpha(C) family deconfounds allocationphysics from activationcurve shape;Ridge receives same explicit cap inputs',
        information='same nRT or nRT+.1nT labels;wholeDOIexclusion/inner7lambda,allpreprocessingtrainingonly;no mu labels',
        matched_activation='nH=background+(C+H)*(1-exp[-s(C)]),samepositiveconstant/softplusdose functions ascomplexallocator;no defectsolver',
        count='90nonlinearactivationfits/180starts,180convexRidgefits/6alphas percontext;runtime andcandidatecounts disclosedseparately',
        timing='protocol added during basefit execution,before any newouter scoring;allwork retrospective,not blind',
        derivative_checks=checks,maximum_derivative_error=max(c['error'] for c in checks),goal_complete=False)
    pp=OUT/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text())
        for k in ('code_sha256','methods','input_hashes','derivative_checks'):assert old[k]==energy.clean(protocol[k]),k
    else:energy.write(pp,protocol)


def all_fits():
    rows={}
    for p in (OUT/'fits').glob('*.json'):
        f=json.loads(p.read_text());f['composite_origin']=dict(path=str(p.relative_to(ROOT)),sha256=energy.sha(p));rows[f['key']]=f
    return rows


def worker(job):return fit(*job)['key']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();freeze()
    if a.freeze_only:return
    jobs=[(f,m,c) for n in (3,4) for f in combinations(study.FAMILIES,n) for m in METHODS for c in study.core.grids(m)]
    assert len(jobs)==270
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(worker,jobs):pass
    energy.write(OUT/'training_complete.json',dict(created_utc=energy.now(),fit_records=len(all_fits()),goal_complete=False))


if __name__=='__main__':main()
