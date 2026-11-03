"""Bounded submission controls. Inputs/splits are fixed before fitting.

Runs inside the existing submission tree; never writes to historical research.
"""
from pathlib import Path
from itertools import combinations
import argparse, hashlib, json, os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import least_squares
from scipy.interpolate import CubicHermiteSpline

BASE=Path(__file__).resolve().parents[1]
RESEARCH=BASE/'research' if (BASE/'research').is_dir() else BASE.parents[3]
sys.path.insert(0,str(RESEARCH/'src'))
DATA=BASE/'source_data';AUDIT=BASE/'verification'
LAMBDAS=(0.,.001,.01,.1,1.,10.)
LN10=np.log(10.)

def save(name,obj):
    (AUDIT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')

def weights(d):
    return d.scenario_weight.to_numpy(float)/(d.related_cluster.nunique()*
        d.groupby('related_cluster').family.transform('nunique').to_numpy(float)*
        d.groupby('family').sample_id.transform('nunique').to_numpy(float))

def score(d):
    return float(d.assign(ae=abs(d.predicted-d.target)).groupby(['family','sample_id']).ae.mean().groupby('family').mean().mean())

def features(d):
    c=(np.log10(d.incorporated_cm3.to_numpy(float))-18)/2
    g=(d.growth_T_C.to_numpy(float)-700)/300
    sn=d.element.eq('Sn').to_numpy(float)
    return np.c_[np.ones(len(d)),c,c*c,g,sn]

def room():
    from sandbox.coherent_transport import fermi_transport
    from sandbox.coherent_fermi import published_coherent_profile
    d=pd.read_csv(DATA/'validation_room_inputs.csv')
    assert d.Tobs_K.eq(300).all()
    units=sorted(d.related_cluster.unique());profile=published_coherent_profile()
    grid=np.linspace(13.,22.,1801);yv=[];dy=[]
    def hall(x):
        t=fermi_transport(float(x),float(x),300.,Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,
            mass_ratio=profile.electron_mass_ratio,extra_mobility_cm2Vs=205.)
        return t.values[0],t.jacobian[0,0]+t.jacobian[0,1]
    for x in grid:
        a,b=hall(x);yv.append(a);dy.append(b)
    readout=CubicHermiteSpline(grid,yv,dy,extrapolate=False)
    errors=[abs(float(readout(x))-hall(x)[0]) for x in np.linspace(13.0123,21.9876,37)]
    assert max(errors)<2e-7
    def prediction(table,w):
        X=features(table);f=expit(X@w[1:]);C=table.incorporated_cm3.to_numpy(float)
        H=table.H_cm3.to_numpy(float);Nb=10**w[0];n=Nb+H+C*f
        logn=np.log10(n);J=np.c_[Nb/n,((C*f*(1-f)/(n*LN10))[:,None])*X]
        return readout(logn),readout(logn,1)[:,None]*J
    # Analytic derivative check, at an interior positive state.
    probe=np.array([15.,-.7,-.4,.1,.2,-.1]);v,J=prediction(d,probe);je=[]
    for j in range(6):
        step=np.eye(6)[j]*1e-5
        fd=(prediction(d,probe+step)[0]-prediction(d,probe-step)[0])/2e-5
        je.append(float(abs(fd-J[:,j]).max()))
    assert max(je)<1e-6
    fits={};fit_records=[];start=time.perf_counter()
    def fit(train,kind,lam):
        key=(tuple(sorted(train)),kind,lam)
        if key in fits:return fits[key]
        tr=d[d.related_cluster.isin(train)];wgt=weights(tr);sw=np.sqrt(wgt);y=tr.target.to_numpy(float)
        if kind=='DirectQuadratic6':
            # Six coefficients: intercept, dose, dose^2, growth, element, known H/C.
            X=np.c_[features(tr),np.log1p(tr.H_cm3.to_numpy(float)/tr.incorporated_cm3.to_numpy(float))]
            R=np.diag([0.,1.,1.,1.,1.,1.]);coef=np.linalg.lstsq(np.r_[X*sw[:,None],np.sqrt(lam)*R],np.r_[y*sw,np.zeros(6)],rcond=None)[0]
            rec=dict(train=list(train),method=kind,alpha=lam,parameters=coef.tolist(),success=True,starts=0)
        else:
            cache={}
            def calc(w):
                k=tuple(w)
                if k not in cache:
                    p,j=prediction(tr,w);pen=np.r_[0.,0.,w[2:]]
                    P=np.diag([0.,0.,1.,1.,1.,1.])
                    cache.clear();cache[k]=(np.r_[(p-y)*sw,np.sqrt(lam)*pen],np.r_[j*sw[:,None],np.sqrt(lam)*P])
                return cache[k]
            bounds=([14.,-12.,-12.,-12.,-12.,-12.],[18.,12.,12.,12.,12.,12.])
            starts=[np.array([14.5,0.,0.,0.,0.,0.]),np.array([16.,-2.,-1.,0.,0.,0.])]
            trials=[]
            for w in starts:
                op=least_squares(lambda z:calc(z)[0],w,jac=lambda z:calc(z)[1],bounds=bounds,max_nfev=120,
                    ftol=1e-8,xtol=1e-8,gtol=1e-8,x_scale='jac')
                trials.append(dict(parameters=op.x.tolist(),loss=float(op.fun@op.fun),success=bool(op.success),nfev=int(op.nfev)))
            if not all(t['success'] for t in trials):
                for t in list(trials):
                    op=least_squares(lambda z:calc(z)[0],t['parameters'],jac=lambda z:calc(z)[1],bounds=bounds,max_nfev=240,
                        ftol=1e-8,xtol=1e-8,gtol=1e-8,x_scale='jac')
                    trials.append(dict(parameters=op.x.tolist(),loss=float(op.fun@op.fun),success=bool(op.success),nfev=int(op.nfev)))
            best=min(trials,key=lambda t:t['loss'])
            rec=dict(train=list(train),method=kind,alpha=lam,parameters=best['parameters'],success=best['success'],starts=len(trials),trials=trials)
        fits[key]=rec;fit_records.append(rec)
        return rec
    def predict(table,rec):
        w=np.array(rec['parameters'])
        if rec['method']=='ReducedInventory6':return prediction(table,w)[0]
        return np.c_[features(table),np.log1p(table.H_cm3.to_numpy(float)/table.incorporated_cm3.to_numpy(float))]@w
    output=[];selections=[]
    for held in units:
        train=[u for u in units if u!=held]
        for kind in ('ReducedInventory6','DirectQuadratic6'):
            scores=[]
            for lam in LAMBDAS:
                val=[]
                for inner in train:
                    r=fit([u for u in train if u!=inner],kind,lam)
                    if not r['success']:raise RuntimeError('Unconverged selected training candidate')
                    q=d[d.related_cluster.eq(inner)].copy();q['predicted']=predict(q,r);val.append(score(q))
                scores.append((float(np.mean(val)),lam))
            error,lam=min(scores,key=lambda x:(round(x[0],12),-x[1]))
            r=fit(train,kind,lam);assert r['success']
            q=d[d.related_cluster.eq(held)].copy();q['predicted']=predict(q,r);q['method']=kind;q['selected_alpha']=lam
            output.append(q);selections.append(dict(held=held,method=kind,selected_alpha=lam,inner_MAE=error,grid=scores))
            print('Capacity control',held,kind,'inner alpha',lam,flush=True)
    out=pd.concat(output,ignore_index=True)
    out.to_csv(DATA/'capacity_control_predictions.csv',index=False)
    summaries=[]
    for kind,q in out.groupby('method'):
        for scope in ('all35','common20'):
            z=q if scope=='all35' else q[q.primary_common20.eq(True)]
            summaries.append(dict(method=kind,scope=scope,MAE=score(z),conditions=z.sample_id.nunique(),sources=z.family.nunique(),parameters=6))
    pd.DataFrame(summaries).to_csv(DATA/'capacity_control_scores.csv',index=False)
    save('capacity_control_fits.json',dict(fits=fit_records,selections=selections,seconds=time.perf_counter()-start,
        hall_interpolation_max_error_dex=max(errors),analytic_gradient_max_error=max(je),all_fits_converged=all(f['success'] for f in fit_records)))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

def thermal():
    from sandbox.kroger_db import load
    from sandbox.material_profile import PUBLISHED_PROFILE
    from sandbox.temperature_defects import TemperatureDefectLayer
    d=pd.read_csv(DATA/'validation_thermal_inputs.csv');layers={};L0,L1=np.log10(600),np.log10(1950)
    def point(r,w):
        key=(r.element,bool(r.H_known))
        if key not in layers:
            els=(r.element,'H') if r.H_known else (r.element,)
            layers[key]=TemperatureDefectLayer(load(),els,profile=PUBLISHED_PROFILE,donor_extension=True,donor_element=r.element,hall='rH1')
        l=layers[key];s=expit(w[0]);T=10**(L0+(L1-L0)*s)
        doses=(r.incorporated_cm3,r.H_cm3) if r.H_known else (r.incorporated_cm3,)
        pars=l.parameters(T=T,Nd=10**w[1],doses=doses,Tobs=r.Tobs_K,binding_eV=10**w[2],screen_s=w[3])
        p=l.predict(pars);chain=np.zeros((l.np,4));chain[0,0]=(L1-L0)*s*(1-s);chain[1,1]=1.;chain[l.ie,2]=1.;chain[l.ie+1,3]=1.
        return float(p.values[0]),p.jacobian[0]@chain
    sources=sorted(d[d.kind.eq('curve')].family.unique());out=[];fits=[]
    for held in sources:
        tr=d[(~d.family.eq(held))&(d.kind.eq('RT')|d.Tobs_K.le(275))].copy()
        wgt=1/(tr.groupby('sample_id').obs_id.transform('count')*tr.groupby('family').sample_id.transform('nunique'))
        wgt=wgt.to_numpy(float);wgt/=wgt.mean();sw=np.sqrt(wgt);y=tr.target.to_numpy(float);cache={}
        def calc(w):
            k=tuple(w)
            if k not in cache:
                values=[point(r,w) for r in tr.itertuples()];cache.clear()
                cache[k]=(np.array([v[0] for v in values]),np.array([v[1] for v in values]))
            return cache[k]
        trials=[]
        for T,B,E,s in ((1400.,16.,.03,.2),(1750.,15.5,.06,1.)):
            t=(np.log10(T)-L0)/(L1-L0);initial=[np.log(t/(1-t)),B,np.log10(E),s]
            op=least_squares(lambda w:(calc(w)[0]-y)*sw,initial,jac=lambda w:calc(w)[1]*sw[:,None],
                bounds=([-6.,12.,np.log10(.003),0.],[6.,18.,np.log10(.3),5.]),max_nfev=160,
                ftol=1e-7,xtol=1e-7,gtol=1e-7)
            trials.append(dict(parameters=op.x.tolist(),loss=float(op.fun@op.fun),success=bool(op.success),nfev=int(op.nfev),active=op.active_mask.tolist()))
        best=min(trials,key=lambda t:t['loss']);assert best['success'],trials
        fits.append(dict(held=held,train_observations=tr.obs_id.tolist(),trials=trials,selected=best))
        for r in d[d.family.eq(held)&d.kind.eq('curve')].itertuples():
            value,j=point(r,best['parameters']);out.append(dict(obs_id=r.obs_id,sample_id=r.sample_id,family=r.family,Tobs_K=r.Tobs_K,
                observed=r.target,predicted=value,common40_125=40<=r.Tobs_K<=125,method='ExpandedBounds'))
        save('thermal_bound_refits.json',dict(fits=fits))
        print('Thermal widened bounds',held,best,flush=True)
    pd.DataFrame(out).to_csv(DATA/'thermal_bound_predictions.csv',index=False)

def protocol():
    paths=[Path(__file__),DATA/'validation_room_inputs.csv',DATA/'validation_thermal_inputs.csv']
    hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    p=AUDIT/'submission_validation_protocol.json'
    obj=dict(version='submission-revision-20261022',created_utc=pd.Timestamp.now(tz='UTC').isoformat(),hashes=hashes,
        scope='Retrospective fixed-protocol targeted controls; no fresh experimental source',
        room=dict(models=['ReducedInventory6','DirectQuadratic6'],parameters_each=6,alphas=LAMBDAS,
            split='five related groups; inner leave-one-training-group-out selects alpha; original outer queries unchanged',
            features='1,c,c^2,g,Sn; reduced model has logNb, direct model has log(1+H/C) as sixth coefficient',
            reduced_state='Dplus=C*sigmoid(psi),Dneutral=C-Dplus;n=Nb+H+Dplus;Nq2=n;same FD Hall moments and mu0=205',
            physics_scope='fully ionized dilute positive donors and inactive neutral pool; no atomistic complex/natives or oxygen response',
            loss='same scenario/condition/publication/related-group weights; nested L2 on feature slopes; all outcomes retained'),
        thermal=dict(bounds_logNb=[12,18],bounds_E0_eV=[.003,.3],other_bounds_unchanged=True,
            fits='two whole-publication-excluded four-parameter fits, two fixed starts,160 evaluations; zero anchors',
            extra_hall_factors=[1.,1.25,1.5,1.93],hall_scope='frozen prediction sensitivity,not measured Hall factors or confidence intervals'))
    if p.exists():
        old=json.loads(p.read_text());assert old['hashes']==hashes,'Bound code changed after protocol'
    else:save(p.name,obj)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['room','thermal']);args=parser.parse_args()
    protocol();globals()[args.task]()
