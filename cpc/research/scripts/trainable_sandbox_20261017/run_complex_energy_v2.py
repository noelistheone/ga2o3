"""V2: regularized parent-consistent complex formation-energy learning.

Motivated by V1 target sensitivities; this is retrospective model development.
Tf is fixed; native and compensating-complex energy corrections share across
samples. Each outer and inner fit excludes its respective family outcomes.
"""
from pathlib import Path
from itertools import product
import os,sys,json,hashlib,time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).resolve().parent))
from sandbox import kroger_db,full_forward as ff
from sandbox.implicit_defects import ImplicitDefectLayer,ImplicitDomainError
from diagnose_energy_basis import bases
import run_benchmark as v1

OUT=v1.OUT/"v2_complex"
OUT.mkdir(parents=True,exist_ok=True)
DATA=v1.DATA;ROLES=v1.ROLES
DB=kroger_db.load();B,NAMES=bases(DB);B=B[:,:2];NAMES=NAMES[:2]
LAYERS={e:ImplicitDefectLayer(DB,(e,),B) for e in DATA.element.unique()}
CODE=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/"src/sandbox/implicit_defects.py").read_bytes()
    +Path(__file__).with_name("diagnose_energy_basis.py").read_bytes()).hexdigest()
FITS={};PROFILES={}
def write(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(v1.clean(x),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def key(x):return v1.digest(dict(payload=x,code=CODE))

def point(row,w,latent=0.):
    layer=LAYERS[row.element]
    pars=layer.parameters(T=1325.,Nd=10.**w[2],weights=[w[0],w[1]+latent],doses=(row.incorporated_cm3,))
    result=layer.predict(pars,require_regular=(latent==0.))
    return result,result.jacobian[:,[2,3,1]]

def fit(families,penalty):
    families=tuple(sorted(families));ident=key([families,penalty])
    if ident in FITS:return FITS[ident]
    path=OUT/"fits"/f"{ident}.json"
    if path.exists():result=json.loads(path.read_text())
    else:
        tr=DATA[DATA.family.isin(families)]
        y=tr.observed_log_nH.to_numpy();sw=np.sqrt(v1.family_weights(tr));cache={}
        start=time.perf_counter();rootstart=sum(x.root_calls for x in LAYERS.values())
        def values(w):
            k=tuple(w)
            if k not in cache:
                pred=[];jac=[]
                for row in tr.itertuples():
                    r,j=point(row,w);pred.append(r.values[0]);jac.append(j[0])
                cache.clear();cache[k]=(np.array(pred),np.array(jac))
            return cache[k]
        def fun(w):return np.r_[(values(w)[0]-y)*sw,penalty*w[:2]]
        def jac(w):return np.vstack([values(w)[1]*sw[:,None],[[penalty,0.,0.],[0.,penalty,0.]]])
        trials=[]
        for initial in [[0.,0.,16.],[-.2,.2,16.]]:
            try:
                opt=least_squares(fun,initial,jac=jac,bounds=([-.5,-.5,14.],[.5,.5,18.]),
                    max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                trials.append(dict(parameters=opt.x,loss=float(opt.fun@opt.fun),
                    success=bool(opt.success),nfev=opt.nfev,message=opt.message))
            except (ImplicitDomainError,np.linalg.LinAlgError) as e:
                trials.append(dict(parameters=initial,loss=None,error=repr(e),success=False))
        good=[x for x in trials if x["loss"] is not None]
        if not good:raise RuntimeError("No valid V2 fit: "+str(trials))
        best=min(good,key=lambda x:x["loss"])
        # Eligible labels only, as required by V1's secondary label audit.
        pairs=tr[tr.mobility_eligible.fillna(False).astype(bool)&tr.mu_cm2Vs.gt(0)]
        raw=np.array([point(row,best["parameters"])[0].values[1] for row in pairs.itertuples()])
        offset=float(np.average(np.log10(pairs.mu_cm2Vs.to_numpy())-raw,
            weights=v1.family_weights(pairs))) if len(pairs) else 0.
        result=dict(key=ident,train_families=families,train_records=len(tr),parameters=best["parameters"],
            penalty=penalty,trials=trials,transport_offset_dex=offset,transport_labels=len(pairs),
            root_calls=sum(x.root_calls for x in LAYERS.values())-rootstart,
            seconds=time.perf_counter()-start,code_sha256=CODE)
        write(path,result);print("V2 fit",families,penalty,"w",best["parameters"],flush=True)
    FITS[ident]=result;return result

def profile(f,family,latent):
    ident=key([f["key"],family,latent])
    if ident in PROFILES:return PROFILES[ident]
    path=OUT/"profiles"/f"{ident}.npz"
    if path.exists():
        z=np.load(path);out={k:z[k] for k in z.files};PROFILES[ident]=out;return out
    d=DATA[DATA.family.eq(family)]
    grid=np.linspace(-.8,.8,41) if latent else np.array([0.])
    vals=[];valid=[]
    for z in grid:
        vv=[];vd=[]
        for row in d.itertuples():
            r,_=point(row,f["parameters"],float(z));value=r.values.copy();value[1:]+=f["transport_offset_dex"]
            vv.append(value);dg=r.diagnostics
            vd.append(dg["gradient_valid"] and dg["single_electron_Hall_valid"]
                and dg["above_carrier_detection_floor"] and abs(dg["compensation_margin"])>=ff.m_star(dg["Tf_K"])[0])
        vals.append(vv);valid.append(vd)
    out=dict(values=np.array(vals),valid=np.array(valid),grid=grid)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**out)
    PROFILES[ident]=out;return out

def predict(f,family,budget,mode,tau,sigma):
    d=DATA[DATA.family.eq(family)].copy().reset_index(drop=True)
    roles=ROLES[(ROLES.family==family)&ROLES.budget.eq(budget)].set_index("sample_id").role
    anchor=np.array([roles.loc[x]=="anchor" for x in d.sample_id]);query=np.array([roles.loc[x]=="query" for x in d.sample_id])
    pr=profile(f,family,mode=="physical");y=d.observed_log_nH.to_numpy()
    if mode=="physical":
        z=pr["grid"];logprior=-.5*(z/tau)**2 if tau else np.where(abs(z)<1e-9,0.,-np.inf)
        weights=logprior-.5*np.sum(((pr["values"][:,anchor,0]-y[anchor])/sigma)**2,axis=1)
        weights=np.exp(weights-logsumexp(weights))
        pred=np.einsum("g,gij->ij",weights,pr["values"])
        mass=weights@pr["valid"]
        mean=float(weights@z);variance=np.einsum("g,gi->i",weights,(pr["values"][:,:,0]-pred[:,0])**2)+sigma*sigma
    else:
        pred=pr["values"][0].copy()
        v=0. if tau==0 else 1/(1/tau**2+anchor.sum()/sigma**2)
        mean=float(v*np.sum(y[anchor]-pred[anchor,0])/sigma**2)
        pred[:,0]+=mean;pred[:,2]+=mean
        mass=pr["valid"][0];variance=np.full(len(d),v+sigma*sigma)
    return dict(data=d,values=pred,roles=roles,mass=mass,variance=variance,latent_mean=mean,
        mae=float(abs(pred[query,0]-y[query]).mean()))

def run():
    assert json.loads((v1.OUT/"multi_basis_gradient_gate.json").read_text())["passed"]
    protocol=OUT/"protocol_v2.json"
    if protocol.exists():
        assert json.loads(protocol.read_text())["code_sha256"]==CODE,"Version changed code before rerunning"
    else:
        write(protocol,dict(version=2,code_sha256=CODE,created_utc=v1.now(),
            hypothesis="V1 isolated-vacancy basis misses target-sensitive compensating dopant complexes.",
            basis_names=NAMES,energy_shift="parent-consistent shared shifts, fixed Tf=1325 K",
            penalties=[.1,1.,3.],energy_bounds_eV=[-.5,.5],Nd_log10_bounds=[14.,18.],
            noise_grid=[.1,.3,.6,1.],offset_prior_sd=[0.,.1,.3,.6,1.],physical_prior_sd_eV=[0.,.05,.15,.35],
            latent="one complex-energy direction, Gaussian-weighted 41-point finite support [-.8,.8] eV",
            split="same nested leave-family-out and same 0/1/2 anchors as V1",
            validation_grade="retrospective model development informed by V1, not an independent blind test",
            property_scope="nH primary; mu/sigma secondary use eligible labels and training-only offset",
            energy_interpretation="effective inventory closure, not independently validated microscopic energy correction"))
    families=sorted(DATA.family.unique())
    for held in families:
        path=OUT/"outer"/f"{key(held)}.csv"
        if path.exists():continue
        training=[f for f in families if f!=held];records=[];selection=[]
        for mode in ("offset","physical"):
            taus=[0.,.1,.3,.6,1.] if mode=="offset" else [0.,.05,.15,.35]
            for budget in (0,1,2):
                candidates=[]
                for penalty,tau,sigma in product([.1,1.,3.],taus,[.1,.3,.6,1.]):
                    errors=[]
                    for inner in training:
                        f=fit([x for x in training if x!=inner],penalty)
                        errors.append(predict(f,inner,budget,mode,tau,sigma)["mae"])
                    candidates.append(dict(penalty=penalty,tau=tau,sigma=sigma,inner_mae=float(np.mean(errors))))
                best=min(candidates,key=lambda x:(round(x["inner_mae"],10),x["tau"],-x["penalty"]))
                f=fit(training,best["penalty"])
                pred=predict(f,held,budget,mode,best["tau"],best["sigma"])
                name="physics_complex_energy_"+("latent" if mode=="physical" else "offset")
                selection.append(dict(family=held,budget=budget,method=name,best=best,candidates=candidates,fit_key=f["key"]))
                for i,row in enumerate(pred["data"].itertuples()):
                    val=pred["values"][i]
                    records.append(dict(method=name,budget=budget,family=held,sample_id=row.sample_id,
                        role=pred["roles"].loc[row.sample_id],observed_log_nH=row.observed_log_nH,
                        predicted_log_nH=val[0],predicted_log_muH=val[1],predicted_log_sigma=val[2],
                        physical_valid_mass=pred["mass"][i],conditional_nH_sd=np.sqrt(pred["variance"][i]),
                        latent_mean=pred["latent_mean"],selected_penalty=best["penalty"],selected_tau=best["tau"],
                        selected_sigma=best["sigma"],inner_mae=best["inner_mae"],fit_key=f["key"],
                        train_families="|".join(training)))
                print("V2 outer",held,name,budget,"inner",best["inner_mae"],"outer",pred["mae"],flush=True)
        write(OUT/"selections"/f"{key(held)}.json",selection)
        path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(records).to_csv(path,index=False)
    d=pd.concat([pd.read_csv(x) for x in (OUT/"outer").glob("*.csv")],ignore_index=True)
    d.to_csv(OUT/"predictions_v2.csv",index=False);q=d[d.role.eq("query")]
    rows=[]
    for keys,g in q.groupby(["method","budget","family"]):
        e=g.predicted_log_nH-g.observed_log_nH
        rows.append(dict(method=keys[0],budget=keys[1],family=keys[2],n=len(g),
            mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),
            physically_covered=int(g.physical_valid_mass.ge(.95).sum())))
    by=pd.DataFrame(rows);by.to_csv(OUT/"by_family_v2.csv",index=False)
    by.groupby(["method","budget"]).agg(families=("family","nunique"),n=("n","sum"),
        family_equal_mae=("mae","mean"),worst_family_mae=("mae","max"),
        physically_covered=("physically_covered","sum")).reset_index().to_csv(OUT/"summary_v2.csv",index=False)

if __name__=="__main__":run()
