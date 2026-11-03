"""Nested family exclusion for learned internal defect relationships.

All outer outcomes are retrospective development evidence. The entire fitting,
latent adaptation and hyperparameter selection chain excludes outer query labels.
Saved fits contain parameters and preprocessing, not third-party defect arrays.
"""
from pathlib import Path
from dataclasses import dataclass
from itertools import product
from datetime import datetime,timezone
import sys,os,json,time,hashlib,argparse
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit,logsumexp
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from sandbox import kroger_db,full_forward as ff
from sandbox.implicit_defects import ImplicitDefectLayer,energy_basis,ImplicitDomainError
OUT=ROOT/"results/trainable_sandbox_20261017"
DATA=pd.read_csv(OUT/"dataset_v1.csv")
ROLES=pd.read_csv(OUT/"episodes_v1.csv")
CONTRACT=json.loads((OUT/"protocol_v1.json").read_text())
DB=kroger_db.load()
BASIS=energy_basis(DB)
LAYERS={e:ImplicitDefectLayer(DB,(e,),BASIS) for e in DATA.element.unique()}
L0,L1=np.log10(600.),np.log10(1950.)
CODE_HASH=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/"src/sandbox/implicit_defects.py").read_bytes()).hexdigest()
FITS={};PROFILES={}

def now():return datetime.now(timezone.utc).isoformat()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()[:20]
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def write(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def event(kind,**details):
    with (OUT/"benchmark_events.jsonl").open("a") as f:
        f.write(json.dumps(clean(dict(utc=now(),event=kind,**details)),ensure_ascii=False)+"\n")
def family_weights(d):
    w=1/d.family.map(d.family.value_counts()).to_numpy(float)
    return w/w.mean()
def xprocess(row):
    return (float(row.growth_T_C)+273.15-1100.)/300.
def features(d):
    z=d.log_dose.to_numpy(float)-18.
    return np.c_[z,z*z,(d.growth_T_C.to_numpy(float)+273.15-1100.)/300.,
                 d.element.eq("Sn").to_numpy(float)]

def physical_point(row,w,kind,latent=0.):
    layer=LAYERS[row.element]
    if kind=="physics_reference":
        pars=layer.parameters(T=1325.,Nd=9.1e15,doses=(row.incorporated_cm3,))
        chain=None
    elif kind in ("physics_global_process","physics_shared_process_input"):
        x=xprocess(row)
        t=w[0]+(w[2]*x if kind=="physics_shared_process_input" else 0.)
        s=expit(t);logT=L0+(L1-L0)*s+latent
        if not L0-1e-12<=logT<=L1+1e-12:
            raise ImplicitDomainError("latent_temperature_prior_support")
        pars=layer.parameters(T=10.**logT,Nd=10.**w[1],doses=(row.incorporated_cm3,))
        chain=np.zeros((layer.np,len(w)))
        chain[0,0]=(L1-L0)*s*(1-s);chain[1,1]=1.
        if len(w)==3:chain[0,2]=chain[0,0]*x
    elif kind=="physics_shared_energy":
        pars=layer.parameters(T=1325.,Nd=10.**w[1],weights=[w[0]+latent],doses=(row.incorporated_cm3,))
        chain=np.zeros((layer.np,2));chain[2,0]=1.;chain[1,1]=1.
    else:raise ValueError(kind)
    pred=layer.predict(pars,require_regular=(latent==0.))
    return pred,(pred.jacobian@chain if chain is not None else None)

def fit_physical(tr,kind):
    start=time.perf_counter()
    if kind=="physics_reference":
        best=dict(parameters=[],trials=[],loss=None)
    else:
        lo,hi=([-6.,14.],[6.,18.]) if kind!="physics_shared_energy" else ([-.5,14.],[.5,18.])
        if kind=="physics_shared_process_input":lo+=[-3.];hi+=[3.]
        def transform_T(T):return np.log((np.log10(T)-L0)/(L1-np.log10(T)))
        starts=([[transform_T(850.),15.5],[transform_T(1550.),16.5]]
                if kind!="physics_shared_energy" else [[-.2,15.5],[.2,16.5]])
        if kind=="physics_shared_process_input":starts=[x+[0.] for x in starts]
        sw=np.sqrt(family_weights(tr));y=tr.observed_log_nH.to_numpy()
        cache={}
        def evaluate(w):
            key=tuple(float(x) for x in w)
            if key not in cache:
                vals=[];jac=[]
                for row in tr.itertuples():
                    p,j=physical_point(row,w,kind)
                    vals.append(p.values[0]);jac.append(j[0])
                cache.clear()
                cache[key]=(np.array(vals),np.array(jac))
            return cache[key]
        def residual(w):
            v,j=evaluate(w)
            r=(v-y)*sw
            # A fixed weak slope prior, declared before outcomes, stabilizes the
            # input closure; model exclusion itself is selected inside CV.
            return np.r_[r,.01*w[2]] if len(w)==3 else r
        def jacobian(w):
            v,j=evaluate(w);j=j*sw[:,None]
            return np.vstack([j,[0.,0.,.01]]) if len(w)==3 else j
        trials=[]
        for initial in starts:
            try:
                opt=least_squares(residual,initial,jac=jacobian,bounds=(lo,hi),
                    max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                trials.append(dict(parameters=opt.x,loss=float(opt.fun@opt.fun),
                    success=bool(opt.success),nfev=opt.nfev,message=opt.message))
            except (ImplicitDomainError,np.linalg.LinAlgError) as e:
                trials.append(dict(parameters=initial,loss=None,success=False,error=repr(e)))
                event("physical_fit_trial_failure",model=kind,families=sorted(tr.family.unique()),error=repr(e))
        finite=[x for x in trials if x["loss"] is not None]
        if not finite:raise RuntimeError(f"All optimization starts failed for {kind}")
        best=dict(min(finite,key=lambda r:r["loss"]),trials=trials)
    # A shared transport location control uses only training mu labels. It is
    # secondary to the nH endpoint and does not enter nH fitting or selection.
    raw=[];obs=[];families=[]
    for row in tr.itertuples():
        pred,_=physical_point(row,best["parameters"],kind)
        if np.isfinite(row.mu_cm2Vs) and row.mu_cm2Vs>0:
            raw.append(pred.values[1]);obs.append(np.log10(row.mu_cm2Vs));families.append(row.family)
    f=pd.DataFrame(dict(family=families))
    offset=float(np.average(np.array(obs)-np.array(raw),weights=family_weights(f))) if raw else 0.
    return dict(kind=kind,parameters=best["parameters"],fit=best,
        transport_offset_dex=offset,transport_training_labels=len(raw),seconds=time.perf_counter()-start)

def getfit(families,spec):
    families=tuple(sorted(families))
    key=digest(dict(families=families,spec=spec,code=CODE_HASH,
                    dataset=CONTRACT["dataset"]["dataset_sha256"]))
    if key in FITS:return FITS[key]
    path=OUT/"fits"/f"{key}.json"
    if path.exists():fit=json.loads(path.read_text())
    else:
        tr=DATA[DATA.family.isin(families)].copy()
        kind=spec.split(":")[0]
        if kind.startswith("physics"):fit=fit_physical(tr,kind)
        else:
            X=features(tr);y=tr.observed_log_nH.to_numpy();weights=family_weights(tr)
            if kind=="constant":
                coef=[float(np.average(y,weights=weights))]
                center=[];scale=[]
            elif kind=="unit_dose":
                coef=[float(np.average(y-tr.log_dose,weights=weights))]
                center=[];scale=[]
            elif kind=="line_dose":
                A=np.c_[np.ones(len(tr)),tr.log_dose.to_numpy()-18.]
                coef=np.linalg.lstsq(A*np.sqrt(weights[:,None]),y*np.sqrt(weights),rcond=None)[0]
                center=[];scale=[]
            elif kind=="Ridge_process":
                center=X.mean(axis=0);scale=X.std(axis=0);scale[scale<1e-12]=1.
                model=Ridge(alpha=float(spec.split(":")[1])).fit((X-center)/scale,y,sample_weight=weights)
                coef=np.r_[model.intercept_,model.coef_]
            else:raise ValueError(kind)
            use=tr.mu_cm2Vs.notna() & tr.mu_cm2Vs.gt(0)
            # Same available property labels, input descriptors and train-only scaling.
            xm=X[use];ym=np.log10(tr.loc[use,"mu_cm2Vs"].to_numpy())
            if len(ym):
                mu_center=xm.mean(axis=0);mu_scale=xm.std(axis=0);mu_scale[mu_scale<1e-12]=1.
                a=float(spec.split(":")[1]) if kind=="Ridge_process" else 1.
                mm=Ridge(alpha=a).fit((xm-mu_center)/mu_scale,ym,sample_weight=family_weights(tr[use]))
                mc=np.r_[mm.intercept_,mm.coef_]
            else:mc=np.array([2.,0.,0.,0.,0.]);mu_center=np.zeros(4);mu_scale=np.ones(4)
            fit=dict(kind=kind,parameters=coef,center=center,scale=scale,
                     mu_parameters=mc,mu_center=mu_center,mu_scale=mu_scale)
        fit.update(key=key,spec=spec,train_families=families,train_records=len(tr),code_sha256=CODE_HASH)
        write(path,fit)
        print("fit",spec,len(families),families,flush=True)
        event("fit_complete",key=key,spec=spec,train_families=families,seconds=fit.get("seconds"))
    FITS[key]=fit
    return fit

def empirical_values(fit,d):
    X=features(d);w=np.array(fit["parameters"]);kind=fit["kind"]
    if kind=="constant":y=np.full(len(d),w[0])
    elif kind=="unit_dose":y=d.log_dose.to_numpy()+w[0]
    elif kind=="line_dose":y=w[0]+w[1]*(d.log_dose.to_numpy()-18.)
    else:y=w[0]+((X-np.array(fit["center"]))/np.array(fit["scale"]))@w[1:]
    mw=np.array(fit["mu_parameters"])
    mu=mw[0]+((X-np.array(fit["mu_center"]))/np.array(fit["mu_scale"]))@mw[1:]
    return np.c_[y,mu,np.log10(1.602176634e-19)+y+mu]

def profile(fit,family,physical_latent=False,grid_size=41):
    key=digest(dict(fit=fit["key"],family=family,physical_latent=physical_latent,
                    grid_size=grid_size,profile_version=1))
    if key in PROFILES:return PROFILES[key]
    path=OUT/"profiles"/f"{key}.npz"
    if path.exists():
        f=np.load(path);out={k:f[k] for k in f.files};PROFILES[key]=out;return out
    d=DATA[DATA.family.eq(family)].copy()
    kind=fit["kind"]
    if not kind.startswith("physics"):
        values=empirical_values(fit,d)[None,:,:];grid=np.array([0.])
        valid=np.ones((1,len(d)),bool);margins=np.full((1,len(d)),np.nan)
    else:
        limit=.8 if kind=="physics_shared_energy" else .2
        grid=np.linspace(-limit,limit,grid_size) if physical_latent else np.array([0.])
        values=[];valid=[];margins=[];kept=[]
        for latent in grid:
            vals=[];vs=[];ms=[]
            supported=True
            for row in d.itertuples():
                try:
                    pred,_=physical_point(row,fit["parameters"],kind,float(latent))
                except ImplicitDomainError as e:
                    if str(e)=="latent_temperature_prior_support":
                        supported=False;break
                    event("profile_failure",spec=fit["spec"],family=family,latent=latent,error=repr(e))
                    raise
                value=pred.values.copy()
                value[1:]+=fit["transport_offset_dex"]
                vals.append(value)
                diag=pred.diagnostics
                ms.append(diag["compensation_margin"])
                ms_star=ff.m_star(diag["Tf_K"])[0]
                vs.append(diag["gradient_valid"] and diag["single_electron_Hall_valid"]
                    and diag["above_carrier_detection_floor"] and abs(diag["compensation_margin"])>=ms_star)
            if supported:
                values.append(vals);valid.append(vs);margins.append(ms);kept.append(latent)
        grid=np.array(kept);values=np.array(values);valid=np.array(valid);margins=np.array(margins)
        if not len(grid):raise RuntimeError("Empty declared prior support")
    out=dict(grid=grid,values=values,valid=valid,margins=margins,
             sample_ids=d.sample_id.to_numpy(str))
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**out)
    PROFILES[key]=out;return out

def evaluate(fit,family,budget,mode,tau,sigma,*,grid_size=41):
    d=DATA[DATA.family.eq(family)].copy().reset_index(drop=True)
    roles=ROLES[ROLES.family.eq(family)&ROLES.budget.eq(budget)].set_index("sample_id").role
    anchor=np.array([roles.loc[x]=="anchor" for x in d.sample_id])
    query=np.array([roles.loc[x]=="query" for x in d.sample_id])
    p=profile(fit,family,mode=="physical",grid_size)
    if mode=="physical":
        grid=p["grid"]
        if tau==0:
            z=int(np.argmin(abs(grid)))
            logprior=np.full(len(grid),-np.inf);logprior[z]=0.
        else:logprior=-.5*(grid/tau)**2
        # Conditional finite-support Gaussian prior; no hidden interpolation.
        logweights=logprior-.5*np.sum(((p["values"][:,anchor,0]-d.observed_log_nH.to_numpy()[anchor])/sigma)**2,axis=1)
        weights=np.exp(logweights-logsumexp(logweights))
        values=np.einsum("g,gij->ij",weights,p["values"])
        variance=np.einsum("g,gi->i",weights,(p["values"][:,:,0]-values[:,0])**2)+sigma*sigma
        valid_mass=weights@p["valid"].astype(float)
        latent_mean=float(weights@grid)
        latent_sd=float(np.sqrt(weights@(grid-latent_mean)**2))
    else:
        values=p["values"][0].copy()
        residual=d.observed_log_nH.to_numpy()[anchor]-values[anchor,0]
        v=0. if tau==0 else 1/(1/tau**2+anchor.sum()/sigma**2)
        delta=v*residual.sum()/sigma**2
        values[:,0]+=delta;values[:,2]+=delta
        variance=np.full(len(d),v+sigma*sigma)
        valid_mass=p["valid"][0].astype(float)
        latent_mean=float(delta);latent_sd=float(np.sqrt(v))
    y=d.observed_log_nH.to_numpy()
    return dict(rows=d,query=query,values=values,variance=variance,valid_mass=valid_mass,
                latent_mean=latent_mean,latent_sd=latent_sd,
                mae=float(np.abs(values[query,0]-y[query]).mean()))

def methods():
    out=[]
    for name in ["constant","unit_dose","line_dose","Ridge_process","physics_reference",
                 "physics_global_process","physics_shared_process_input","physics_shared_energy"]:
        specs=[f"Ridge_process:{a}" for a in CONTRACT["ridge_alpha"]] if name=="Ridge_process" else [name]
        candidates=[dict(spec=s,mode="offset",tau=t,sigma=n) for s,t,n in product(
            specs,CONTRACT["offset_prior_sd_dex"],CONTRACT["within_noise_dex"])]
        out.append(dict(name=name+"_offset",candidates=candidates))
    for name in ["physics_global_process","physics_shared_process_input","physics_shared_energy"]:
        grid=CONTRACT["energy_prior_sd_eV"] if name.endswith("energy") else CONTRACT["process_prior_sd_log10T"]
        candidates=[dict(spec=name,mode="physical",tau=t,sigma=n) for t,n in product(grid,CONTRACT["within_noise_dex"])]
        out.append(dict(name=name+"_latent",candidates=candidates))
    return out

def freeze_implementation():
    path=OUT/"benchmark_implementation_v1.json"
    payload=dict(code_sha256=CODE_HASH,process_mapping="bounded sigmoid log10 Tf on growth temperature; w=0 global",
        physical_fit_starts=2,max_nfev_per_start=60,analytic_IFT_jacobian=True,
        process_input_slope_penalty=.01,process_prior_grid=[-.2,.2,41],energy_prior_grid=[-.8,.8,41],
        prior_support="finite Gaussian-weighted latent grid; process truncated to 600..1950 K based on inputs",
        property_labels="nH trains internal relation; training-only shared mu location control; no target mu anchor labels",
        methods=methods(),selection="inner equal-family query MAE separately per anchor budget",
        fixed_pressure_atm=1e-5,validation_grade="retrospective",
        numerical_gate=json.loads((OUT/"numerical_gate_cpu.json").read_text())["passed"])
    if path.exists():
        old=json.loads(path.read_text())
        if old["code_sha256"]!=CODE_HASH:raise RuntimeError("Implementation changed; version protocol before running")
    else:write(path,payload)
    assert payload["numerical_gate"]

def run(helds):
    freeze_implementation()
    allfamilies=sorted(DATA.family.unique());specs=methods()
    for held in helds:
        target=OUT/"outer"/f"{digest(held)}.csv"
        if target.exists():continue
        training=[x for x in allfamilies if x!=held]
        records=[];selections=[]
        for method in specs:
            for budget in (0,1,2):
                scored=[]
                for candidate in method["candidates"]:
                    errors=[]
                    for inner in training:
                        families=[x for x in training if x!=inner]
                        fit=getfit(families,candidate["spec"])
                        errors.append(evaluate(fit,inner,budget,**{k:v for k,v in candidate.items() if k!="spec"})["mae"])
                    scored.append(dict(candidate,inner_mae=float(np.mean(errors))))
                # Grid iteration and the tolerance tie rule prefer no adaptation
                # and stronger Ridge when effectively indistinguishable.
                best=min(scored,key=lambda x:(round(x["inner_mae"],10),x["tau"],
                    -float(x["spec"].split(":")[1]) if ":" in x["spec"] else 0.))
                fit=getfit(training,best["spec"])
                args={k:best[k] for k in ("mode","tau","sigma")}
                pred=evaluate(fit,held,budget,**args)
                selections.append(dict(method=method["name"],budget=budget,best=best,all_candidates=scored,
                    outer_family=held,fit_key=fit["key"]))
                for i,row in enumerate(pred["rows"].itertuples()):
                    role=ROLES[(ROLES.family==held)&(ROLES.budget==budget)&(ROLES.sample_id==row.sample_id)].role.iloc[0]
                    value=pred["values"][i]
                    mu=float(np.log10(row.mu_cm2Vs)) if np.isfinite(row.mu_cm2Vs) and row.mu_cm2Vs>0 else None
                    records.append(dict(family=held,sample_id=row.sample_id,element=row.element,
                        method=method["name"],budget=budget,role=role,observed_log_nH=row.observed_log_nH,
                        predicted_log_nH=value[0],observed_log_muH=mu,predicted_log_muH=value[1],
                        observed_log_sigma=(np.log10(1.602176634e-19)+row.observed_log_nH+mu) if mu is not None else None,
                        predicted_log_sigma=value[2],nH_model_sd=np.sqrt(pred["variance"][i]),
                        physical_valid_mass=pred["valid_mass"][i],latent_mean=pred["latent_mean"],
                        latent_sd=pred["latent_sd"],selected_mode=best["mode"],selected_tau=best["tau"],
                        selected_sigma=best["sigma"],selected_spec=best["spec"],
                        inner_mae=best["inner_mae"],train_families="|".join(training),
                        fit_key=fit["key"],oxygen_pressure_is_scenario=True))
                print("outer",held,method["name"],budget,"inner",best["inner_mae"],
                      "outer",pred["mae"],flush=True)
            write(OUT/"selections"/f"{digest([held,method['name']])}.json",selections[-3:])
            pd.DataFrame(records).to_csv(OUT/"outer"/f"{digest(held)}.partial.csv",index=False)
        target.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(records).to_csv(target,index=False)
        event("outer_complete",family=held,rows=len(records))
    summarize()

def summarize():
    files=[p for p in (OUT/"outer").glob("*.csv") if ".partial." not in p.name]
    if not files:return
    d=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    d.to_csv(OUT/"predictions_v1.csv",index=False)
    q=d[d.role.eq("query")].copy()
    rows=[]
    for (method,budget,family),g in q.groupby(["method","budget","family"]):
        err=g.predicted_log_nH-g.observed_log_nH
        row=dict(method=method,budget=budget,family=family,n=len(g),mae=float(abs(err).mean()),
                 rmse=float(np.sqrt(np.mean(err**2))),returned=int(g.predicted_log_nH.notna().sum()),
                 legacy_physical_covered=int(g.physical_valid_mass.ge(.95).sum()),
                 conditional_95_coverage=float((abs(err)<=1.96*g.nH_model_sd).mean()))
        for prop in ("muH","sigma"):
            good=g["observed_log_"+prop].notna()
            row[prop+"_mae"]=float(abs(g.loc[good,"predicted_log_"+prop]-g.loc[good,"observed_log_"+prop]).mean())
        rows.append(row)
    by=pd.DataFrame(rows);by.to_csv(OUT/"by_family_v1.csv",index=False)
    agg=by.groupby(["method","budget"]).agg(families=("family","nunique"),query_records=("n","sum"),
        family_equal_mae=("mae","mean"),worst_family_mae=("mae","max"),
        family_equal_rmse=("rmse","mean"),numeric_returned=("returned","sum"),
        legacy_physical_covered=("legacy_physical_covered","sum"),
        conditional_95_coverage=("conditional_95_coverage","mean"),
        mobility_family_equal_mae=("muH_mae","mean"),conductivity_family_equal_mae=("sigma_mae","mean")).reset_index()
    agg.to_csv(OUT/"summary_v1.csv",index=False)
    write(OUT/"progress.json",dict(status="active",stage="nested_prediction_benchmark",
        completed_outer_families=int(by.family.nunique()),total_outer_families=5,
        numerical_gate_passed=True,physics_internal_learning_implemented=True,
        complete_goal=False,validation_grade="retrospective",latest_utc=now(),
        next=["inspect complete same-budget comparisons","test concrete remaining model/input risks",
              "energy-label and target-directed selection stage"]))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--held-family");ap.add_argument("--summarize",action="store_true")
    a=ap.parse_args()
    (OUT/"outer").mkdir(parents=True,exist_ok=True)
    if a.summarize:summarize()
    else:run([a.held_family] if a.held_family else sorted(DATA.family.unique()))
