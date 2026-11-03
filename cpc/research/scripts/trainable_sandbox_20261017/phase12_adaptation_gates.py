"""Independent scalar-search and saved physical-map checks before evaluation."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
from scipy.special import logit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.scalar_profile_adaptation import ScalarProfile
from sandbox.source_stock_model import SourceStockModel
from sandbox.material_profile import PUBLISHED_PROFILE
import phase12_base as study


def main():
    e=study.energy;core=study.core;out=study.OUT
    files=[Path(__file__),ROOT/"src/sandbox/scalar_profile_adaptation.py",ROOT/"src/sandbox/source_stock_model.py"]
    p=dict(created_utc=e.now(),code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in files},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (out/"base_gates.json",e.BASE/"artifact_manifest_phase11.json")},
        scope="scalar profile/map and actual zero-update checks before adaptive comparison")
    path=out/"adaptation_numerical_protocol.json"
    if path.exists():assert json.loads(path.read_text())["code_hashes"]==p["code_hashes"]
    else:e.write(path,p)
    checks=[];gradients=[];replays=[]
    a=np.array([1.2,-.8]);base=np.array([.5,-.2]);y=np.array([1.1,.1])
    profile=ScalarProfile(lambda z:(base+a*z,a))
    for lam in (0.,.001,.1,1.,10.):
        r=profile.fit(y,lam);expected=float(a@(y-base)/(a@a+lam));err=abs(r.shift-expected)
        checks.append(dict(name="linear_MAP_analytic_solution",lam=lam,error=err,passed=err<2e-6))
    count=profile.forward_calls;r=profile.fit(y,np.inf)
    checks.append(dict(name="infinite_shrinkage_no_calls_exact_zero",passed=r.shift==0 and profile.forward_calls==count))
    r=profile.fit([],1.)
    checks.append(dict(name="zero_anchor_no_calls_exact_zero",passed=r.shift==0 and profile.forward_calls==count))
    def plateau(z):
        t=np.tanh((z-2.5)/.15)
        return np.array([.2*t]),np.array([.2/.15*(1-t*t)])
    r=ScalarProfile(plateau).fit([0.],.0001)
    checks.append(dict(name="nonlocal_profile_overcomes_flat_start",shift=r.shift,objective=r.diagnostics["objective"],
        passed=r.shift>2 and r.diagnostics["objective"]<.002))
    r=ScalarProfile(lambda z:(np.array([z*z-4]),np.array([2*z]))).fit([0.],0.)
    roots=[c["shift"] for c in r.candidates if c["valid"] and c["objective"]<1e-8]
    checks.append(dict(name="multiple_minima_retained",passed=any(x< -1.9 for x in roots) and any(x>1.9 for x in roots)))
    def partial(z):
        if z>1:raise ValueError("synthetic invalid state")
        return np.array([z]),np.array([1.])
    r=ScalarProfile(partial).fit([3.],0.)
    checks.append(dict(name="invalid_state_not_selected",passed=r.shift<=1 and bool(r.diagnostics["state_failures"])))
    meta=core.feature_metadata(core.FAMILIES[:-1])
    modes=[("original_RT","stock"),("original_RT","Tf"),("original_RT","offset"),
        ("stock_rbf_noext_RT","stock"),("stock_rbf_noext_RT","offset"),
        ("joint_baseline","stock"),("parent_rbf_joint","stock"),
        ("out_rbf_noext_RT","offset"),("tf_rbf_noext_RT","offset")]
    for method,kind in modes:
        definition=core.METHODS[method];nbase=4 if definition["training"]=="joint" else 2
        k=0 if definition["field"]=="none" else 7
        w=np.r_[[logit((np.log10(1500)-core.prior.L0)/(core.prior.L1-core.prior.L0)),15.5],
            [np.log10(.06),.5] if nbase==4 else [],np.linspace(-.15,.3,k)]
        model=SourceStockModel(w,definition,meta,profile=PUBLISHED_PROFILE,pO2_atm=1e-5)
        for X,H,C,T in (("Si",0.,1e18,300.),("Sn",0.,1e20,300.),("Si",1.8e17,5.4e17,150.)):
            # Deliberately has no observed/property labels or sample identifier.
            row=SimpleNamespace(element=X,H_known=H>0,H_cm3=H,incorporated_cm3=C,growth_T_C=800.,Tobs_K=T)
            zero=model.predict(row,source_kind=kind);legacy=core.point(row,w,method,meta)[0]
            error=float(abs(zero.values-legacy).max())
            checks.append(dict(name="saved_map_exact_zero",method=method,kind=kind,error=error,passed=error<2e-8))
            for z in (-1.,0.,1.):
                val=model.predict(row,source_kind=kind,source_shift=z)
                fd=(model.predict(row,source_kind=kind,source_shift=z+1e-5).values-
                    model.predict(row,source_kind=kind,source_shift=z-1e-5).values)/(2e-5)
                error=float(np.max(abs(fd-val.source_jacobian)/(1+abs(fd))))
                gradients.append(dict(method=method,kind=kind,element=X,T_K=T,z=z,error=error))
                checks.append(dict(name="actual_coupled_source_derivative",error=error,passed=error<3e-5))
    for p in (study.REF/"fits").glob("*.json"):
        f=json.loads(p.read_text())
        if f["method"] not in ("original_RT","joint_baseline"):continue
        model=SourceStockModel(f["parameters"],core.METHODS[f["method"]],f["feature_metadata"],
            profile=PUBLISHED_PROFILE,pO2_atm=1e-5)
        pred={r["obs_id"]:r for r in f["predictions"]}
        for row in core.PRED[core.PRED.kind.eq("RT")].itertuples():
            value=model.predict(row,source_kind="stock")
            err=abs(value.values[0]-pred[row.obs_id]["predicted"])
            replays.append(dict(fit_key=f["key"],obs_id=row.obs_id,error=float(err)))
    assert len(replays)==870
    checks.append(dict(name="real_saved_models_zero_replays",count=len(replays),error=max(r["error"] for r in replays),
        passed=max(r["error"] for r in replays)<2e-8))
    result=dict(passed=all(c["passed"] for c in checks),created_utc=e.now(),checks=checks,gradients=gradients,
        replays=replays,maximum_source_derivative_error=max(r["error"] for r in gradients),
        inference_scope="physical map reads no training tables and accepts no measurement label",
        scope="analytic/synthetic/map checks and real-state replay,not new material accuracy")
    e.write(out/"adaptation_gates.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),real_zero_replays=len(replays),
        maximum_derivative=result["maximum_source_derivative_error"]),indent=2))
    assert result["passed"]


if __name__=="__main__":main()
