"""No-extension shared-field gradients, exact limits and source-input audit."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
from scipy.special import logit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase12_base as study
from sandbox.kroger_db import load


def main():
    energy=study.energy;core=study.core;out=study.OUT
    inputs=[energy.BASE/"artifact_manifest_phase11.json",energy.BASE/"dataset_v1.csv",
        energy.BASE/"phase8_process_temperature/coimpurity_catalogue_audit.json"]
    payload=dict(created_utc=energy.now(),code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in
        (Path(__file__),Path(study.__file__),ROOT/"src/sandbox/parent_allocation.py",
         ROOT/"docs/ML_SANDBOX_SOURCE_STOCK_ADAPTATION_20261017.md")},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        stage="no-extension field gates before new material fitting")
    pp=out/"base_numerical_protocol.json"
    if pp.exists():assert json.loads(pp.read_text())["code_hashes"]==payload["code_hashes"]
    else:energy.write(pp,payload)
    checks=[];gradients=[];replays=[]
    db=load();old=json.loads(inputs[-1].read_text())
    assert "C" not in db.elements and not old["carbon_in_current_energetic_catalogue"]
    checks.append(dict(name="carbon_catalogue_absent_no_invented_energies",passed=True))
    checks.append(dict(name="reported_H_supply_supported",passed="H" in db.elements))
    metadata=core.feature_metadata(core.FAMILIES[:-1])
    for method,definition in study.NEW_METHODS.items():
        k=1 if definition["features"]=="constant" else 7
        w=np.r_[[logit((np.log10(1500)-core.prior.L0)/(core.prior.L1-core.prior.L0)),15.5],np.linspace(-.2,.3,k)]
        for X,H,C in (("Si",0.,1e18),("Sn",0.,1e20),("Si",1.8e17,5.4e17)):
            row=SimpleNamespace(element=X,H_known=H>0,H_cm3=H,incorporated_cm3=C,growth_T_C=800.,Tobs_K=300.)
            v,J,dg,_=core.point(row,w,method,metadata)
            for j in range(len(w)):
                a=w.copy();b=w.copy();a[j]+=1e-5;b[j]-=1e-5
                fd=(core.point(row,a,method,metadata)[0]-core.point(row,b,method,metadata)[0])/(2e-5)
                error=float(np.max(abs(fd-J[:,j])/(1+abs(fd))))
                gradients.append(dict(method=method,element=X,H_cm3=H,coordinate=j,error=error))
                checks.append(dict(name="complete_noextension_field_derivative",error=error,passed=error<3e-5))
            z=w.copy();z[2:]=0.
            err=float(abs(core.point(row,z,method,metadata)[0]-core.point(row,z[:2],"original_RT",metadata)[0]).max())
            checks.append(dict(name="zero_function_original_not_neutral_extension",method=method,error=err,passed=err<2e-8))
        m,c=study.canonical(method,dict(reg=np.inf))
        checks.append(dict(name="infinite_regularization_correct_alias",method=method,passed=m=="original_RT" and not c))
    # All fifteen frozen original-RT fits and their actual RT inputs.
    for p in (study.REF/"fits").glob("*.json"):
        f=json.loads(p.read_text())
        if f["method"]!="original_RT":continue
        prediction={r["obs_id"]:r for r in f["predictions"]}
        for row in core.PRED[core.PRED.kind.eq("RT")].itertuples():
            v,_,_,_=core.point(row,np.r_[f["parameters"],0.],"stock_const_noext_RT",f["feature_metadata"])
            e=abs(v[0]-prediction[row.obs_id]["predicted"])
            replays.append(dict(fit_key=f["key"],sample_id=row.sample_id,error=float(e)))
    assert len(replays)==435
    checks.append(dict(name="real_zero_noextension_replays",count=len(replays),error=max(r["error"] for r in replays),
        passed=max(r["error"] for r in replays)<2e-8))
    result=dict(passed=all(r["passed"] for r in checks),created_utc=energy.now(),checks=checks,
        gradients=gradients,replays=replays,maximum_gradient_error=max(r["error"] for r in gradients),
        input_audit="C absent and reported surface accumulation,not homogeneous modeled supply;H explicitly supplied where measured",
        scope="numerical/source-input gates,not materials prediction accuracy")
    energy.write(out/"base_gates.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),zero_replays=len(replays),maximum_gradient=result["maximum_gradient_error"]),indent=2))
    assert result["passed"]


if __name__=="__main__":main()
