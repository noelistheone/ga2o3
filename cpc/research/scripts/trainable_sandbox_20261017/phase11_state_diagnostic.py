"""Separate learned inventory effects from refitting global physical parameters."""
from pathlib import Path
import sys,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase11_prediction as study
import phase2_energy as energy


def inventory(row,fit,zero=False):
    p=np.asarray(fit["parameters"],float).copy()
    nbase=4 if study.METHODS[fit["method"]]["training"]=="joint" else 2
    if zero:p[nbase:]=0.
    values,_,diag,extra=study.point(row,p,fit["method"],fit["feature_metadata"])
    layer=study.layer_for(row,study.METHODS[fit["method"]]["thermal"])
    core=layer.core;state=core.last_state
    original=core.original_parent_ids[core.first]
    atoms=core.db.dm[core.first,core.db.col(row.element)]
    amounts=state["totals"]*atoms
    assert np.isclose(amounts.sum(),row.incorporated_cm3,rtol=2e-7)
    isolated=np.isin(original,layer.contrast.isolated_parent_ids)
    h=(core.db.dm[core.first,core.db.col("H")]>0)
    neg=(core.db.dm[:,core.db.col(row.element)]>0)&(core.q<0)
    out=dict(log10_nH=float(values[0]),log10_Nabs=float(values[1]),log10_Nq2=float(values[2]),
        process_bias=extra["process_bias"],Tf_K=extra["Tf_K"],
        isolated_X_atom_fraction=float(amounts[isolated].sum()/row.incorporated_cm3),
        multi_X_atom_fraction=float(amounts[atoms>=2].sum()/row.incorporated_cm3),
        H_complex_X_atom_fraction=float(amounts[h].sum()/row.incorporated_cm3),
        negative_X_complex_charge_cm3=float(-(core.q[neg]@state["cold_charge"][neg])),
        maximum_site_occupancy=diag["max_site_occupancy"],
        root_residual=diag["root_residual_log"],gradient_valid=diag["gradient_valid"],
        stock_scope="model-implied inventory,not a unique measured microscopic population")
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--directory",default="phase11_parent_allocation");a=ap.parse_args()
    folder=energy.BASE/a.directory
    assert json.loads((folder/"contract_audit.json").read_text())["passed"]
    payload=dict(created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),
        audit_sha256=energy.sha(folder/"contract_audit.json"),
        scope="post-result diagnostic only;no fitting/selection;raw source-excluded states before anchor output correction",
        counterfactual="same trained global coordinates with psi=0,versus separately fitted matching no-field baseline",
        limits="stock attribution conditional on supplied catalogue/profile/boundaries;no true vacancy count,scattering law or compensation mechanism identified")
    path=folder/"state_diagnostic_protocol.json"
    if path.exists():assert json.loads(path.read_text())["code_sha256"]==payload["code_sha256"]
    else:energy.write(path,payload)
    fits={p.stem:json.loads(p.read_text()) for p in (folder/"fits").glob("*.json")}
    lookup={(f["method"],tuple(f["train_families"])):f for f in fits.values() if not f["config"]}
    rows=[];checks=[]
    for method in ("parent_constant_joint","parent_linear_joint","parent_rbf_joint","parent_rbf_RT"):
        choices=json.loads((folder/"methods"/method/"selections.json").read_text())
        for choice in choices:
            f=fits[choice["fit_key"]]
            baseline="joint_baseline" if study.METHODS[method]["training"]=="joint" else "thermal_RT"
            base=lookup[baseline,tuple(f["train_families"])]
            for r in study.PRED[study.PRED.family.eq(choice["held"])&study.PRED.kind.eq("RT")].itertuples():
                try:
                    trained=inventory(r,f);zero=inventory(r,f,zero=True);control=inventory(r,base)
                    total=trained["log10_nH"]-control["log10_nH"]
                    inventory_delta=trained["log10_nH"]-zero["log10_nH"]
                    global_delta=zero["log10_nH"]-control["log10_nH"]
                    identity_error=abs(total-inventory_delta-global_delta)
                    assert identity_error<1e-12
                    checks.append(identity_error)
                    rows.append(dict(method=method,budget=choice["budget"],family=choice["held"],
                        sample_id=r.sample_id,fit_key=f["key"],baseline_fit_key=base["key"],canonical_method=f["method"],
                        observed=r.target,chemical_dose_cm3=r.incorporated_cm3,
                        total_logn_change=total,inventory_logn_effect=inventory_delta,global_refit_logn_effect=global_delta,
                        **{"trained_"+k:v for k,v in trained.items()},
                        **{"zero_field_"+k:v for k,v in zero.items()},
                        **{"control_"+k:v for k,v in control.items()},numeric=True))
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    rows.append(dict(method=method,budget=choice["budget"],family=choice["held"],
                        sample_id=r.sample_id,fit_key=f["key"],numeric=False,error=repr(exc)))
    energy.csvwrite(folder/"inventory_counterfactuals.csv",rows)
    energy.write(folder/"state_diagnostic_audit.json",dict(passed=all(r["numeric"] for r in rows),
        rows=len(rows),identity_checks=len(checks),maximum_additive_identity_error=max(checks),
        rejected_rows=sum(not r["numeric"] for r in rows),
        interpretation="raw model-state counterfactuals,not independent physical measurements or causal identification"))
    print("state diagnostics",len(rows),"rows",flush=True)


if __name__=="__main__":main()
