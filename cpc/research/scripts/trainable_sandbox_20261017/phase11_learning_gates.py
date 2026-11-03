"""Check complete trainable field Jacobians and exact no-learning limits."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
from scipy.special import logit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase11_prediction as study
import phase2_energy as energy


def main():
    protocol=dict(created_utc=energy.now(),stage="trainable field gradients before material fitting",
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (Path(__file__),
            ROOT/"scripts/trainable_sandbox_20261017/phase11_prediction.py",ROOT/"src/sandbox/parent_allocation.py")},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (study.OUT/"numerical_gates.json",study.OUT/"zero_replays.json")})
    path=study.OUT/"learning_protocol.json"
    if path.exists():assert json.loads(path.read_text())["code_hashes"]==protocol["code_hashes"]
    else:energy.write(path,protocol)
    checks=[];rows=[]
    meta=study.feature_metadata(study.FAMILIES[:-1])
    for mode in ("parent_linear_joint","parent_rbf_joint","tf_rbf_joint","output_rbf_joint","parent_rbf_RT"):
        d=study.METHODS[mode];nbase=4 if d["training"]=="joint" else 2;k=4 if d["features"]=="linear" else 7
        w=np.r_[[logit((np.log10(1450)-study.prior.L0)/(study.prior.L1-study.prior.L0)),15.],
            [np.log10(.06),.5] if nbase==4 else [],np.linspace(-.15,.3,k)]
        for element,H,To in (("Sn",None,300.),("Si",1.8e17,150.)):
            r=SimpleNamespace(element=element,H_known=H is not None,H_cm3=H or 0.,
                incorporated_cm3=5.4e17 if H else 1e19,growth_T_C=800.,Tobs_K=To)
            a,J,_,_=study.point(r,w,mode,meta)
            for j in range(len(w)):
                h=1e-5;x=w.copy();y=w.copy();x[j]+=h;y[j]-=h
                fd=(study.point(r,x,mode,meta)[0]-study.point(r,y,mode,meta)[0])/(2*h)
                error=float(np.max(abs(fd-J[:,j])/(1+abs(fd))))
                rows.append(dict(mode=mode,element=element,Tobs=To,coordinate=j,scaled_error=error))
                checks.append(dict(name="whole_field_IFT",mode=mode,coordinate=j,error=error,passed=error<3e-5))
            zero=w.copy();zero[nbase:]=0
            z=study.point(r,zero,mode,meta)[0]
            baseline="joint_baseline" if nbase==4 else "thermal_RT"
            b=study.point(r,zero[:nbase],baseline,meta)[0]
            e=float(abs(z-b).max())
            checks.append(dict(name="zero_field_exact_baseline",mode=mode,element=element,error=e,passed=e<2e-8))
    x=np.linspace(-3,3,11);p=np.array([18.,.8,.5,7.])
    f,J=study.activation_curve(p,x)
    for j in range(4):
        h=1e-5;a=p.copy();b=p.copy();a[j]+=h;b[j]-=h
        fd=(study.activation_curve(a,x)[0]-study.activation_curve(b,x)[0])/(2*h)
        error=float(abs(fd-J[:,j]).max())
        checks.append(dict(name="strong_turnover_control_gradient",coordinate=j,error=error,passed=error<1e-7))
    # Every primary RT sample retains its full original loss weight in joint training.
    d=study.training_data(study.FAMILIES,"joint_baseline")
    rt=d[d.kind.eq("RT")];c=d[d.kind.eq("curve")]
    totals=rt.groupby("family").loss_weight.sum()
    checks.append(dict(name="main_task_weight_contract",RT_total=float(rt.loss_weight.sum()),curve_total=float(c.loss_weight.sum()),
        passed=np.allclose(totals,.2,atol=1e-13) and abs(c.loss_weight.sum()-.1)<1e-13))
    result=dict(passed=all(c["passed"] for c in checks),checks=checks,gradients=rows,
        maximum_field_gradient_scaled_error=max(r["scaled_error"] for r in rows))
    energy.write(study.OUT/"learning_gates.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),maximum_gradient=result["maximum_field_gradient_scaled_error"]),indent=2))
    assert result["passed"]


if __name__=="__main__":main()
