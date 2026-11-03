"""Numerical and real-state replay gates before local-adaptation evaluation."""
from pathlib import Path
import os,sys,json
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.parameter_adaptation import regularized_adapt,local_information
import phase10_internal_adaptation as study
import phase2_energy as energy


def main():
    study.freeze();checks=[];replays=[];derivatives=[]
    J=np.array([[1.,.2],[-.3,2.],[.7,-.5]])
    theta=np.array([.1,-.1]);scales=np.array([.3,.6]);y=np.array([.5,-.8,.2])
    def linear(t):return J@t,J
    for lam in (.001,.01,.1,1.,10.):
        f=regularized_adapt(linear,y,theta,scales,[-10,-10],[10,10],lam)
        A=J*scales;z=np.linalg.solve(A.T@A+lam*np.eye(2),A.T@(y-J@theta))
        error=float(abs(f.standardized_shift-z).max())
        checks.append(dict(name="linear_closed_form",lam=lam,error=error,passed=error<1e-9))
    one=local_information(np.array([[1.,2.]]),[1.,1.],.1)
    checks.append(dict(name="one_anchor_two_coordinate_null",passed=one["rank"]==1 and
        abs(one["singular_values"][-1])==0 and abs(one["data_resolution_eigenvalues"][-1])==0))
    bounded=regularized_adapt(lambda t:(t.copy(),np.ones((1,1))),[10.],[0.],[1.],[-1.],[1.],.001)
    checks.append(dict(name="active_bound",passed=abs(bounded.parameters[0]-1)<1e-7))
    def never(t):raise AssertionError("Callback should not be evaluated")
    for obs,lam in (([],.1),([1.],np.inf)):
        f=regularized_adapt(never,obs,[.1],[1.],[-1.],[1.],lam)
        checks.append(dict(name="no_update_without_information",passed=f.status=="base_no_update" and f.parameters[0]==.1))
    def broken(t):raise RuntimeError("declared_failure_probe")
    f=regularized_adapt(broken,[1.],[.1],[1.],[-1.],[1.],.1)
    checks.append(dict(name="failure_explicit_fallback",passed=f.status=="base_fallback" and len(f.diagnostics["state_failures"])==1))
    for method in ("no_extension_RT","fixed_binding_joint","learned_screen_joint"):
        for held in study.FAMILIES:
            path,base=study.base_fit([s for s in study.FAMILIES if s!=held],method)
            rows=study.PRED[study.PRED.family.eq(held)&study.PRED.kind.eq("RT")]
            extra=[]
            for sample in study.prior.SPECS[study.prior.SPECS.family.eq(held)].index:
                c=study.PRED[study.PRED.sample_id.eq(sample)&study.PRED.kind.eq("curve")]
                extra.extend([c.loc[c.Tobs_K.idxmin()],c.loc[abs(c.Tobs_K-150).idxmin()]])
            if extra:
                import pandas as pd
                rows=pd.concat([rows,pd.DataFrame(extra)],ignore_index=True)
            raw={r["obs_id"]:r["predicted"] for r in base["predictions"]}
            for row in rows.itertuples():
                theta=study.initial_coordinates(base);p,j=study.physical_point(row,base,theta)
                error=float(abs(p.values[0]-raw[row.obs_id]))
                replays.append(dict(method=method,obs_id=row.obs_id,error=error))
    checks.append(dict(name="129actual_zero_update_replays",count=len(replays),
        error=max(r["error"] for r in replays),passed=len(replays)==129 and max(r["error"] for r in replays)<2e-8))
    # Real Si/H and Si-only roots at both original/near150K conditions.
    for sample in ("10.1007/s40843-022-2167-x::C","10.1063/5.0101132::3"):
        held=sample.split("::")[0]
        _,base=study.base_fit([s for s in study.FAMILIES if s!=held],"learned_screen_joint")
        rows=study.PRED[study.PRED.sample_id.eq(sample)]
        for index in (rows[rows.kind.eq("RT")].index[0],abs(rows.Tobs_K-150).idxmin()):
            r=next(rows.loc[[index]].itertuples());theta=study.initial_coordinates(base)
            v,J=study.physical_point(r,base,theta)
            for j in (0,1):
                h=1e-5;a=theta.copy();b=theta.copy();a[j]+=h;b[j]-=h
                fd=(study.physical_point(r,base,a)[0].values-study.physical_point(r,base,b)[0].values)/(2*h)
                error=float(np.max(abs(fd-J[:,j])/(1+abs(fd))))
                derivatives.append(dict(sample_id=sample,obs_id=r.obs_id,coordinate=j,scaled_error=error))
                checks.append(dict(name="physical_coordinate_IFT",sample_id=sample,coordinate=j,
                    error=error,passed=error<3e-5))
    result=dict(passed=all(c["passed"] for c in checks),checks=checks,replays=replays,derivatives=derivatives,
        maximum_replay_error=max(r["error"] for r in replays),
        maximum_gradient_scaled_error=max(r["scaled_error"] for r in derivatives),
        interpretation="implementation gates,not material accuracy or microscopic identification")
    energy.write(study.OUT/"numerical_gates.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("checks","replays","derivatives")},indent=2))
    assert result["passed"]


if __name__=="__main__":main()
