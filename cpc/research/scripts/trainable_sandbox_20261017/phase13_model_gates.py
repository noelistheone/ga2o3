"""Verify the actual trainable map and source-coordinate chain rules."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import phase13_prediction as study
import phase2_energy as energy


def main():
    checks=[]
    assert json.loads((study.OUT/"numerical_gates.json").read_text())["passed"]
    for method,d in study.METHODS.items():
        if "empirical" in d:continue
        model=study.mapper(method);_,_,starts=study.specification(method);w=np.array(starts[0])
        rows=(SimpleNamespace(element="Si",incorporated_cm3=1e18,growth_T_C=830.,Tobs_K=40.,H_known=False,H_cm3=0.),
            SimpleNamespace(element="Sn",incorporated_cm3=3e18,growth_T_C=700.,Tobs_K=300.,H_known=False,H_cm3=0.),
            SimpleNamespace(element="Si",incorporated_cm3=5.4e17,growth_T_C=830.,Tobs_K=150.,H_known=True,H_cm3=1.8e17))
        for row in rows:
            a=model.point(row,w)
            for j,name in enumerate(model.parameter_names):
                h=1e-5;x=w.copy();y=w.copy();x[j]+=h;y[j]-=h
                fd=(model.point(row,x).values-model.point(row,y).values)/(2*h)
                err=float(np.max(abs(fd-a.jacobian[:,j])/(1+abs(fd))))
                checks.append(dict(name="trained_model_full_chain",method=method,element=row.element,T=row.Tobs_K,
                    parameter=name,error=err,passed=err<3e-5))
            for kind in (("stock","Tf","localized","offset") if model.ia is not None else ("stock","Tf","offset")):
                z=.2;h=1e-5;a=model.point(row,w,source_kind=kind,source_shift=z)
                fd=(model.point(row,w,source_kind=kind,source_shift=z+h).values-
                    model.point(row,w,source_kind=kind,source_shift=z-h).values)/(2*h)
                err=float(np.max(abs(fd-a.source_jacobian)/(1+abs(fd))))
                checks.append(dict(name="source_coordinate",kind=kind,method=method,element=row.element,error=err,passed=err<3e-5))
                zero=model.point(row,w,source_kind=kind,source_shift=0)
                checks.append(dict(name="source_zero_exact",kind=kind,method=method,
                    error=float(abs(zero.values-model.point(row,w).values).max()),
                    passed=bool(np.array_equal(zero.values,model.point(row,w).values))))
    result=dict(passed=all(c["passed"] for c in checks),checks=checks,
        maximum_error=max(c["error"] for c in checks),map_module_reads_property_tables=False,
        independent_runtime_file_access_guard_tested=False,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (Path(__file__),
            ROOT/"scripts/trainable_sandbox_20261017/phase13_prediction.py",ROOT/"src/sandbox/donor_mixture_model.py")},
        scope="numerical/explicit-input map gates,not material accuracy")
    energy.write(study.OUT/"model_gates.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),max_error=result["maximum_error"],
        failed=[c for c in checks if not c["passed"]]),indent=2))
    if not result["passed"]:raise RuntimeError("Trainable map failed:trainingprohibited")


if __name__=="__main__":main()
