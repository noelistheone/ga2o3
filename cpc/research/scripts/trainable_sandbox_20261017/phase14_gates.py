"""Gate the actual combined E(C)/fraction maps before any new fitting."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import phase14_prediction as study
import phase2_energy as energy


def main():
    checks=[];derivatives=[]
    def add(name,good,**kw):checks.append(dict(name=name,passed=bool(good),**kw))
    for method in study.NEW_METHODS:
        model=study.core.mapper(method);lo,hi,starts=study.core.specification(method)
        add("expected_parameter_capacity",len(starts[0])==(5 if method=="fraction_screen_constant_joint" else 6),method=method,names=model.parameter_names)
        for To in (40.,150.,300.):
            for element,C,H in (("Si",1e18,0.),("Sn",3e18,0.),("Si",5.4e17,1.8e17)):
                row=SimpleNamespace(element=element,incorporated_cm3=C,growth_T_C=830.,Tobs_K=To,H_known=H>0,H_cm3=H)
                w=np.array(starts[0]);a=model.point(row,w)
                for j,name in enumerate(model.parameter_names):
                    h=1e-5;x=w.copy();y=w.copy();x[j]+=h;y[j]-=h
                    fd=(model.point(row,x).values-model.point(row,y).values)/(2*h)
                    err=float(max(abs(fd-a.jacobian[:,j])/(1+abs(fd))))
                    derivatives.append(err);add("combined_actual_parameter_chain",err<3e-5,method=method,element=element,T=To,parameter=name,error=err)
                # Independent raw input-concentration derivative includes both learned relations.
                layer=model.layers[(element,"H") if H else (element,)]
                p=layer.parameters(T=a.diagnostics["Tf_K"],Nd=10**w[1],weights=[0.],pO2=1e-5,
                    doses=(C,H) if H else (C,),Tobs=To,binding_eV=10**w[model.iE],screen_s=w[model.iscreen],
                    localized_logit=w[model.ia] if model.ia is not None else 0.,localized_slope=w[model.ib] if model.ib is not None else 0.)
                raw=layer.predict(p);wanted=raw.jacobian[:,layer.idose].copy()
                if model.iout is not None:wanted[0]-=w[model.iout+1]
                x=SimpleNamespace(**vars(row));y=SimpleNamespace(**vars(row));x.incorporated_cm3*=10**1e-5;y.incorporated_cm3*=10**-1e-5
                fd=(model.point(x,w).values-model.point(y,w).values)/(2e-5)
                err=float(max(abs(fd-wanted)/(1+abs(fd))));derivatives.append(err)
                add("both_relations_chemical_input_chain",err<3e-5,method=method,element=element,T=To,error=err)
                kinds=("stock","Tf","offset") if model.ia is None else ("stock","Tf","localized","offset")
                for kind in kinds:
                    z=.2;h=1e-5;a=model.point(row,w,source_kind=kind,source_shift=z)
                    fd=(model.point(row,w,source_kind=kind,source_shift=z+h).values-model.point(row,w,source_kind=kind,source_shift=z-h).values)/(2*h)
                    err=float(max(abs(fd-a.source_jacobian)/(1+abs(fd))));derivatives.append(err)
                    add("combined_source_coordinate",err<3e-5,method=method,kind=kind,element=element,T=To,error=err)
                    zero=model.point(row,w,source_kind=kind,source_shift=0.)
                    add("combined_exact_zero_update",np.array_equal(zero.values,model.point(row,w).values),method=method,kind=kind,element=element,T=To)
    files=(Path(__file__),ROOT/"scripts/trainable_sandbox_20261017/phase14_prediction.py",ROOT/"src/sandbox/donor_mixture_model.py",
        ROOT/"src/sandbox/donor_mixture.py",ROOT/"docs/ML_SANDBOX_JOINT_STATISTICS_20261017.md")
    result=dict(passed=all(c["passed"] for c in checks),created_utc=energy.now(),checks=checks,
        maximum_gradient_error=max(derivatives),code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in files},
        scope="combined map/input/source chain numerical gates,not newmaterialaccuracy")
    energy.write(study.OUT/"model_gates.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),maximum_gradient_error=result["maximum_gradient_error"],
        failed=[c for c in checks if not c["passed"]]),indent=2))
    if not result["passed"]:raise RuntimeError("Combined map gate failed;training prohibited")


if __name__=="__main__":main()
