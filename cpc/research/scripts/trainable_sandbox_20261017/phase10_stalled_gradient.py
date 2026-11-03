"""Targeted derivative checks at the actual two stalled calibration states."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import phase10_internal_adaptation as study
import phase2_energy as energy


def main():
    out=study.OUT/"solver_budget_sensitivity";results=[]
    for path in (out/"refined").glob("*.json"):
        c=json.loads(path.read_text())
        if c["status"]!="adapted_nonconverged":continue
        parent=json.loads(next((study.OUT/"calibrations").glob("*/"+c["parent_key"]+".json")).read_text())
        base=json.loads((ROOT/c["base_path"]).read_text())
        data=study.episode(c["task"],c["context"],c["budget"])
        theta=np.array(c["parameters"]);D=np.array([.05,c["config"]["E_scale"]])
        diffs=[];values=[];js=[]
        for row in data[data.role.eq("anchor")].itertuples():
            p,J=study.physical_point(row,base,theta);values.append(p.values[0]);js.append(J[0])
            for h in (1e-3,1e-4,1e-5,1e-6,1e-7):
                for j in (0,1):
                    a=theta.copy();b=theta.copy();a[j]+=h;b[j]-=h
                    fd=(study.physical_point(row,base,a)[0].values[0]-study.physical_point(row,base,b)[0].values[0])/(2*h)
                    diffs.append(dict(obs_id=row.obs_id,h=h,coordinate=j,IFT=float(J[0,j]),finite_difference=float(fd),scaled_error=float(abs(fd-J[0,j])/(1+abs(fd)))))
        r=np.asarray(values)-data[data.role.eq("anchor")].target.to_numpy(float)
        z=(theta-study.initial_coordinates(base))/D
        grad=(np.asarray(js)*D).T@r+c["lam"]*z
        results.append(dict(parent_key=c["parent_key"],theta=theta,standardized_gradient=grad,
            objective=float(r@r+c["lam"]*(z@z)),checks=diffs,
            maximum_fine_step_error=max(v["scaled_error"] for v in diffs if v["h"] in (1e-5,1e-6))))
    energy.write(out/"stalled_gradient_checks.json",dict(code_sha256=energy.sha(Path(__file__)),results=results))
    print(json.dumps([{k:v for k,v in energy.clean(r).items() if k!="checks"} for r in results],indent=2))


if __name__=="__main__":main()
