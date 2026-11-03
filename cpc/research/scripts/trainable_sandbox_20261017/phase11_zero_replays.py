"""Replay real saved states after adding the zero parent-bias coordinate."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.parent_allocation import ParentAllocationLayer
from sandbox.kroger_db import load
import phase9_joint_prediction as prior
import phase11_allocation_gates as gates
import phase2_energy as energy


def main():
    rows=[];layers={};fits=[json.loads(p.read_text()) for p in (prior.OUT/"fits").glob("*.json")]
    for method,extension in (("no_extension_RT",False),("learned_screen_joint",True)):
        for held in sorted(prior.DATA.family.unique()):
            f=next(f for f in fits if f["method"]==method and set(f["train_families"])==set(prior.DATA.family)-{held})
            w=f["parameters"];T=10**(prior.L0+(prior.L1-prior.L0)*expit(w[0]))
            data=prior.PRED[prior.PRED.family.eq(held)&prior.PRED.kind.eq("RT")]
            extra=[]
            for sample in prior.SPECS[prior.SPECS.family.eq(held)].index:
                d=prior.PRED[prior.PRED.sample_id.eq(sample)&prior.PRED.kind.eq("curve")]
                extra.extend([d.loc[d.Tobs_K.idxmin()],d.loc[abs(d.Tobs_K-150).idxmin()]])
            if extra:
                import pandas as pd
                data=pd.concat([data,pd.DataFrame(extra)],ignore_index=True)
            saved={r["obs_id"]:r["predicted"] for r in f["predictions"]}
            for r in data.itertuples():
                elements=(r.element,"H") if r.H_known else (r.element,)
                key=(elements,extension)
                if key not in layers:layers[key]=ParentAllocationLayer(load(),elements,extension=extension)
                doses=(r.incorporated_cm3,r.H_cm3) if r.H_known else (r.incorporated_cm3,)
                p,J,_=layers[key].point(Tf_K=T,background_cm3=10**w[1],doses=doses,Tobs_K=r.Tobs_K,
                    E0_eV=10**w[2] if extension else .0366,s=w[3] if extension else 0.,bias=0.)
                error=float(abs(p.values[0]-saved[r.obs_id]))
                rows.append(dict(method=method,obs_id=r.obs_id,error_dex=error))
    result=dict(passed=len(rows)==86 and max(r["error_dex"] for r in rows)<2e-8,
        checks=rows,maximum_error_dex=max(r["error_dex"] for r in rows),
        code_sha256=energy.sha(Path(__file__)),core_sha256=energy.sha(ROOT/"src/sandbox/parent_allocation.py"))
    energy.write(gates.OUT/"zero_replays.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="checks"},indent=2));assert result["passed"]


if __name__=="__main__":main()
