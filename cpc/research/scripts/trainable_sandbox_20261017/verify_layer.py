"""Numerical gate: unchanged reference equations, finite differences and autograd.

Virtual states test implementation and numerical correctness, not material
accuracy. All attempted cases and errors are saved before the gate is asserted.
"""
from pathlib import Path
from dataclasses import replace
import sys, json, time, argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from sandbox import kroger_db,engine,transport
from sandbox.implicit_defects import ImplicitDefectLayer,energy_basis,ImplicitDomainError
OUT=ROOT/"results/trainable_sandbox_20261017"

def reference(db,elems,pars,basis):
    B=basis.shape[1]
    T,Nd=10.**pars[:2]
    doses=10.**pars[2+B:2+B+len(elems)]
    d=replace(db,dHo=db.dHo+basis@pars[2:2+B])
    eq=engine.solve_single(d,T,pO2=10.**pars[-1],fixed_conc=dict(zip(elems,doses)),fd=True,Nd_bg=Nd)
    q=engine.quench(d,eq,T_quench=300.,fd=True,Nd_bg=Nd)
    rh=float(transport.hall_factor(300.,q.n))
    NI=float(np.abs(db.charge)@q.N_cs+Nd)
    mu=float(transport.mu_caughey_thomas(NI,300.))
    return np.array([np.log10(q.n/rh),np.log10(mu),np.log10(transport.E*q.n/rh*mu)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--torch-device",default="cpu")
    args=ap.parse_args()
    db=kroger_db.load()
    cases=[
        dict(name="Si_lowT_lowdose",elems=("Si",),T=800.,doses=(1e16,),Nd=1e15,w=.0),
        dict(name="Si_reference",elems=("Si",),T=1325.,doses=(1e18,),Nd=9.1e15,w=.0),
        dict(name="Si_hot_highdose",elems=("Si",),T=1700.,doses=(1e20,),Nd=1e16,w=.1),
        dict(name="Sn_lowT",elems=("Sn",),T=1000.,doses=(1e18,),Nd=1e15,w=-.1),
        dict(name="Sn_highdose",elems=("Sn",),T=1450.,doses=(1e19,),Nd=1e16,w=.15),
        dict(name="Mg_electron_Hall_invalid",elems=("Mg",),T=1325.,doses=(1e17,),Nd=1e15,w=.0),
        dict(name="Si_Sn_joint",elems=("Si","Sn"),T=1200.,doses=(1e17,2e17),Nd=1e16,w=.0),
        dict(name="Si_Mg_joint",elems=("Si","Mg"),T=1400.,doses=(1e18,5e17),Nd=1e16,w=.0),
        dict(name="Si_relative_charge_energy",elems=("Si",),T=1300.,doses=(1e18,),Nd=1e16,w=.1,basis="negative_charge_states"),
    ]
    references=[];differences=[];failures=[]
    for c in cases:
        basis=energy_basis(db,c.get("basis","native_Ga_vacancy"))
        layer=ImplicitDefectLayer(db,c["elems"],basis)
        p=layer.parameters(T=c["T"],Nd=c["Nd"],weights=[c["w"]],doses=c["doses"])
        try:
            t=time.perf_counter();pred=layer.predict(p);fast=time.perf_counter()-t
            t=time.perf_counter();ref=reference(db,c["elems"],p,basis);slow=time.perf_counter()-t
            references.append(dict(case=c["name"],reference_max_abs_dex=float(np.max(np.abs(ref-pred.values))),
                implicit_seconds=fast,reference_seconds=slow,**pred.diagnostics))
            for h in [1e-3,1e-4,1e-5,1e-6]:
                for j,name in enumerate(pred.parameter_names):
                    a=p.copy();b=p.copy();a[j]+=h;b[j]-=h
                    fd=(layer.predict(a).values-layer.predict(b).values)/(2*h)
                    for k,out in enumerate(pred.names):
                        exact=pred.jacobian[k,j]
                        differences.append(dict(case=c["name"],parameter=name,output=out,step=h,
                            implicit_derivative=exact,central_derivative=fd[k],absolute_error=abs(fd[k]-exact),
                            scaled_error=abs(fd[k]-exact)/(1+abs(exact))))
        except Exception as e:
            failures.append(dict(case=c["name"],error=repr(e)))
        pd.DataFrame(references).to_csv(OUT/"reference_gate.csv",index=False)
        pd.DataFrame(differences).to_csv(OUT/"gradient_gate.csv",index=False)
        (OUT/"numerical_failures.json").write_text(json.dumps(failures,indent=2)+"\n")
        print(c["name"],"ok" if not failures or failures[-1]["case"]!=c["name"] else failures[-1]["error"],flush=True)
    device=args.torch_device
    import torch
    torch.set_num_threads(1)
    layer=ImplicitDefectLayer(db,("Si",),energy_basis(db))
    p0=layer.parameters()
    p=torch.tensor(p0,dtype=torch.float64,device=device,requires_grad=True)
    gradcheck=torch.autograd.gradcheck(layer.tensor,(p,),eps=1e-5,atol=3e-5,rtol=3e-5)
    # A known shared correction creates two virtual targets. Training must flow
    # through the complete hot/quench root; this is an implementation gate.
    target_pars=[layer.parameters(doses=(d,),weights=[.12]) for d in [1e17,1e19]]
    targets=[layer.predict(v).values[0] for v in target_pars]
    weight=torch.nn.Parameter(torch.zeros((),dtype=torch.float64,device=device))
    opt=torch.optim.LBFGS([weight],lr=.5,max_iter=25,line_search_fn="strong_wolfe",
                          tolerance_grad=1e-10,tolerance_change=1e-13)
    trace=[]
    def closure():
        opt.zero_grad()
        loss=torch.zeros((),dtype=torch.float64,device=device)
        for template,target in zip(target_pars,targets):
            parts=[torch.tensor(template[:2],dtype=torch.float64,device=device),weight.reshape(1),
                   torch.tensor(template[3:],dtype=torch.float64,device=device)]
            y=layer.tensor(torch.cat(parts))[0]
            loss=loss+(y-target)**2/len(targets)
        loss.backward()
        trace.append(dict(loss=float(loss.detach().cpu()),weight_eV=float(weight.detach().cpu())))
        return loss
    opt.step(closure)
    pd.DataFrame(trace).to_csv(OUT/f"autograd_training_{device.replace(':','_')}.csv",index=False)
    d=pd.DataFrame(differences)
    g=d[d.step.eq(1e-5)]
    verdict=dict(reference_cases=len(references),attempted_cases=len(cases),failures=failures,
        reference_max_abs_dex=max((r["reference_max_abs_dex"] for r in references),default=float("inf")),
        gradient_max_scaled_error_h1e5=float(g.scaled_error.max()) if len(g) else None,
        gradient_max_abs_error_h1e5=float(g.absolute_error.max()) if len(g) else None,
        torch_gradcheck=bool(gradcheck),torch_device=device,
        recovered_virtual_weight_eV=float(weight.detach().cpu()),true_virtual_weight_eV=.12,
        initial_virtual_loss=trace[0]["loss"],final_virtual_loss=trace[-1]["loss"],
        scope="numerical and training implementation gates, not material validation")
    if device.startswith("cuda"):
        import os
        verdict.update(CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),gpu_name=torch.cuda.get_device_name())
    verdict["passed"]=(not failures and verdict["reference_max_abs_dex"]<2e-6
        and verdict["gradient_max_scaled_error_h1e5"]<3e-5 and gradcheck
        and abs(verdict["recovered_virtual_weight_eV"]-.12)<1e-3)
    (OUT/f"numerical_gate_{device.replace(':','_')}.json").write_text(json.dumps(verdict,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(verdict,ensure_ascii=False,indent=2))
    assert verdict["passed"],"Numerical gate failed; keep evidence and fix before material fitting"

if __name__=="__main__":
    main()
