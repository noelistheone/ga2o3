"""GPU1 bridge gate; CPU physical roots with CUDA differentiable inputs."""
from pathlib import Path
import os, sys, json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
OUT=ROOT/"results/trainable_sandbox_20261017"

def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="1","Only the user-authorized physical GPU1"
    assert json.loads((OUT/"numerical_gate_cpu.json").read_text())["passed"]
    import torch
    torch.set_num_threads(1)
    from sandbox.kroger_db import load
    from sandbox.implicit_defects import ImplicitDefectLayer,energy_basis
    db=load();layer=ImplicitDefectLayer(db,("Si",),energy_basis(db))
    p0=layer.parameters();r=layer.predict(p0)
    p=torch.tensor(p0,dtype=torch.float64,device="cuda:0",requires_grad=True)
    coefficients=torch.tensor([1.,.3,.1],dtype=torch.float64,device="cuda:0")
    y=layer.tensor(p);(y@coefficients).backward()
    expected=np.array([1.,.3,.1])@r.jacobian
    error=float(np.max(np.abs(p.grad.detach().cpu().numpy()-expected)))
    w=torch.nn.Parameter(torch.zeros((),dtype=torch.float64,device="cuda:0"))
    opt=torch.optim.LBFGS([w],lr=.5,max_iter=25,line_search_fn="strong_wolfe",
                          tolerance_grad=1e-10,tolerance_change=1e-13)
    templates=[layer.parameters(doses=(d,),weights=[.12]) for d in (1e17,1e19)]
    targets=[layer.predict(v).values[0] for v in templates]
    trace=[]
    def closure():
        opt.zero_grad()
        loss=torch.zeros((),dtype=torch.float64,device="cuda:0")
        for template,target in zip(templates,targets):
            p=torch.cat([torch.tensor(template[:2],dtype=torch.float64,device="cuda:0"),
                         w.reshape(1),torch.tensor(template[3:],dtype=torch.float64,device="cuda:0")])
            loss=loss+(layer.tensor(p)[0]-target)**2/2
        loss.backward();trace.append(dict(loss=float(loss.detach().cpu()),weight_eV=float(w.detach().cpu())))
        return loss
    opt.step(closure)
    result=dict(CUDA_VISIBLE_DEVICES="1",visible_device="cuda:0",gpu_name=torch.cuda.get_device_name(),
        gradient_max_abs_error=error,recovered_virtual_energy_eV=float(w.detach().cpu()),
        true_virtual_energy_eV=.12,initial_loss=trace[0]["loss"],final_loss=trace[-1]["loss"],
        root_location="CPU float64",parameter_and_backward_location="physical GPU1 float64",
        scope="device/optimizer bridge correctness on virtual labels; not new material training results",
        trace=trace)
    result["passed"]=error<1e-10 and abs(result["recovered_virtual_energy_eV"]-.12)<1e-3
    (OUT/"gpu1_bridge_gate.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="trace"},indent=2))
    assert result["passed"]

if __name__=="__main__":main()
