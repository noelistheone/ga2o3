"""Independent thermodynamic and full-state gates before phase13 fitting."""
from pathlib import Path
import os,sys,json,argparse,shutil
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.special import expit,xlogy
from sandbox.donor_mixture import DonorMixtureLayer
from sandbox.temperature_defects import TemperatureDefectLayer
from sandbox.material_profile import PUBLISHED_PROFILE,REPOSITORY_PROFILE
from sandbox.implicit_defects import energy_basis
from sandbox.kroger_db import load,NPZ
from sandbox.thermo import KB_EV
import phase2_energy as energy

OUT=energy.BASE/"phase13_donor_mixture"


def run():
    db=load();checks=[];states=[];derivatives=[]
    def add(name,passed,**details):checks.append(dict(name=name,passed=bool(passed),**details))
    # Independent microscopic entropy, energy derivatives, and charge moments.
    for T in (40.,300.,1400.):
        for delta in (-.1,.01):
            for ell in (0.,.4,1.):
                beta=1/(KB_EV*T);Ec=4.;ED=.0366;ef=Ec+delta
                def free(e):
                    gp=.3+e;g0=.3+Ec-ED
                    f0=float(expit(np.log(2)+beta*(e-Ec+ED)));fp=float(expit(-np.log(2)-beta*(e-Ec+ED)))
                    al=fp*gp+f0*g0+KB_EV*T*(xlogy(fp,fp)+xlogy(f0,f0/2))
                    return (1-ell)*gp+ell*al
                f=float(expit(np.log(2)+beta*(ef-Ec+ED)));fp=float(expit(-np.log(2)-beta*(ef-Ec+ED)))
                q=(1-ell)+ell*fp;chi=ell*f*fp
                analytic=.3+ef-ell*KB_EV*T*np.logaddexp(0,np.log(2)+beta*(ef-Ec+ED))
                h=1e-6;dq=(free(ef+h)-free(ef-h))/(2*h)
                # Fix an arbitrary atom reservoir normalization; only response matters.
                def cq(e):
                    f0=expit(np.log(2)+beta*(e-Ec+ED));q0=(1-ell)+ell*expit(-np.log(2)-beta*(e-Ec+ED))
                    c=np.exp(-(free(e)-free(ef))*beta)
                    return c*q0
                hot_deriv=(cq(ef+h)-cq(ef-h))/(2*h)
                err=max(abs(analytic-free(ef)),abs(dq-q),abs(hot_deriv+beta*(q*q+chi))/(1+abs(hot_deriv)))
                add("independent_freeenergy_charge_response",err<3e-6,T=T,ell=ell,delta=delta,error=err)
                add("actual_secondmoment_differs_from_conditional_response",
                    ell*f*(1-ell*f)>=chi-1e-15,T=T,ell=ell,actual_q_squared=q,conditional_susceptibility=chi)
    for prof in (REPOSITORY_PROFILE,PUBLISHED_PROFILE):
        for element in ("Si","Sn"):
            for ell in (0.,.4,1.,None):
                layer=DonorMixtureLayer(db,(element,),profile=prof,hall="rH1",fixed_fraction=ell)
                old=None if ell not in (0.,1.) else TemperatureDefectLayer(db,(element,),profile=prof,hall="rH1",donor_extension=ell==1.)
                for To in (40.,150.,300.,1400.):
                    p=layer.parameters(T=1400,Nd=1e16,doses=(1e18,),Tobs=To,localized_logit=.7,localized_slope=1.2)
                    a=layer.predict(p);s=layer.last_state;dg=a.diagnostics
                    atom_hot=layer.db.dm.T@s["hot_charge"];atom_cold=layer.db.dm.T@s["cold_charge"]
                    ae=float(abs(atom_hot-atom_cold).max()/max(abs(atom_hot).max(),1))
                    pc=np.bincount(layer.parent,weights=s["cold_charge"],minlength=layer.db.nD)
                    pe=float(abs(pc-s["totals"]).sum()/max(s["totals"].sum(),1))
                    add("mixture_stock_charge_probability",ae<1e-11 and pe<1e-11 and dg["parent_probability_error"]<1e-12
                        and dg["charge_relative_residual"]<1e-8 and dg["cold_Q_EF_cm3_per_eV"]<0,
                        profile=prof.name,element=element,ell=ell,Tobs=To,atom_error=ae,parent_error=pe)
                    H=s["hot_mixture"]["raw_minus_charge_atom_jacobian"];scale=max(abs(H).max(),1)
                    symmetry=float(abs(H-H.T).max()/scale);ev=np.linalg.eigvalsh((H+H.T)/(2*scale))
                    add("hot_thermodynamic_Hessian",symmetry<1e-11 and ev.min()>0,
                        profile=prof.name,element=element,ell=ell,Tobs=To,symmetry_error=symmetry,eigenvalues=ev)
                    if To==1400:
                        e=float(abs(s["ef_cold"]-s["hot"][0]));pop=float(abs(s["cold_charge"]-s["hot_charge"]).sum()/s["hot_charge"].sum())
                        add("same_temperature_noquench",e<1e-9 and pop<1e-8,EF_error=e,population_error=pop,ell=ell,element=element,profile=prof.name)
                    if old is not None:
                        before=old.predict(old.parameters(T=1400,Nd=1e16,doses=(1e18,),Tobs=To))
                        err=float(abs(a.values-before.values).max());je=float(abs(a.jacobian[:,:old.np]-before.jacobian).max())
                        nh=old.db.nCS;stateerr=float(abs(s["hot_charge"][:nh]-old.last_state["hot_charge"]).sum()/max(sum(old.last_state["hot_charge"]),1))
                        add("exact_old_endpoint",err<2e-8 and je<3e-6 and stateerr<1e-8,
                            ell=ell,element=element,profile=prof.name,Tobs=To,value_error=err,jacobian_error=je,hot_population_error=stateerr)
                    states.append(dict(profile=prof.name,element=element,ell=ell,Tobs=To,values=a.values,diagnostics=dg))
                add("measurement_temperature_hot_cache",layer.hot_solve_calls==1 and layer.hot_cache_hits==3,
                    profile=prof.name,element=element,ell=ell,calls=layer.hot_solve_calls,hits=layer.hot_cache_hits)
    # Each actual parameter, joint H supply, both bands, Si and Sn, hot and cold.
    for prof in (REPOSITORY_PROFILE,PUBLISHED_PROFILE):
        for elems,doses in ((("Si",),(1e18,)),(("Sn",),(3e18,)),(("Si","H"),(5.4e17,1.8e17))):
            layer=DonorMixtureLayer(db,elems,profile=prof,hall="rH1",basis=energy_basis(db,"negative_charge_states"))
            for To in (40.,300.):
                p=layer.parameters(T=1450,Nd=1e16,doses=doses,Tobs=To,weights=(.03,),binding_eV=.04,screen_s=.25,
                    localized_logit=.7,localized_slope=1.2)
                a=layer.predict(p)
                for j,name in enumerate(layer.parameter_names):
                    h=1e-5;x=p.copy();y=p.copy();x[j]+=h;y[j]-=h
                    fd=(layer.predict(x).values-layer.predict(y).values)/(2*h)
                    error=float(np.max(abs(fd-a.jacobian[:,j])/(1+abs(fd))))
                    derivatives.append(dict(profile=prof.name,elements=elems,Tobs=To,parameter=name,
                        finite_difference=fd,analytic=a.jacobian[:,j],scaled_error=error))
                    add("complete_mixture_IFT",error<3e-5,profile=prof.name,elements=elems,Tobs=To,parameter=name,error=error)
    layer=DonorMixtureLayer(db,("Si",),profile=PUBLISHED_PROFILE,hall="rH1")
    p=layer.parameters(Tobs=150)
    high=p.copy();high[layer.idose]=24.;ell,_=layer.fraction(high)
    add("high_input_concentration_fullionized_limit",ell<1e-4,localized_fraction=ell,
        scope="fraction/conditionalcharge bound only,not dilute modelstate at1e24cm-3")
    try:layer.fixed_fraction=.5
    except AttributeError:locked=True
    else:locked=False
    add("fixed_fraction_immutable",locked)
    # Existing autograd mechanism applies the complete saved IFT, no unrolling.
    import torch
    t=torch.tensor(p,dtype=torch.float64,requires_grad=True);a=layer.predict(p)
    out=layer.tensor(t);weight=torch.tensor([.3,.2,.1],dtype=torch.float64);(weight@out).backward()
    error=float(abs(t.grad.numpy()-weight.numpy()@a.jacobian).max())
    add("complete_firstorder_autograd_CPU",error<1e-9,error=error)
    return dict(passed=all(c["passed"] for c in checks),checks=checks,states=states,gradients=derivatives,
        maximum_gradient_scaled_error=max(r["scaled_error"] for r in derivatives),
        scope="thermodynamic/numerical gates;no new material prediction training or microscopic mechanism validation")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--freeze",action="store_true");a=ap.parse_args()
    files=(Path(__file__),ROOT/"src/sandbox/donor_mixture.py",ROOT/"src/sandbox/temperature_defects.py",
        ROOT/"src/sandbox/material_profile.py",ROOT/"src/sandbox/implicit_defects.py",ROOT/"docs/ML_SANDBOX_DONOR_MIXTURE_THEORY_20261017.md")
    protocol=dict(created_utc=energy.now(),code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in files},
        input_hashes={str(NPZ.relative_to(ROOT)):energy.sha(NPZ)},
        theoretical_scope="fixed supplied-C subgroups;common effective freeenergy;notMottHamiltonian or microscopic truth",
        gate_tolerances="IFT3e-5;endpoint2e-8values/3e-6jac;prob1e-12;noquench1e-8;PSD/symmetry")
    frozen=OUT/"numerical_protocol.json"
    if frozen.exists():
        old=json.loads(frozen.read_text())
        for key in ("code_hashes","input_hashes"):assert old[key]==protocol[key],key
    runpath=OUT/"development_runs"/energy.now().replace(":","-")
    for path in files:
        dest=runpath/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    try:result=run()
    except Exception as exc:
        energy.write(runpath/"fatal.json",dict(error=repr(exc),protocol=protocol));raise
    energy.write(runpath/"gates.json",result);energy.write(runpath/"protocol.json",protocol)
    brief=dict(passed=result["passed"],checks=len(result["checks"]),gradient_error=result["maximum_gradient_scaled_error"],
        failed=[c for c in result["checks"] if not c["passed"]],development_run=str(runpath.relative_to(ROOT)))
    print(json.dumps(brief,ensure_ascii=False,indent=2))
    if a.freeze:
        if not result["passed"]:raise RuntimeError("Failed mixture gate; material fitting prohibited")
        energy.write(OUT/"numerical_gates.json",result)
        if not frozen.exists():energy.write(frozen,protocol)
        print("Numerical code/input protocol frozen before property fitting")
    if not result["passed"]:raise RuntimeError("Mixture gates failed")


if __name__=="__main__":main()
