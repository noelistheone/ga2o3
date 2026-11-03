"""Full catalogue temperature-layer gates, before any new material fitting."""
from pathlib import Path
from dataclasses import FrozenInstanceError
import os,sys,json
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.special import expit
from sandbox import thermo
from sandbox.kroger_db import load
from sandbox.material_profile import MaterialProfile,REPOSITORY_PROFILE,PUBLISHED_PROFILE
from sandbox.temperature_defects import TemperatureDefectLayer
from sandbox.implicit_defects import ImplicitDefectLayer,energy_basis
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.BASE/"phase9_complete_temperature"
def freeze():
    payload=dict(version=1,created_utc=energy.now(),
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (
            Path(__file__),ROOT/"src/sandbox/material_profile.py",ROOT/"src/sandbox/temperature_defects.py",
            ROOT/"src/sandbox/implicit_defects.py",ROOT/"src/sandbox/thermo.py")},
        inputs={str(p.relative_to(ROOT)):energy.sha(p) for p in (
            ROOT/"results/phase0/kroger_ga2o3_092724_numeric.npz",
            energy.BASE/"phase8_process_temperature/band_profiles_v2/all_predictions.csv")},
        profiles=[REPOSITORY_PROFILE.to_dict(),PUBLISHED_PROFILE.to_dict()],
        stage="numerical fullhot/frozencold layer,not predictive accuracy",
        shared_relation="neutral donor in bothhot andcold: H0=Hplus+Ec(T)-E0exp(-s(C/1e18)^(1/3)),spin2",
        outputs=["Halln","absolutecharge","q_squaredcharge"],transport="CTcompatibility300Konly",
        gates="32states,stock/charge/probability,sameTidentity,old300replay,fullIFT,profileinterleaving/cache and JSONroundtrip")
    p=OUT/"numerical_protocol.json"
    if p.exists():
        d=json.loads(p.read_text())
        for k in ("code_hashes","inputs"):assert d[k]==payload[k]
    else:energy.write(p,payload)

def main():
    freeze();db=load();checks=[];states=[];gradients=[]
    initial_constants=(thermo.NC_REF,thermo.NV_REF,thermo.EG0)
    for prof in (REPOSITORY_PROFILE,PUBLISHED_PROFILE):
        try:prof.Nc_ref=1.
        except FrozenInstanceError:immutable=True
        else:immutable=False
        checks.append(dict(name="immutable_profile",profile=prof.name,passed=immutable))
        again=MaterialProfile.from_dict(json.loads(json.dumps(prof.to_dict())))
        checks.append(dict(name="serialized_profile",profile=prof.name,passed=again.fingerprint==prof.fingerprint))
        for element in ("Si","Sn"):
            for ext in (False,True):
                layer=TemperatureDefectLayer(db,(element,),profile=prof,donor_extension=ext)
                try:layer.profile=REPOSITORY_PROFILE
                except AttributeError:locked=True
                else:locked=False
                checks.append(dict(name="layer_profile_readonly",profile=prof.name,element=element,ext=ext,passed=locked))
                for To in (40.,150.,300.,1400.):
                    p=layer.parameters(T=1400,Nd=1e16,doses=(1e18,),Tobs=To,screen_s=.5)
                    a=layer.predict(p);s=layer.last_state;d=a.diagnostics
                    atom_hot=layer.db.dm.T@s["hot_charge"];atom_cold=layer.db.dm.T@s["cold_charge"]
                    atomerr=float(abs(atom_hot-atom_cold).max()/max(abs(atom_hot).max(),1.))
                    parentcold=np.bincount(layer.parent,weights=s["cold_charge"],minlength=layer.db.nD)
                    perr=float(abs(parentcold-s["totals"]).sum()/max(s["totals"].sum(),1))
                    good=(atomerr<1e-11 and perr<1e-11 and d["parent_probability_error"]<1e-12
                          and d["charge_relative_residual"]<1e-8 and d["cold_Q_EF_cm3_per_eV"]<0
                          and abs(d["achieved_concentrations_cm3"][element]/1e18-1)<1e-8)
                    checks.append(dict(name="stock_and_charge",profile=prof.name,element=element,extension=ext,
                        Tobs=To,atom_error=atomerr,parent_error=perr,passed=good))
                    states.append(dict(profile=prof.name,element=element,extension=ext,Tobs=To,
                                       values=a.values,diagnostics=d))
                    if To==1400:
                        eferr=float(abs(s["ef_cold"]-s["hot"][0]))
                        poperr=float(abs(s["cold_charge"]-s["hot_charge"]).sum()/s["hot_charge"].sum())
                        checks.append(dict(name="noquench_identity",profile=prof.name,element=element,
                            extension=ext,EF_error=eferr,population_L1_relative=poperr,passed=eferr<1e-9 and poperr<1e-8))
                checks.append(dict(name="same_stock_temperature_cache",profile=prof.name,element=element,extension=ext,
                    solves=layer.hot_solve_calls,hits=layer.hot_cache_hits,passed=layer.hot_solve_calls==1 and layer.hot_cache_hits==3))
    # All input partials, including energies, temperatures, materials and shared binding.
    for prof in (REPOSITORY_PROFILE,PUBLISHED_PROFILE):
        for elems,doses in ((("Si",),(1e18,)),(("Si","H"),(5.4e17,1.8e17))):
            b=energy_basis(db,"negative_charge_states")
            layer=TemperatureDefectLayer(db,elems,profile=prof,basis=b,donor_extension=True,donor_element="Si")
            for To in (40.,300.):
                p=layer.parameters(T=1450,Nd=1e16,doses=doses,Tobs=To,weights=(.03,),screen_s=.5)
                a=layer.predict(p)
                for j,name in enumerate(layer.parameter_names):
                    h=1e-5;x=p.copy();y=p.copy();x[j]+=h;y[j]-=h
                    fd=(layer.predict(x).values-layer.predict(y).values)/(2*h)
                    err=float(np.max(abs(fd-a.jacobian[:,j])/(1+abs(fd))))
                    gradients.append(dict(profile=prof.name,elements=elems,Tobs=To,parameter=name,
                        finite_difference=fd,analytic=a.jacobian[:,j],scaled_error=err))
                    checks.append(dict(name="full_IFT",profile=prof.name,elements=elems,Tobs=To,
                                       parameter=name,value=err,passed=err<3e-5))
    # Replay both complete oldband scenarios at their actual sourceexcluded parameters.
    reference=read_result_csv(energy.BASE/"phase8_process_temperature/band_profiles_v2/all_predictions.csv")
    reference=reference[reference.process_mode.eq("refit")&reference.budget.eq(0)&
                        reference.profile.isin(["repository_default","published_DOS_gap"])]
    data=read_result_csv(energy.BASE/"dataset_v1.csv").set_index("sample_id")
    cache={};replay=[]
    for r in reference.itertuples():
        row=data.loc[r.sample_id];key=(r.profile,row.element)
        if key not in cache:
            prof=REPOSITORY_PROFILE if r.profile=="repository_default" else PUBLISHED_PROFILE
            cache[key]=TemperatureDefectLayer(db,(row.element,),profile=prof)
        layer=cache[key]
        path=energy.BASE/f"phase8_process_temperature/band_profiles_v2/{r.profile}/refit/fits/{r.fit_key}.json"
        w=json.loads(path.read_text())["parameters"];l0,l1=np.log10(600),np.log10(1950)
        T=10**(l0+(l1-l0)*expit(w[0]))
        p=layer.parameters(T=T,Nd=10**w[1],doses=(row.incorporated_cm3,))
        a=layer.predict(p);err=float(abs(a.values[0]-r.predicted_log_nH))
        replay.append(dict(profile=r.profile,sample_id=r.sample_id,error_dex=err))
    checks.append(dict(name="58actual300K_replays",maximum=max(r["error_dex"] for r in replay),
                       passed=len(replay)==58 and max(r["error_dex"] for r in replay)<2e-8))
    # Scalar state + all old parameter derivatives in empiricalRTcompatibility head.
    old=ImplicitDefectLayer(db,("Si",));p=old.parameters(T=1400,Nd=1e16,doses=(1e18,))
    a=old.predict(p);l=TemperatureDefectLayer(db,("Si",))
    r=l.predict(l.parameters(T=1400,Nd=1e16,doses=(1e18,)));values,jac=l.legacy_300_outputs(r)
    checks.append(dict(name="oldRTvalues_and_partials",value_error=float(abs(values-a.values).max()),
        jacobian_error=float(abs(jac[:,:old.np]-a.jacobian).max()),
        passed=bool(abs(values-a.values).max()<1e-8 and abs(jac[:,:old.np]-a.jacobian).max()<1e-7)))
    try:
        pp=l.parameters(Tobs=40);l.legacy_300_outputs(l.predict(pp))
    except ValueError:rejected=True
    else:rejected=False
    checks.append(dict(name="reject_RTtransport_outside300",passed=rejected))
    x=TemperatureDefectLayer(db,("Si",),profile=REPOSITORY_PROFILE)
    y=TemperatureDefectLayer(db,("Si",),profile=PUBLISHED_PROFILE)
    xp=x.parameters();a=x.predict(xp).values.copy();y.predict(y.parameters());b=x.predict(xp).values
    checks.append(dict(name="interleaved_profiles_no_global_mutation",
        passed=bool(np.array_equal(a,b) and initial_constants==(thermo.NC_REF,thermo.NV_REF,thermo.EG0))))
    # Actual CPU first-order Torch chain, no claim of a new GPU-trained model.
    import torch
    l=TemperatureDefectLayer(db,("Si",),profile=PUBLISHED_PROFILE,donor_extension=True)
    p=l.parameters(Tobs=150,screen_s=.5);a=l.predict(p)
    t=torch.tensor(p,dtype=torch.float64,requires_grad=True)
    out=l.tensor(t);weights=torch.tensor([.3,.2,.1],dtype=torch.float64);(weights@out).backward()
    err=float(abs(t.grad.numpy()-weights.numpy()@a.jacobian).max())
    checks.append(dict(name="Torch_first_order_full_chain_CPU",value=err,passed=err<1e-9))
    result=dict(passed=all(c["passed"] for c in checks),checks=checks,states=states,
                gradients=gradients,replay=replay,maximum_gradient_scaled_error=max(v["scaled_error"] for v in gradients))
    energy.write(OUT/"numerical_gates.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),
        maximum_gradient_scaled_error=result["maximum_gradient_scaled_error"],
        failed=[c for c in checks if not c["passed"]]),indent=2))
    if not result["passed"]:raise RuntimeError("Full temperature gates failed; fitting prohibited")

if __name__=="__main__":main()
