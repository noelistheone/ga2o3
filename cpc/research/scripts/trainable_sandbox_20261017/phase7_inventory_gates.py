"""Pre-fit inventory algebra, IFT and limiting-case checks. No outcome training."""
from pathlib import Path
import os, sys, json, argparse
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sandbox.kroger_db import load
from sandbox.implicit_defects import ImplicitDefectLayer, energy_basis
from sandbox.inventory_closure import InventoryClosureLayer
from sandbox.property_metric_learning import element_database
from sandbox import transport
import phase2_energy as energy

OUT = energy.BASE/"phase7_inventory"
MODELS = {
    "eq": 2, "fill_endpoint": 2, "fill_const": 3,
    "fill_temperature": 4, "temperature_relation": 3,
    "temperature_element_relation": 4,
}

def freeze():
    p = OUT/"protocol.json"
    code = [Path(__file__), ROOT/"src/sandbox/inventory_closure.py",
            ROOT/"scripts/trainable_sandbox_20261017/phase7_inventory_property.py",
            ROOT/"src/sandbox/implicit_defects.py",
            ROOT/"src/sandbox/property_metric_learning.py",
            ROOT/"docs/ML_SANDBOX_INVENTORY_PHASE7_20261017.md"]
    payload = dict(created_utc=energy.now(), version=1, models=MODELS,
        code_hashes={str(q.relative_to(ROOT)):energy.sha(q) for q in code},
        input_hashes={str(q.relative_to(ROOT)):energy.sha(q) for q in (
            energy.BASE/"dataset_v1.csv", energy.BASE/"episodes_v1.csv",
            ROOT/"results/phase0/kroger_ga2o3_092724_numeric.npz")},
        theory="docs/ML_SANDBOX_INVENTORY_PHASE7_20261017.md",
        hot_model="fixed full energetic input, unchanged specified FD/thermochemistry",
        frozen_inventory="nonnegative openGa map, equal two audited donor-site allocation, no fitted site weights",
        primary="fill_temperature versus eq,fill_const,temperature_element_relation",
        process_parameters="logit effectiveT, log10Nd; then eta logit or temperature coefficients",
        bounds=dict(effectiveT_K=[600,1950], log10Nd=[14,18], logits=[-6,6], slopes=[-4,4]),
        optimization="two declared starts per model; least_squares analytic gradients; max_nfev60",
        anchor_adaptation="exact frozen0/1/2 protocol; inner source-held-out lambda .1/1/10/inf",
        source_grade="retrospective five DOI groups; related-study outcomes already development exposed",
        pressure="all29 unreported; fixed unmeasured1e-5atm scenario",
        measured_endpoint="300K single-electron Hall carrier density; not lifetime or rareearth ranking",
        metadata="11growth-temperature lower boundary proxies retained",
        reference_temperature="effective reference; not independently measured freeze temperature",
        gpu="authorized GPU1, small dense fitting CPU; no use of GPU0",
        all_results_retained=True)
    if p.exists():
        old=json.loads(p.read_text())
        for k in ("models","code_hashes","input_hashes"): assert old[k]==payload[k], k
    else: energy.write(p,payload)

def gates():
    freeze()
    db=load(); checks=[]; records=[]; derivative_rows=[]; limits=[]; maps=[]
    for element in ("Si","Sn"):
        layer=InventoryClosureLayer(db,element)
        assoc=InventoryClosureLayer(db,element,map_kind="association")
        reduced,_=element_database(db,element); old=ImplicitDefectLayer(reduced,(element,))
        maps.append(layer.map_audit); maps.append(assoc.map_audit)
        for T,C in ((1050.,1e17),(1325.,1e19),(1750.,1e19)):
            basep=layer.parameters(T=T,Nd=9.1e15,doses=(C,))
            baseline=old.predict(basep)
            zero=layer.predict(np.r_[basep,0.])
            error=float(abs(zero.values-baseline.values).max())
            checks.append(dict(name="eta0_old_replay",element=element,T=T,C=C,value=error,passed=error<1e-8))
            for eta in (0.,.35,1.):
                pred=layer.predict(np.r_[basep,eta])
                dg=pred.diagnostics; change=np.array(dg["atom_change_cm3"])
                atom_scale=max(C,dg["open_Ga_consumed_cm3"],1)
                wanted=np.zeros(len(db.elements));wanted[0]=dg["open_Ga_consumed_cm3"]
                atom_err=float(abs(change-wanted).max()/atom_scale)
                stockerr=abs(dg["achieved_concentrations_cm3"][element]/C-1)
                passed=(atom_err<1e-10 and stockerr<1e-8
                    and dg["cold_Q_EF_cm3_per_eV"]<0 and dg["gradient_valid"]
                    and dg["charge_relative_residual"]<1e-8
                    and layer.last_inventory["mixed"].min()>=0)
                records.append(dict(element=element,T=T,C=C,predicted_log_nH=pred.values[0],**dg))
                checks.append(dict(name="inventory_and_charge",element=element,T=T,C=C,eta=eta,
                                   atom_error=atom_err,dopant_error=stockerr,passed=passed))
                nalg=zero.diagnostics["free_n_cm3"]+3*eta*dg["vacancy_equivalent_hot_cm3"]
                halg=np.log10(nalg/float(transport.hall_factor(300,nalg)))
                limits.append(dict(element=element,T=T,C=C,eta=eta,
                    full_log_nH=pred.values[0],algebraic_log_nH=halg,difference_dex=pred.values[0]-halg))
            middle=layer.predict(np.r_[basep,.35])
            J=layer.last_inventory["jacobian"][:,[0,-1]]
            J=J/np.maximum(np.linalg.norm(J,axis=0),1e-300)
            sv=np.linalg.svd(J,compute_uv=False)
            records[-1]["stock_T_eta_singular_values_column_normalized"]=sv.tolist()
            # Exact saturated-charge association accounting, independent of fits.
            qsat=layer.kX-3*np.maximum(layer.vGa,0)
            # Only changed columns obey this donor+vacancy branch construction.
            changed=np.any(assoc.D!=0,axis=0)
            deltaq=qsat@assoc.D
            checks.append(dict(name="association_saturated_charge_conservation",element=element,
                value=float(abs(deltaq[changed]).max(initial=0)),passed=bool(np.all(deltaq[changed]==0))))
            ap=assoc.predict(np.r_[basep,1.])
            limits.append(dict(element=element,T=T,C=C,eta=1,kind="association_actual_multicharge",
                               difference_dex=ap.values[0]-baseline.values[0]))
        # The meaningful risk is the full implicit gradient, including energy inputs.
        b=energy_basis(db,"negative_charge_states")
        gl=InventoryClosureLayer(db,element,basis=b)
        for T,C in ((1200.,1e18),(1550.,1e19)):
            p=gl.inventory_parameters(T=T,Nd=9.1e15,doses=(C,),weights=(.03,),eta=.4)
            pred=gl.predict(p)
            for j,name in enumerate(pred.parameter_names):
                h=1e-5;plus=p.copy();minus=p.copy();plus[j]+=h;minus[j]-=h
                fd=(gl.predict(plus).values-gl.predict(minus).values)/(2*h)
                absolute=float(abs(fd-pred.jacobian[:,j]).max())
                scaled=float(np.max(abs(fd-pred.jacobian[:,j])/(1+abs(fd))))
                derivative_rows.append(dict(element=element,T=T,C=C,parameter=name,
                    finite_difference=fd,analytic=pred.jacobian[:,j],absolute_error=absolute,scaled_error=scaled))
                checks.append(dict(name="IFT_central_difference",element=element,T=T,C=C,
                                   parameter=name,value=scaled,passed=scaled<3e-5))
    energy.write(OUT/"numerical_gates.json",dict(passed=all(c["passed"] for c in checks),
        checks=checks,states=records,gradients=derivative_rows,limits=limits,map_audits=maps))
    print(json.dumps(dict(checks=len(checks),passed=all(c["passed"] for c in checks),
                         max_gradient_scaled_error=max(x["scaled_error"] for x in derivative_rows)),indent=2))
    if not all(c["passed"] for c in checks):raise RuntimeError("Inventory gates failed; real fitting prohibited")

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--freeze",action="store_true")
    a=ap.parse_args();freeze() if a.freeze else gates()
