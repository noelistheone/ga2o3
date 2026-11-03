"""Post-evaluation 2x2 energy/process swap, not a new predictor or tuning step."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.special import expit
from sandbox.energy_learning import reconstruct
from sandbox.property_metric_learning import element_database
from sandbox.implicit_defects import ImplicitDefectLayer
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as fitter

OUT=energy.BASE/"phase56_validation/counterfactual"
SPECS={
 "tuned_ridge":dict(method="ridge_base_energy",folder=energy.BASE/"phase5_reference_features/real_property",energies="phase5"),
 "projected_property":dict(method="rbf_closed_property",folder=energy.BASE/"phase5_reference_features/real_property",energies="phase5"),
 "nonlinear":dict(method="kernel_nonlinear_l1",folder=energy.BASE/"phase4_nonlinear/real_property",energies="phase4"),
 "goal_ridge_2":dict(method="Ridge_diagonal_xi_k2",folder=energy.OUT/"acquired_property_addendum",energies="two")}
LAYERS={}

def load_fit(folder,held,method):
    d=read_result_csv(folder/"outer"/f"{fitter.digest(held)}__{method}.csv")
    key=d.fit_key.iloc[0]
    fit=json.loads((folder/"fits"/f"{key}.json").read_text())
    assert held not in fit["train_families"]
    return fit,d[d.budget.eq(0)].set_index("sample_id")

def library(db,records,e,spec):
    rr=[r for r in records if r["element"]==e]
    if spec["energies"]=="two":
        means=np.load(energy.OUT/"acquisition_ablation/states"/f"{e}__ridge__diagonal_xi.npz")["means"][2]
    else:
        folder=energy.BASE/("phase5_reference_features" if spec["energies"]=="phase5" else "phase4_nonlinear")
        paths=sorted((folder/"energy_outer").glob(f"{e}__*.csv")) if spec["energies"]=="phase5" else [folder/"energy_outer"/f"{e}.csv"]
        data=pd.concat([read_result_csv(p) for p in paths],ignore_index=True)
        data=data[data.method.eq(spec["method"])]
        lookup=dict(zip(data.label_id,data.predicted_crossing_eV))
        means=np.array([lookup[r["label_id"]] for r in rr])
    return reconstruct(db,rr,means)

def predict(db,e,w,C,key):
    if key not in LAYERS:
        active,_=element_database(db,e);LAYERS[key]=ImplicitDefectLayer(active,(e,))
    layer=LAYERS[key];s=expit(w[0]);T=10**(fitter.L0+(fitter.L1-fitter.L0)*s)
    try:
        r=layer.predict(layer.parameters(T=T,Nd=10**w[1],doses=(C,),pO2=1e-5))
        return dict(y=float(r.values[0]),physical=bool(r.diagnostics["single_electron_Hall_valid"] and
            r.diagnostics["above_carrier_detection_floor"]),failure=None)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        return dict(y=None,physical=False,failure=repr(exc))

def main():
    db,r,_=energy.records();d=read_result_csv(energy.BASE/"dataset_v1.csv")
    payload=dict(created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),
        models={m:s["method"] for m,s in SPECS.items()},queries="all29zero-anchor samples,5sources",
        design="A0,w0 full-energy/source-held-out reference; A1,w1 ML/source-held-out fit; evaluate00,10,01,11",
        identity="e11=e00+(y10-y00)+(y11-y10); MAEs are not additive",
        interpretation="posthoc model-internal counterfactual; oracle energetic inputs; no new physical-cause identification",
        fits="frozen source-excluded fits only, no refit or selection",
        numeric_refusals="all retained, decomposition only when required cells finite")
    path=OUT/"protocol.json"
    if path.exists():
        old=json.loads(path.read_text());assert old["code_sha256"]==payload["code_sha256"]
    else:energy.write(path,payload)
    rows=[];fits=[]
    for held,g in d.groupby("family"):
        e=g.element.iloc[0]
        ref,refpred=load_fit(energy.BASE/"phase3_target_loss/real_property",held,"oracle_full")
        for method,spec in SPECS.items():
            fit,actual=load_fit(spec["folder"],held,spec["method"])
            approx=library(db,r,e,spec)
            fits.append(dict(method=method,held=held,reference_fit_key=ref["key"],ML_fit_key=fit["key"],
                train_sources=fit["train_families"],reference_parameters=ref["parameters"],ML_parameters=fit["parameters"]))
            for row in g.itertuples():
                a=predict(db,e,ref["parameters"],row.incorporated_cm3,("full",e))
                b=predict(approx,e,ref["parameters"],row.incorporated_cm3,(method,e))
                c=predict(db,e,fit["parameters"],row.incorporated_cm3,("full",e))
                f=predict(approx,e,fit["parameters"],row.incorporated_cm3,(method,e))
                out=dict(method=method,family=held,sample_id=row.sample_id,element=e,
                    observed_log_nH=row.observed_log_nH)
                for name,v in zip(("00","10","01","11"),(a,b,c,f)):
                    out["y"+name]=v["y"];out["physical"+name]=v["physical"];out["failure"+name]=v["failure"]
                if a["y"] is not None:
                    out["reference_replay_error"]=abs(a["y"]-refpred.loc[row.sample_id,"predicted_log_nH"])
                    assert out["reference_replay_error"]<2e-6
                if f["y"] is not None:
                    out["ML_replay_error"]=abs(f["y"]-actual.loc[row.sample_id,"predicted_log_nH"])
                    assert out["ML_replay_error"]<2e-6
                if all(v["y"] is not None for v in (a,b,f)):
                    e0=a["y"]-row.observed_log_nH;de=b["y"]-a["y"];dp=f["y"]-b["y"]
                    e1=f["y"]-row.observed_log_nH
                    out.update(reference_error=e0,energy_substitution_change=de,
                        process_refit_change=dp,final_error=e1,identity_error=abs(e1-e0-de-dp),
                        opposite_changes=bool(de*dp<0),
                        cancellation_fraction=1-abs(de+dp)/(abs(de)+abs(dp)) if abs(de)+abs(dp)>1e-12 else None)
                    assert out["identity_error"]<1e-10
                if all(v["y"] is not None for v in (a,b,c,f)):
                    out["energy_process_interaction"]=f["y"]-b["y"]-c["y"]+a["y"]
                rows.append(out)
    energy.csvwrite(OUT/"predictions.csv",rows);energy.write(OUT/"fit_sources.json",fits)
    data=pd.DataFrame(rows);by=[]
    for (m,f),g in data.groupby(["method","family"]):
        row=dict(method=m,family=f,n=len(g),
            four_cell_numeric=int(g[["y00","y10","y01","y11"]].notna().all(axis=1).sum()),
            energy_cross_physical=int(g.physical10.sum()),process_cross_physical=int(g.physical01.sum()),
            cancellation_count=int(g.opposite_changes.fillna(False).sum()))
        for col in ("reference_error","energy_substitution_change","process_refit_change","final_error","energy_process_interaction"):
            row["mean_abs_"+col]=float(abs(g[col]).mean()) if g[col].notna().all() else None
        by.append(row)
    by=pd.DataFrame(by);by.to_csv(OUT/"by_family.csv",index=False)
    summary=by.groupby("method").agg(families=("family","nunique"),queries=("n","sum"),
        four_cell_numeric=("four_cell_numeric","sum"),cancellation_count=("cancellation_count","sum"),
        reference_MAE=("mean_abs_reference_error","mean"),energy_change=("mean_abs_energy_substitution_change","mean"),
        process_change=("mean_abs_process_refit_change","mean"),final_MAE=("mean_abs_final_error","mean"),
        interaction=("mean_abs_energy_process_interaction","mean")).reset_index()
    summary.to_csv(OUT/"summary.csv",index=False)
    print(summary.to_string(index=False),flush=True)

if __name__=="__main__":main()
