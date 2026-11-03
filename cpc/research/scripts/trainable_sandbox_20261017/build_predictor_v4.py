"""Add the stronger same-input Ridge control without changing previous modes."""
from pathlib import Path
import os,sys,json,pickle,shutil
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.special import expit
from sandbox.trainable_predictor_v4 import BaseFeatureBank
from sandbox.energy_learning import reconstruct
from sandbox.property_metric_learning import element_database
from sandbox.implicit_defects import ImplicitDefectLayer
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as fitter

PARENT=energy.BASE/"predictor_bundle_v3"
OUT=energy.BASE/"predictor_bundle_v4";TRAIN=OUT/"training"
METHOD="ridge_energy_tuned";DBS={}

def model_path(e):return energy.BASE/"phase5_reference_features/models"/f"{e}__ridge_base_energy.pkl"
def freeze():
    p=TRAIN/"protocol.json"
    inputs=[PARENT/"manifest.json",*[model_path(e) for e in ("Si","Sn")],
        energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv",
        energy.BASE/"phase5_reference_features/real_property/summary.csv",
        energy.BASE/"phase5_reference_features/new_property_source/summary.csv"]
    payload=dict(created_utc=energy.now(),new_method=METHOD,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (
            Path(__file__),Path(fitter.__file__),ROOT/"src/sandbox/trainable_predictor_v4.py")},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        energy_learning="same18features, normalizedweights and6alphas, nested whole-element energy selection",
        attribution="regularization-grid control, not native/reference-feature novelty",
        final_fit="all29original records/5sources; innerLOFO shrinkage, deployment not heldout evidence",
        inherited_modes="copied exactly; no new fit or changed energetic observations",
        budget="zero extra energetic observations beyond Si844/Sn850supplied")
    if p.exists():
        old=json.loads(p.read_text())
        for key in ("code_hashes","input_hashes"):assert old[key]==payload[key]
    else:energy.write(p,payload)

def database(element,method):
    if element not in DBS:
        db,r,_=energy.records();target=[v for v in r if v["element"]==element]
        with model_path(element).open("rb") as f:m=pickle.load(f)
        assert element not in m.training_elements and m.kind=="base"
        DBS[element]=reconstruct(db,target,m.predict(target,BaseFeatureBank()))
    return DBS[element]

def point(row,w,target,method):
    key=(target,row.element)
    if key not in fitter.LAYERS:
        db,_=element_database(database(target,method),row.element)
        fitter.LAYERS[key]=ImplicitDefectLayer(db,(row.element,))
    layer=fitter.LAYERS[key];s=expit(w[0]);logT=fitter.L0+(fitter.L1-fitter.L0)*s
    p=layer.predict(layer.parameters(T=10**logT,Nd=10**w[1],doses=(row.incorporated_cm3,),pO2=1e-5))
    chain=np.zeros((layer.np,2));chain[0,0]=(fitter.L1-fitter.L0)*s*(1-s);chain[1,1]=1.
    return p,p.jacobian@chain

def main():
    freeze();fitter.OUT=TRAIN;fitter.freeze=freeze;fitter.database=database;fitter.point=point
    fitter.read_result_csv=read_result_csv
    fitter.DATA=read_result_csv(energy.BASE/"dataset_v1.csv");fitter.ROLES=read_result_csv(energy.BASE/"episodes_v1.csv")
    fitter.FITS.clear();fitter.LAYERS.clear()
    manifest=json.loads((PARENT/"manifest.json").read_text())
    for entry in manifest["files"]:
        dest=OUT/entry["path"];dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert energy.sha(dest)==entry["sha256"]
        else:shutil.copyfile(PARENT/entry["path"],dest)
    manifest["version"]=4;manifest["parent_manifest_sha256"]=energy.sha(PARENT/"manifest.json")
    manifest["methods"].append(METHOD)
    families=sorted(fitter.DATA.family.unique());selections=[]
    for e in ("Si","Sn"):
        folds={held:fitter.fit([f for f in families if f!=held],e,METHOD) for held in families}
        shrink={}
        for budget in (0,1,2):
            rows=[]
            for lam in ([np.inf] if budget==0 else [.1,1.,10.,np.inf]):
                score=float(np.mean([fitter.evaluate(folds[held],held,budget,lam)[0] for held in families]))
                rows.append(dict(lam="infinity" if np.isinf(lam) else lam,score=score))
            best=min(rows,key=lambda x:(round(x["score"],10),-np.inf if x["lam"]=="infinity" else -x["lam"]))
            shrink[str(budget)]=best["lam"];selections.append(dict(element=e,budget=budget,selected=best,candidates=rows))
        fit=fitter.fit(families,e,METHOD);a,b=fit["parameters"];s=expit(a)
        manifest["process_models"][e][METHOD]=dict(effective_Tf_K=float(10**(fitter.L0+(fitter.L1-fitter.L0)*s)),
            effective_Nd_cm3=float(10**b),fit_key=fit["key"],anchor_shrinkage=shrink,
            training_records=29,training_families=families)
        source=model_path(e);dest=OUT/"models"/source.name;shutil.copyfile(source,dest)
        with source.open("rb") as f:model=pickle.load(f)
        manifest["files"].append(dict(path=str(dest.relative_to(OUT)),sha256=energy.sha(dest)))
        manifest["energy_models"][e][METHOD]=dict(path=str(dest.relative_to(OUT)),
            sha256=energy.sha(dest),prediction_interface="explicit_base_bank",
            training_elements=model.training_elements,kind=model.kind,alpha=model.alpha)
        manifest["energy_budget"][e][METHOD]=dict(manifest["energy_budget"][e]["ridge"])
        manifest["supplied_energy_fingerprints"][e][METHOD]=manifest["supplied_energy_fingerprints"][e]["ridge"]
    result=read_result_csv(energy.BASE/"phase5_reference_features/real_property/summary.csv")
    result=result[result.method.eq("ridge_base_energy")&result.scope.eq("all_query")].sort_values("budget")
    src=read_result_csv(energy.BASE/"phase5_reference_features/new_property_source/summary.csv")
    src=src[src.method.eq("ridge_base_energy")].sort_values("budget")
    manifest["validation_evidence"]["zero_extra_tuned_ridge"]=dict(
        original_cohort_MAE_dex=result.family_equal_MAE.tolist(),
        related_study_MAE_dex=src.nH_MAE.tolist(),
        claim="stronger regularization-grid control; no reference-feature increment claimed",
        not_blind=True)
    energy.write(OUT/"manifest.json",manifest);energy.write(TRAIN/"anchor_selection.json",selections)
    print(OUT,flush=True)

if __name__=="__main__":main()
