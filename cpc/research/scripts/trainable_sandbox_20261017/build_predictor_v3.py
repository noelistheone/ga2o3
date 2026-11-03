"""Deliver the strongest supported two-energy ML+Sandbox variants explicitly."""
from pathlib import Path
import os,sys,json,pickle,shutil
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.special import expit
from sandbox.energy_learning import reconstruct
from sandbox.property_metric_learning import element_database
from sandbox.implicit_defects import ImplicitDefectLayer
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as fitter

PARENT=energy.BASE/"predictor_bundle_v2"
OUT=energy.BASE/"predictor_bundle_v3";TRAIN=OUT/"training"
METHODS=("goal_ridge_2","goal_mean_2")
DBS={};OBS={};MODELS={}

def prior(element,method):
    model="ridge" if method=="goal_ridge_2" else "global_mean"
    cov="diagonal_xi" if method=="goal_ridge_2" else "diagonal_A"
    return energy.OUT/"models"/f"{element}__{model}.pkl",energy.OUT/"acquisition_ablation/states"/f"{element}__{model}__{cov}.npz"

def freeze():
    inputs=[PARENT/"manifest.json",energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv"]
    inputs += [p for e in ("Si","Sn") for m in METHODS for p in prior(e,m)]
    payload=dict(version=3,created_utc=energy.now(),methods=METHODS,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (
            Path(__file__),Path(fitter.__file__),ROOT/"src/sandbox/trainable_predictor_v3.py")},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        selection="reuse fixed phase2 energy label choices, no query-material-label selection",
        observation_budget="Si844+2 supplied/Sn850+2 supplied; remaining27/21 target labels unknown",
        process="same all29 final fit and5fold source-CV anchor shrinkage; deployment, not heldout accuracy",
        parent="v2 fits and modes copied exactly, no retraining",
        grade="local research bundle, all material evidence retrospective")
    p=TRAIN/"protocol.json"
    if p.exists():
        old=json.loads(p.read_text())
        for k in ("code_hashes","input_hashes"):assert old[k]==payload[k]
    else:energy.write(p,payload)

def database(element,method):
    key=(element,method)
    if key not in DBS:
        db,r,_=energy.records();target=[v for v in r if v["element"]==element]
        modelpath,states=prior(element,method)
        with modelpath.open("rb") as f:model=pickle.load(f)
        assert element not in model.training_elements
        z=np.load(states);labels=str(z["labels"][2]).split("|")
        assert len(labels)==len(set(labels))==2
        pred=model.predict(target);observations=[]
        for i,rec in enumerate(target):
            if rec["label_id"] in labels:
                observations.append(dict(label_id=rec["label_id"],index=rec["index"],
                    formation_intercept_eV=float(db.dHo[rec["index"]]),
                    reference_index=rec["reference_index"],source="published KROGER withheld-label replay",
                    new_calculation=False))
                pred[i]=rec["crossing_eV"]
        assert abs(pred-z["means"][2]).max()<1e-12
        DBS[key]=reconstruct(db,target,pred);OBS[key]=observations;MODELS[key]=model
    return DBS[key]

def point(row,w,target,method):
    key=(target,method,row.element)
    if key not in fitter.LAYERS:
        db,_=element_database(database(target,method),row.element)
        fitter.LAYERS[key]=ImplicitDefectLayer(db,(row.element,))
    layer=fitter.LAYERS[key];s=expit(w[0]);logT=fitter.L0+(fitter.L1-fitter.L0)*s
    pred=layer.predict(layer.parameters(T=10**logT,Nd=10**w[1],
                       doses=(row.incorporated_cm3,),pO2=1e-5))
    chain=np.zeros((layer.np,2));chain[0,0]=(fitter.L1-fitter.L0)*s*(1-s);chain[1,1]=1.
    return pred,pred.jacobian@chain

def main():
    freeze();fitter.OUT=TRAIN;fitter.freeze=freeze;fitter.database=database;fitter.point=point
    fitter.read_result_csv=read_result_csv
    fitter.DATA=read_result_csv(energy.BASE/"dataset_v1.csv")
    fitter.ROLES=read_result_csv(energy.BASE/"episodes_v1.csv")
    fitter.FITS.clear();fitter.LAYERS.clear()
    manifest=json.loads((PARENT/"manifest.json").read_text())
    for item in manifest["files"]:
        dest=OUT/item["path"];dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert energy.sha(dest)==item["sha256"]
        else:shutil.copyfile(PARENT/item["path"],dest)
    manifest["version"]=3;manifest["parent_manifest_sha256"]=energy.sha(PARENT/"manifest.json")
    manifest["methods"]+=list(METHODS)
    families=sorted(fitter.DATA.family.unique());all_selection=[]
    for e in ("Si","Sn"):
        for method in METHODS:
            database(e,method)
            folds={held:fitter.fit([f for f in families if f!=held],e,method) for held in families}
            shrink={}
            for budget in (0,1,2):
                rows=[]
                for lam in ([np.inf] if budget==0 else [.1,1.,10.,np.inf]):
                    score=float(np.mean([fitter.evaluate(folds[held],held,budget,lam)[0] for held in families]))
                    rows.append(dict(lam="infinity" if np.isinf(lam) else lam,score=score))
                best=min(rows,key=lambda x:(round(x["score"],10),-np.inf if x["lam"]=="infinity" else -x["lam"]))
                shrink[str(budget)]=best["lam"]
                all_selection.append(dict(element=e,method=method,budget=budget,candidates=rows,selected=best))
            final=fitter.fit(families,e,method)
            a,b=final["parameters"];s=expit(a)
            manifest["process_models"][e][method]=dict(
                effective_Tf_K=float(10**(fitter.L0+(fitter.L1-fitter.L0)*s)),effective_Nd_cm3=float(10**b),
                fit_key=final["key"],anchor_shrinkage=shrink,
                training_records=29,training_families=families)
            source,_=prior(e,method);dest=OUT/"models"/source.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert energy.sha(dest)==energy.sha(source)
            else:shutil.copyfile(source,dest)
            manifest["files"].append(dict(path=str(dest.relative_to(OUT)),sha256=energy.sha(dest)))
            model=MODELS[(e,method)]
            manifest["energy_models"][e][method]=dict(path=str(dest.relative_to(OUT)),
                sha256=energy.sha(dest),training_elements=model.training_elements,
                known_energy_observations=OBS[(e,method)])
            old=manifest["energy_budget"][e]["ridge"]
            manifest["energy_budget"][e][method]=dict(
                supplied_energy_states=old["supplied_energy_states"]+2,
                initially_hidden_energy_states=old["hidden_energy_states"],
                hidden_energy_states=old["hidden_energy_states"]-2,
                extra_revealed_energies=2,catalogue_and_parent_references_supplied=True,
                observed_energies_provided_in_manifest=True)
            manifest["supplied_energy_fingerprints"][e][method]=manifest["supplied_energy_fingerprints"][e]["ridge"]
    manifest["files"]=list({f["path"]:f for f in manifest["files"]}.values())
    manifest["validation_evidence"]["two_energy_variants"]=dict(
        goal_ridge_2_MAE_dex=[.259106,.228140,.212431],
        goal_mean_2_MAE_dex=[.259410,.228677,.212683],
        comparison="same2energy budget; selected sets differ in Sn, so not pure ML-mean attribution",
        related_study_goal_ridge_2_MAE_dex=[.541016,1.041222],
        interpretation="strong on old cohort, adverse related-study transfer; no independent accuracy guarantee")
    energy.write(OUT/"manifest.json",manifest)
    energy.write(TRAIN/"anchor_selection.json",all_selection)
    energy.write(OUT/"build_audit.json",dict(parent_manifest_sha256=energy.sha(PARENT/"manifest.json"),
        methods=METHODS,observations={f"{e}/{m}":o for (e,m),o in OBS.items()},
        final_fit_data="29/5carrier, inherited52/11transport",
        material_query_labels_used_for_energy_selection=False,
        full_target_nonreference_labels_needed_at_inference=False))
    print(OUT,flush=True)

if __name__=="__main__":main()
