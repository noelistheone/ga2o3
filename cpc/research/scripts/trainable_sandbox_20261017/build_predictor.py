"""Train final shared process calibration and assemble local research predictor.

Final all-data fits are deployment artifacts, never held-out accuracy evidence.
The shared energy learners remain their frozen target-element-excluded models.
"""
from pathlib import Path
import os,sys,json,pickle,shutil,argparse
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.special import expit
from sandbox.energy_learning import reconstruct
from sandbox.property_metric_learning import element_database
from sandbox.implicit_defects import ImplicitDefectLayer
from sandbox.trainable_predictor import database_fingerprint
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as fitter
import phase3_metric as metric
import phase4_nested as nonlinear
import transport_chain as transport

OUT=energy.BASE/"predictor_bundle_v1"
TRAIN=OUT/"training"
ALIASES={"nonlinear_kernel":"kernel_nonlinear_l1","ordinary_kernel":"kernel_energy",
         "ridge":"ridge","full_library":"oracle_full"}
MODELDB={};MODELS={}

def source_model(element,alias):
    if alias=="nonlinear_kernel":return nonlinear.OUT/"models"/f"{element}__kernel_nonlinear_l1.pkl"
    if alias=="ordinary_kernel":return metric.OUT/"models"/f"{element}__kernel_energy.pkl"
    if alias=="ridge":return energy.OUT/"models"/f"{element}__ridge.pkl"
    return None

def freeze():
    inputs=[energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv",
        ROOT/"results/chain_diagnosis_20261015/executed/mobility_bounds_and_inputs.csv"]
    inputs += [source_model(e,a) for e in ("Si","Sn") for a in ALIASES if a!="full_library"]
    payload=dict(version=1,created_utc=energy.now(),aliases=ALIASES,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in
            (Path(__file__),ROOT/"src/sandbox/trainable_predictor.py",Path(fitter.__file__))},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        final_fit="all original29 carrier records/5 sources; source-equal process fit with frozen equations",
        deployment_not_validation=True,
        energy_training="reuse nested held-element learners; target energies not included in runtime features/catalogue",
        adaptation_selection="LOFO on the29 training records at fixed0/1/2 roles; stronger shrinkage wins ties",
        transport="all original52 pairs/11sources, CT+family-weighted median; not held-out validation",
        support="Si/Sn, room-temperature electron Hall only; chemical dose is actual incorporation",
        oxygen="fixed unmeasured1e-5atm effective-O2 scenario; no ozone-equivalence claim",
        all_previous_studies_unchanged=True)
    p=TRAIN/"protocol.json"
    if p.exists():
        old=json.loads(p.read_text())
        for key in ("code_hashes","input_hashes"):assert old[key]==payload[key]
    else:energy.write(p,payload)

def database(element,method):
    key=(element,method)
    if key not in MODELDB:
        db,records,_=energy.records()
        if method!="oracle_full":
            alias=next(a for a,m in ALIASES.items() if m==method)
            with source_model(element,alias).open("rb") as f:model=pickle.load(f)
            assert element not in model.training_elements
            target=[r for r in records if r["element"]==element]
            db=reconstruct(db,target,model.predict(target))
            MODELS[(element,alias)]=model
        MODELDB[key]=db
    return MODELDB[key]

def point(row,w,target,method):
    key=(target,method,row.element)
    if key not in fitter.LAYERS:
        db,_=element_database(database(target,method),row.element)
        fitter.LAYERS[key]=ImplicitDefectLayer(db,(row.element,))
    layer=fitter.LAYERS[key];s=expit(w[0])
    logT=fitter.L0+(fitter.L1-fitter.L0)*s
    pred=layer.predict(layer.parameters(T=10**logT,Nd=10**w[1],
                       doses=(row.incorporated_cm3,),pO2=1e-5))
    chain=np.zeros((layer.np,2));chain[0,0]=(fitter.L1-fitter.L0)*s*(1-s);chain[1,1]=1.
    return pred,pred.jacobian@chain

def configure():
    freeze();fitter.OUT=TRAIN;fitter.freeze=freeze;fitter.database=database;fitter.point=point
    fitter.read_result_csv=read_result_csv
    fitter.DATA=read_result_csv(energy.BASE/"dataset_v1.csv")
    fitter.ROLES=read_result_csv(energy.BASE/"episodes_v1.csv")
    fitter.FITS.clear();fitter.LAYERS.clear()

def build():
    configure()
    db,records,_=energy.records()
    families=sorted(fitter.DATA.family.unique())
    process={e:{} for e in ("Si","Sn")};selection=[]
    for e in process:
        for alias,method in ALIASES.items():
            candidates={}
            folds={held:fitter.fit([f for f in families if f!=held],e,method) for held in families}
            for budget in (0,1,2):
                rows=[]
                for lam in ([np.inf] if budget==0 else [.1,1.,10.,np.inf]):
                    errors=[fitter.evaluate(folds[held],held,budget,lam)[0] for held in families]
                    rows.append(dict(lam="infinity" if np.isinf(lam) else lam,
                                     macro_MAE=float(np.mean(errors))))
                best=min(rows,key=lambda r:(round(r["macro_MAE"],10),
                         -np.inf if r["lam"]=="infinity" else -r["lam"]))
                candidates[str(budget)]=best["lam"]
                selection.append(dict(element=e,method=alias,budget=budget,candidates=rows,selected=best))
            fit=fitter.fit(families,e,method)
            if fit["parameters"] is None:raise RuntimeError("Final process calibration failed")
            a,b=fit["parameters"];s=expit(a)
            process[e][alias]=dict(
                effective_Tf_K=float(10**(fitter.L0+(fitter.L1-fitter.L0)*s)),
                effective_Nd_cm3=float(10**b),anchor_shrinkage=candidates,
                fit_key=fit["key"],training_records=len(fitter.DATA),training_families=families)
    # Final transport model is trained on the original cohort, never new-study labels.
    mobility=read_result_csv(ROOT/"results/chain_diagnosis_20261015/executed/mobility_bounds_and_inputs.csv")
    w=1/mobility.family.map(mobility.family.value_counts()).to_numpy(float)
    baseline=np.array([transport.logCT(x) for x in np.log10(mobility.carrier_cm3)])
    offset=transport.median(np.log10(mobility.mu_cm2Vs).to_numpy()-baseline,w)
    runtime_models={e:{} for e in process};files=[];budget={e:{} for e in process}
    for e in process:
        for alias in ALIASES:
            if alias=="full_library":
                budget[e][alias]=dict(supplied_energy_states=873,hidden_energy_states=0,
                    catalogue_and_parent_references_supplied=True,extra_revealed_energies=0)
                continue
            p=source_model(e,alias);dest=OUT/"models"/p.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert energy.sha(dest)==energy.sha(p)
            else:shutil.copyfile(p,dest)
            with p.open("rb") as f:model=pickle.load(f)
            runtime_models[e][alias]=dict(path=str(dest.relative_to(OUT)),
                sha256=energy.sha(dest),training_elements=model.training_elements)
            files.append(dict(path=str(dest.relative_to(OUT)),sha256=energy.sha(dest)))
            hidden=sum(r["element"]==e for r in records)
            budget[e][alias]=dict(supplied_energy_states=873-hidden,hidden_energy_states=hidden,
                catalogue_and_parent_references_supplied=True,extra_revealed_energies=0)
    forbidden={"crossing_eV","target_energy_eV","reference_energy_eV"}
    target_records={e:[{k:v for k,v in r.items() if k not in forbidden}
                       for r in records if r["element"]==e] for e in process}
    support={e:[float(fitter.DATA.loc[fitter.DATA.element.eq(e),"incorporated_cm3"].min()),
                float(fitter.DATA.loc[fitter.DATA.element.eq(e),"incorporated_cm3"].max())] for e in process}
    manifest=dict(version=1,created_utc=energy.now(),supported_dopants=["Si","Sn"],
        methods=list(ALIASES),database_fingerprint=database_fingerprint(db),
        fixed_pressure_atm=1e-5,files=files,target_records=target_records,
        energy_models=runtime_models,energy_budget=budget,process_models=process,
        chemical_support_cm3=support,transport=dict(offset_dex=offset,training_records=len(mobility),
            training_families=sorted(mobility.family.unique())),
        validation_evidence=dict(primary="29records/5sources retrospective source-held-out; see phase4_nonlinear/real_property",
            nonlinear_MAE_dex=[.32439542685277706,.3024349940364199,.27305077952751866],
            nonlinear_advantage_established=False,
            conditional_transport="52pairs/11sources,0.3666089dex; given measured nH",
            full_chain="5matched pairs/2sources; nonlinear zero-anchor mu MAE~1.276dex",
            new_study="Azizie2023 related-study corrected-label test,2rounded chemical inputs; adverse nH transfer",
            final_all_data_fit_is_heldout_evidence=False),
        privacy="Contains private third-party-derived catalogue and model artifacts; do not publish the bundle")
    energy.write(OUT/"manifest.json",manifest)
    energy.write(TRAIN/"anchor_selection.json",selection)
    energy.write(OUT/"build_audit.json",dict(
        target_labels_removed=all(not forbidden.intersection(r) for rr in target_records.values() for r in rr),
        target_elements_excluded=True,process_fit_records=len(fitter.DATA),transport_fit_records=len(mobility),
        source_protocol_sha256=energy.sha(TRAIN/"protocol.json"),bundle_manifest_sha256=energy.sha(OUT/"manifest.json"),
        final_training_not_accuracy_evidence=True))
    print("Local predictor bundle:",OUT,flush=True)

if __name__=="__main__":build()
