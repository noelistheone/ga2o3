"""Real-source validation of the known-reference feature ablation."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os,sys,json,hashlib,shutil,argparse
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
import phase3_metric as metric
import phase4_property as previous
import phase5_reference as training

OUT=training.OUT/"real_property"
METHODS=training.METHODS
CONTROLS=("oracle_full","global_mean","ridge","kernel_energy_select_energy","kernel_nonlinear_l1")
DBS={};ENERGY={};ORIGINAL_SUMMARY=fitter.summarize

def energy_frame(element):
    if element not in ENERGY:
        files=sorted((training.OUT/"energy_outer").glob(f"{element}__*.csv"))
        if len(files)!=8:raise RuntimeError("Target-element energy training is incomplete: "+element)
        import pandas as pd
        ENERGY[element]=pd.concat([read_result_csv(p) for p in files],ignore_index=True)
    return ENERGY[element]

def aliases():
    groups={};mapping={}
    for method in METHODS:
        h=hashlib.sha256()
        for e in ("Si","Sn"):
            d=energy_frame(e);d=d[d.method.eq(method)].sort_values("label_id")
            h.update("|".join(d.label_id).encode())
            h.update(np.asarray(d.predicted_crossing_eV,dtype="<f8").tobytes())
        mapping[method]=groups.setdefault(h.hexdigest(),method)
    return mapping

def freeze():
    training.frozen()
    assert json.loads((training.OUT/"feature_gates.json").read_text())["passed"]
    mapping=aliases()
    input_paths=[p for e in ("Si","Sn") for p in sorted((training.OUT/"energy_outer").glob(f"{e}__*.csv"))]
    controls=sorted(p for p in (previous.OUT/"outer").glob("*.csv")
                    if p.name.split("__",1)[1][:-4] in CONTROLS)
    if len(controls)!=25:raise RuntimeError("Missing frozen controls")
    payload=dict(version=1,created_utc=energy.now(),methods=METHODS,controls=CONTROLS,
        exact_energy_aliases=mapping,
        alias_rule="byte-identical sorted Si and Sn predicted-energy arrays; selected before any new material fit",
        alias_computation="identical reconstructed DB and process fit equations may reuse exact predictions; all names retained",
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (
            Path(__file__),Path(fitter.__file__),ROOT/"src/sandbox/property_metric_learning.py",
            ROOT/"scripts/trainable_sandbox_20261017/csv_io_v2.py")},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in input_paths},
        control_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in controls},
        energy_protocol_sha256=energy.sha(training.OUT/"protocol.json"),
        primary=["rbf_base_energy","rbf_closed_energy"],
        process_fit="exact frozen phase2 family-weighted equations/bounds/two starts/max_nfev60",
        source_split="same5 outer families; inner source-held-out shrinkage at0/1/2 anchors",
        common_queries="fixed18, plus all29/23/18 with guarded near duplicates",
        pressure="all unreported; common unmeasured1e-5atm scenario",
        budget="same Si844/Sn850 supplied energies and catalogue; references reused, no new DFT",
        grade="retrospective source-held-out material validation, not model-generated response accuracy")
    path=OUT/"protocol.json"
    if path.exists():
        old=json.loads(path.read_text())
        for key in ("code_hashes","input_hashes","control_hashes","exact_energy_aliases"):
            assert old[key]==payload[key]
    else:energy.write(path,payload)
    return mapping,controls

def prepare():
    mapping,controls=freeze()
    for p in controls:
        dest=OUT/"outer"/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert energy.sha(dest)==energy.sha(p)
        else:shutil.copyfile(p,dest)
    return mapping

def database(element,method):
    key=(element,method)
    if key not in DBS:
        db,records,_=energy.records()
        target=[r for r in records if r["element"]==element]
        data=energy_frame(element);data=data[data.method.eq(method)]
        lookup=dict(zip(data.label_id,data.predicted_crossing_eV))
        if set(lookup)!={r["label_id"] for r in target}:raise RuntimeError("Incomplete energy outputs")
        DBS[key]=reconstruct(db,target,[lookup[r["label_id"]] for r in target])
    return DBS[key]

def point(row,w,target_element,method):
    key=(target_element,method,row.element)
    if key not in fitter.LAYERS:
        db,_=element_database(database(target_element,method),row.element)
        fitter.LAYERS[key]=ImplicitDefectLayer(db,(row.element,))
    layer=fitter.LAYERS[key];s=expit(w[0])
    logT=fitter.L0+(fitter.L1-fitter.L0)*s
    pred=layer.predict(layer.parameters(T=10**logT,Nd=10**w[1],
                       doses=(row.incorporated_cm3,),pO2=1e-5))
    chain=np.zeros((layer.np,2));chain[0,0]=(fitter.L1-fitter.L0)*s*(1-s);chain[1,1]=1.
    return pred,pred.jacobian@chain

def configure():
    fitter.OUT=OUT;fitter.METHODS=METHODS;fitter.freeze=freeze
    fitter.database=database;fitter.point=point;fitter.read_result_csv=read_result_csv
    fitter.DATA=read_result_csv(energy.BASE/"dataset_v1.csv")
    fitter.ROLES=read_result_csv(energy.BASE/"episodes_v1.csv")
    fitter.FITS.clear();fitter.DBS.clear();fitter.LAYERS.clear();DBS.clear()

def worker(method):
    configure();fitter.summarize=lambda:None;fitter.run(method)
    return method

def copy_aliases(mapping):
    for alias,representative in mapping.items():
        if alias==representative:continue
        for family in sorted(fitter.DATA.family.unique()):
            name=f"{fitter.digest(family)}__"
            source=OUT/"outer"/f"{name}{representative}.csv"
            data=read_result_csv(source);data["method"]=alias
            target=OUT/"outer"/f"{name}{alias}.csv"
            data.to_csv(target,index=False)
            choices=json.loads((OUT/"selections"/f"{name}{representative}.json").read_text())
            for row in choices:row["exact_alias_of"]=representative
            energy.write(OUT/"selections"/f"{name}{alias}.json",choices)

def run(workers):
    mapping=prepare();representatives=list(dict.fromkeys(mapping.values()))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for method in pool.map(worker,representatives):print("reference property complete",method,flush=True)
    configure();copy_aliases(mapping);ORIGINAL_SUMMARY()

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=4)
    run(ap.parse_args().workers)
