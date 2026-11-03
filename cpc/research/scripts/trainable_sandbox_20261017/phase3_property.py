"""Real dopant->carrier source validation of property-metric energy learners."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,os,json,argparse
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

OUT=metric.OUT/"real_property"
METHODS=("oracle_full",*metric.SPECS,"kernel_energy_select_energy")
DBS={}
ORIGINAL_SUMMARY=fitter.summarize

def freeze():
    metric.frozen()
    path=OUT/"protocol.json"
    model_paths=[metric.OUT/"energy_outer"/f"{e}.csv" for e in ("Si","Sn")]
    for p in model_paths:
        if not p.exists():raise RuntimeError("Whole-element fitting is not complete for "+p.stem)
    payload=dict(version=1,created_utc=energy.now(),methods=METHODS,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in
            (Path(__file__),Path(fitter.__file__),ROOT/"src/sandbox/property_metric_learning.py",
             ROOT/"scripts/trainable_sandbox_20261017/csv_io_v2.py")},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in model_paths},
        training_protocol_sha256=energy.sha(metric.OUT/"protocol.json"),
        fit="exact frozen phase2_property fitting equations, bounds, two starts, max_nfev60, source-equal residuals",
        modification="all methods use the phase3 exact absent-impurity catalogue; prediction/IFT checked against legacy",
        primary_training_loss_contrast=["kernel_energy","kernel_diagonal_l1","kernel_full_l1"],
        secondary="fixed-strength sensitivity and energy-selected hyperparameter control are all reported",
        actual_data="same29samples/5sources; same fixed0/1/2 source anchor roles",
        energy_training="whole target element excluded, no actual query carrier labels",
        process_training="outer entire source excluded, inner source exclusion for offset shrinkage",
        energy_budget="same provided library and missing Si29/Sn23 entries; known catalogue/reference states",
        missing_inputs="all29 missing pO2, fixed unmeasured1e-5atm; growth-temperature proxies unchanged",
        outcome="measured carrier logMAE, common18queries, per-source and numeric/physical coverage",
        validation="retrospective; synthetic property-metric training is not material validation",
        gpu="no GPU required for these small kernel/root fits")
    if path.exists():
        old=json.loads(path.read_text())
        assert old["code_hashes"]==payload["code_hashes"]
        assert old["input_hashes"]==payload["input_hashes"]
    else:energy.write(path,payload)

def database(element,method):
    key=(element,method)
    if key in DBS:return DBS[key]
    db,r,p=energy.records()
    if method!="oracle_full":
        rr=[v for v in r if v["element"]==element]
        data=read_result_csv(metric.OUT/"energy_outer"/f"{element}.csv")
        data=data[data.method.eq(method)]
        lookup=dict(zip(data.label_id,data.predicted_crossing_eV))
        if set(lookup)!={v["label_id"] for v in rr}:raise RuntimeError("Missing energy predictions")
        db=reconstruct(db,rr,[lookup[v["label_id"]] for v in rr])
    DBS[key]=db
    return db

def point(row,w,target_element,method):
    key=(target_element,method,row.element)
    if key not in fitter.LAYERS:
        db,indices=element_database(database(target_element,method),row.element)
        fitter.LAYERS[key]=ImplicitDefectLayer(db,(row.element,))
    layer=fitter.LAYERS[key];s=expit(w[0])
    logT=fitter.L0+(fitter.L1-fitter.L0)*s
    pars=layer.parameters(T=10**logT,Nd=10**w[1],doses=(row.incorporated_cm3,),pO2=1e-5)
    pred=layer.predict(pars)
    chain=np.zeros((layer.np,2));chain[0,0]=(fitter.L1-fitter.L0)*s*(1-s);chain[1,1]=1.
    return pred,pred.jacobian@chain

def configure():
    fitter.OUT=OUT;fitter.METHODS=METHODS
    fitter.freeze=freeze;fitter.database=database;fitter.point=point
    fitter.read_result_csv=read_result_csv
    fitter.DATA=read_result_csv(energy.BASE/"dataset_v1.csv")
    fitter.ROLES=read_result_csv(energy.BASE/"episodes_v1.csv")
    fitter.FITS.clear();fitter.DBS.clear();fitter.LAYERS.clear();DBS.clear()

def worker(method):
    configure();fitter.summarize=lambda:None
    fitter.run(method)
    return method

def run(workers):
    freeze()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for method in pool.map(worker,METHODS):print("real metric complete",method,flush=True)
    configure();ORIGINAL_SUMMARY()

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=4)
    a=ap.parse_args();run(a.workers)
