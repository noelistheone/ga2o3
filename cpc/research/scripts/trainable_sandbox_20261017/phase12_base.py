"""Versioned no-neutral-state shared learners and frozen phase11 controls.

The old module is used in a separate process; its files and results are never
modified. This adapter explicitly records its new definitions and canonical
zero-learning limit (the original RT model, not the neutral extension).
"""
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import sys,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase11_prediction as core
import phase2_energy as energy

OUT=energy.BASE/"phase12_stock_adaptation"
NEW=OUT/"new_base_models"
PARENT=energy.BASE/"phase11_parent_allocation"
REF=PARENT/"solver_budget_sensitivity"
NEW_METHODS={
    "stock_const_noext_RT":dict(field="parent",features="constant",training="RT",thermal=False),
    "stock_rbf_noext_RT":dict(field="parent",features="rbf",training="RT",thermal=False),
    "out_rbf_noext_RT":dict(field="output",features="rbf",training="RT",thermal=False),
    "tf_rbf_noext_RT":dict(field="Tf",features="rbf",training="RT",thermal=False),
}
OLD_CANONICAL=core.canonical
core.METHODS.update(NEW_METHODS)
core.OUT=NEW
FROZEN={}


def canonical(method,config):
    config=dict(config)
    if "reg" in config and config["reg"] is None:config["reg"]=np.inf
    if method in NEW_METHODS and np.isinf(config.get("reg",0.)):
        return "original_RT",{}
    return OLD_CANONICAL(method,config)


core.canonical=canonical


def getfit(families,method,config):
    method,config=canonical(method,config)
    if method in NEW_METHODS:return core.fit(families,method,config)
    if not FROZEN:
        for p in (REF/"fits").glob("*.json"):
            f=json.loads(p.read_text());f["frozen_origin"]=dict(path=str(p.relative_to(ROOT)),sha256=energy.sha(p))
            FROZEN[f["method"],tuple(f["train_families"]),json.dumps(f["config"],sort_keys=True)]=f
        p=REF/"retained_stall_refinement/corrected_fit.json";f=json.loads(p.read_text())
        f["frozen_origin"]=dict(path=str(p.relative_to(ROOT)),sha256=energy.sha(p))
        FROZEN[f["method"],tuple(f["train_families"]),json.dumps(f["config"],sort_keys=True)]=f
    return FROZEN[method,tuple(sorted(families)),json.dumps(energy.clean(config),sort_keys=True)]


def freeze():
    gate=json.loads((OUT/"base_gates.json").read_text());assert gate["passed"]
    paths=[Path(__file__),ROOT/"scripts/trainable_sandbox_20261017/phase12_base_gates.py",
        ROOT/"scripts/trainable_sandbox_20261017/phase11_prediction.py",ROOT/"src/sandbox/parent_allocation.py",
        ROOT/"src/sandbox/temperature_defects.py",ROOT/"src/sandbox/material_profile.py",
        ROOT/"docs/ML_SANDBOX_SOURCE_STOCK_ADAPTATION_20261017.md"]
    inputs=[OUT/"base_gates.json",energy.BASE/"artifact_manifest_phase11.json",
        energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv",
        core.prior.TEMPDIR/"paired_specimens.csv",core.prior.TEMPDIR/"temperature_observations.csv"]
    payload=dict(created_utc=energy.now(),version=1,new_methods=NEW_METHODS,
        configs={m:core.grids(m) for m in NEW_METHODS},
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        objective="RT source-equal squared loss + reg||psi||²;no temperature labels for new basemodels",
        exact_zero_limit="infinite reg for these no-extension methods aliases original_RT",
        optimizer="unchanged two starts/max_nfev60/tolerances1e-7/analyticIFT,allfailuresretained",
        inputs="same actualSi/Sn+reportedH;C absent from catalogue/nonuniform report not homogeneous supply;pO2scenario1e-5;rH1,publishedprofile",
        scope="150 new fitrecords on15three/four-source contexts;frozen oldcontrols retained,not blind,not newDFT/MD")
    p=NEW/"prediction_protocol.json"
    if p.exists():
        old=json.loads(p.read_text())
        for k in ("code_hashes","input_hashes","new_methods","configs"):assert old[k]==energy.clean(payload[k]),k
    else:energy.write(p,payload)


def one(job):
    families,method,config=job
    return getfit(families,method,config)["key"]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=4);a=ap.parse_args();freeze()
    jobs=[]
    for n in (3,4):
        for families in combinations(core.FAMILIES,n):
            for method in NEW_METHODS:
                for config in core.grids(method):
                    if not np.isinf(config.get("reg",0.)):jobs.append((families,method,config))
    assert len(jobs)==150
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(one,jobs):pass
    energy.write(OUT/"base_training_execution.json",dict(created_utc=energy.now(),new_records=len(jobs),
        original_primary_preserved=True,goal_complete=False))
    print("phase12 new base fits completed",len(jobs),flush=True)


if __name__=="__main__":main()
