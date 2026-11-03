"""Evaluate acquired energy libraries on the real source-held-out carrier task.

The existing frozen phase2_property fitter is reused without editing it. Each
replayed energy-acquisition path was selected without carrier measurement labels.
Only the training source labels calibrate Tf, Nd and anchor shrinkage.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os,sys,json,shutil,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.energy_learning import reconstruct
from sandbox.goal_acquisition import condition_coordinate
from csv_io import read_result_csv
import phase2_energy as energy
import phase2_acquisition as acquisition
import phase2_property as property_fit

OUT=energy.OUT/"acquired_property"
SPECS={}
for k in (2,4):
    SPECS[f"ET_goal_k{k}"]=dict(source="learned",model="extra_trees",policy="goal",seed=0,budget=k)
    SPECS[f"ET_variance_k{k}"]=dict(source="learned",model="extra_trees",policy="energy_variance",seed=0,budget=k)
    SPECS[f"mean_goal_k{k}"]=dict(source="diagonal_A",model="global_mean",policy="goal",seed=0,budget=k)
    for seed in (0,1):
        SPECS[f"ET_random{seed}_k{k}"]=dict(source="learned",model="extra_trees",policy="random",seed=seed,budget=k)

def freeze():
    energy.frozen()
    p=OUT/"protocol.json"
    payload=dict(version=1,created_utc=energy.now(),
        code_hashes={str(q.relative_to(ROOT)):energy.sha(q) for q in
            (Path(__file__),Path(property_fit.__file__),Path(acquisition.__file__),
             ROOT/"scripts/trainable_sandbox_20261017/phase2_acquisition_ablation.py")},
        parent_property_protocol_sha256=energy.sha(energy.OUT/"property/protocol.json"),
        specs=SPECS,worker_processes=2,selection_uses_measured_query_carriers=False,
        data="same29records/5families and 0/1/2 nH anchors",
        input_energy_counts="base Si844/Sn850 plus exactly k=2 or4 previously masked entries",
        oracle_and_zero_energy_acquisition="copied unchanged from phase2_property full/mean/ET cases",
        fit_and_adaptation="reuse exact frozen phase2_property equations/bounds/starts/inner source selection",
        covariance_ablation="simple global crossing mean plus independent original-A variance is no-feature baseline",
        random_scope="only two predetermined random seeds0/1 for real task; full32seed virtual replay is separate",
        retrospective=True,
        limit="same-property-source cohorts/process assumptions; not new DFT or independent material confirmation")
    if p.exists():assert json.loads(p.read_text())["code_hashes"]==payload["code_hashes"]
    else:energy.write(p,payload)

def acquired_database(element,method):
    spec=SPECS[method]
    db,rr,_=energy.records()
    targets=[r for r in rr if r["element"]==element]
    if spec["source"]=="diagonal_A":
        path=energy.OUT/"acquisition_ablation/states"/f"{element}__global_mean__diagonal_A.npz"
        x=np.load(path);mean=x["means"][spec["budget"]]
    else:
        db,targets,mean,cov=acquisition.prior(element,spec["model"])
        p=read_result_csv(energy.OUT/"acquisition/replays"/
            f'{element}__{spec["model"]}__{spec["policy"]}__{spec["seed"]}.csv')
        raw=p[p.budget.eq(spec["budget"])].acquired_labels.iloc[0]
        labels=str(raw).split("|")
        if len(labels)!=spec["budget"]:raise RuntimeError("Wrong acquisition budget")
        index={r["label_id"]:i for i,r in enumerate(targets)}
        for name in labels:
            i=index[name]
            mean,cov=condition_coordinate(mean,cov,i,targets[i]["crossing_eV"])
    return reconstruct(db,targets,mean)

def configure():
    property_fit.OUT=OUT
    property_fit.METHODS=tuple(SPECS)
    property_fit.database=acquired_database
    property_fit.freeze=freeze
    property_fit.FITS.clear();property_fit.DBS.clear();property_fit.LAYERS.clear()

def worker(method):
    configure()
    # Each worker writes unique method/fold files. Aggregate only in the parent,
    # after all workers finish, to avoid readers seeing a partial shared CSV.
    property_fit.summarize=lambda:None
    property_fit.run(method)
    return method

def run(workers):
    freeze()
    # Add fixed previous references without recomputing or changing their values.
    target=OUT/"outer";target.mkdir(parents=True,exist_ok=True)
    for p in (energy.OUT/"property/outer").glob("*.csv"):
        if any(p.name.endswith("__"+m+".csv") for m in ("oracle_full","global_mean","extra_trees")):
            q=target/p.name
            if not q.exists():shutil.copy2(p,q)
    methods=list(SPECS)
    if workers==1:
        for m in methods:worker(m)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for m in pool.map(worker,methods):print("real task complete",m,flush=True)
    # Configure a fresh module instance in the parent; its summary is unpatched.
    configure();property_fit.summarize()
    print("acquired energies -> real carrier validation complete",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=2)
    a=p.parse_args();run(a.workers)
