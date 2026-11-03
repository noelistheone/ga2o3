"""New joint-statistics fits with immutable phase13 controls and same tasks."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase13_prediction as core
import phase13_refinement as parent
import phase2_energy as energy

OUT=energy.BASE/"phase14_joint_statistics"
NEW_METHODS={
    "fraction_screen_constant_joint":dict(fraction="constant",binding="screen",training="joint"),
    "fraction_screen_joint":dict(fraction="dose",binding="screen",training="joint"),
    "output_screen_joint":dict(fraction="one",binding="screen",output="linear",training="joint"),
}
OLD_FIT=core.fit
core.METHODS.update(NEW_METHODS);core.OUT=OUT
BANK={}


def oldbank():
    global BANK
    if not BANK:
        BANK={(f["method"],tuple(sorted(f["train_families"])),json.dumps(f["config"],sort_keys=True)):f
            for f in parent.load_composite().values()}
    return BANK


def getfit(families,method,config):
    if method in NEW_METHODS:return OLD_FIT(families,method,config)
    return oldbank()[method,tuple(sorted(families)),json.dumps(config,sort_keys=True)]


core.fit=getfit


def freeze():
    gate=json.loads((OUT/"model_gates.json").read_text());assert gate["passed"]
    for path,sha in gate["code_hashes"].items():assert energy.sha(ROOT/path)==sha,path
    paths=(Path(__file__),ROOT/"src/sandbox/donor_mixture_model.py",ROOT/"src/sandbox/donor_mixture.py",
        ROOT/"scripts/trainable_sandbox_20261017/phase13_prediction.py",
        ROOT/"docs/ML_SANDBOX_JOINT_STATISTICS_20261017.md")
    inputs=(OUT/"model_gates.json",energy.BASE/"artifact_manifest_phase13.json",energy.BASE/"dataset_v1.csv",
        energy.BASE/"episodes_v1.csv",core.prior.prior.TEMPDIR/"paired_specimens.csv",
        core.prior.prior.TEMPDIR/"temperature_observations.csv")
    payload=dict(created_utc=energy.now(),version=1,new_methods=NEW_METHODS,all_methods=core.METHODS,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        inherited_controls=[dict(key=f["key"],**f["composite_origin"]) for f in oldbank().values()],
        task="unchanged29Si/Sn5DOI/0-1-2sampleanchors/29-23-18queries/common18",
        supplementary="same7Si,142nreads;RTselectedlambda andoldtemperatureanchors/guards",
        new_fit_records=45,new_optimizer_starts=90,optimizer="samephase13twoinitializations/max_nfev120/analyticIFT/tolerances1e-7",
        loss="same sourceequalRTsquared +.1source/specimen/readequal<=275Kcurve squared",
        selection="sameinnerwholeDOI and7lambda grid;no outerquerylabels used;retrospective",
        physical_scope="same energy/profile/H/pO2scenario/rH1;fixedinputsubgroups;notMott/spectrum ornewDFTlabels",
        unchanged_transport=True,new_physical_experiments=0,new_DFT_MD=0,goal_complete=False)
    pp=OUT/"prediction_protocol.json"
    if pp.exists():
        old=json.loads(pp.read_text())
        for k in ("code_hashes","input_hashes","new_methods","all_methods","inherited_controls"):assert old[k]==energy.clean(payload[k]),k
    else:energy.write(pp,payload)


def composite():
    rows={f["key"]:f for f in oldbank().values()}
    for path in (OUT/"fits").glob("*.json"):
        f=json.loads(path.read_text());f["composite_origin"]=dict(path=str(path.relative_to(ROOT)),sha256=energy.sha(path))
        rows[f["key"]]=f
    return rows


def worker(job):return getfit(*job)["key"]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=4);a=ap.parse_args();freeze()
    jobs=[(fam,m,{}) for n in (3,4) for fam in combinations(core.FAMILIES,n) for m in NEW_METHODS];assert len(jobs)==45
    energy.write(OUT/"training_execution_started.json",dict(created_utc=energy.now(),new_jobs=45,workers=a.workers))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(worker,jobs):pass
    energy.write(OUT/"training_execution_complete.json",dict(created_utc=energy.now(),new_jobs=45,goal_complete=False))
    for method in core.METHODS:core.select(method)
    energy.write(OUT/"selection_execution_complete.json",dict(created_utc=energy.now(),methods=len(core.METHODS),goal_complete=False))


if __name__=="__main__":main()
