"""Common 180-evaluation sensitivity for every fit with a failed V1 start.

Keeps the original 60-evaluation fit, all failures and all V1 predictions.
No change to physical equations, data, initial states, bounds or tolerances.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys, json, shutil, argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase11_prediction as study
import phase2_energy as energy

PARENT=study.OUT
OUT=PARENT/"solver_budget_sensitivity"
LEAST_SQUARES=study.least_squares


def extended_least_squares(*args,**kwargs):
    assert kwargs["max_nfev"]==60
    kwargs["max_nfev"]=180
    return LEAST_SQUARES(*args,**kwargs)


def one(source):
    old=json.loads(source.read_text())
    study.OUT=OUT;study.least_squares=extended_least_squares
    fit=study.fit(old["train_families"],old["method"],old["config"])
    fit["solver_sensitivity"]=dict(original_path=str(source.relative_to(ROOT)),
        original_sha256=energy.sha(source),max_nfev=180,original_max_nfev=60,
        protocol_sha256=energy.sha(OUT/"sensitivity_protocol.json"))
    energy.write(OUT/"fits"/(fit["key"]+".json"),fit)
    return fit["key"]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=4);args=ap.parse_args()
    files=sorted((PARENT/"fits").glob("*.json"));assert len(files)==540
    affected=[]
    for p in files:
        f=json.loads(p.read_text())
        if any(not t["success"] for t in f["trials"]):affected.append(p)
    OUT.mkdir(exist_ok=True);(OUT/"fits").mkdir(exist_ok=True)
    payload=dict(created_utc=energy.now(),version=1,code_sha256=energy.sha(Path(__file__)),
        parent_protocol_sha256=energy.sha(PARENT/"prediction_protocol.json"),
        parent_fit_sha256={p.name:energy.sha(p) for p in files},
        affected_keys=[p.stem for p in affected],selection="all fits with any unsuccessful start,not selected by held-out errors",
        change="max_nfev60 to180 for both original starts;same tolerances,bounds,objective and implicit derivatives",
        inference="reselect all model/anchor settings using training sources only; retain V1 and composite V2 outputs",
        scope="optimization sensitivity,not a new physical model;composite audit costs include inherited fits,report new-refinement costs separately")
    pp=OUT/"sensitivity_protocol.json"
    if pp.exists():
        prior=json.loads(pp.read_text());assert prior["code_sha256"]==payload["code_sha256"]
        assert prior["parent_fit_sha256"]==payload["parent_fit_sha256"]
    else:energy.write(pp,payload)
    for name in ("prediction_protocol.json","numerical_protocol.json","learning_protocol.json",
                 "numerical_gates.json","learning_gates.json","zero_replays.json"):
        dst=OUT/name
        if not dst.exists():shutil.copyfile(PARENT/name,dst)
        assert energy.sha(dst)==energy.sha(PARENT/name)
    affected_set={p.stem for p in affected}
    for p in files:
        if p.stem not in affected_set:
            dst=OUT/"fits"/p.name
            if not dst.exists():shutil.copyfile(p,dst)
            assert energy.sha(dst)==energy.sha(p)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        keys=list(pool.map(one,affected))
    # Selection reads the complete refined-or-inherited fit map; no fitting here.
    lookup={}
    for p in (OUT/"fits").glob("*.json"):
        f=json.loads(p.read_text())
        lookup[(f["method"],tuple(f["train_families"]),json.dumps(f["config"],sort_keys=True))]=f
    assert len(lookup)==540
    def lookup_fit(families,method,config):
        method,config=study.canonical(method,config)
        return lookup[method,tuple(sorted(families)),json.dumps(energy.clean(config),sort_keys=True)]
    study.OUT=OUT;study.fit=lookup_fit
    for method in study.METHODS:study.selected(method)
    refined=[f for f in lookup.values() if f["key"] in keys]
    trials=[t for f in refined for t in f["trials"]]
    energy.write(OUT/"execution.json",dict(created_utc=energy.now(),affected_records=len(keys),new_starts=len(trials),
        nonconverged_starts=sum(not t["success"] for t in trials),
        state_failures=sum(len(f["state_failures"]) for f in refined),
        selected_nonconverged=sum(not min([t for t in f["trials"] if t.get("loss") is not None],key=lambda t:t["loss"])["success"] for f in refined),
        new_fit_seconds=sum(f["seconds"] for f in refined),
        root_counters={k:sum(f["counters"][k] for f in refined) for k in
            ("root_calls","hot_solve_calls","hot_cache_hits","root_evaluations")}))
    print("refinement complete",len(keys),flush=True)


if __name__=="__main__":main()
