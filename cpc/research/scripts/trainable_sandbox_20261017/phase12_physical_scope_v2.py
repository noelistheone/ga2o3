"""Enforce cached zero-update Hall-domain refusals and preserve V1 results.

Nonzero physical updates already use SourceStockModel's observation gate.
The V1 cached zero branch omitted its false physical-validity flag. No fitted
function, scalar profile, data split or numerical parameter is changed here.
"""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase12_prediction as v1

OUT=v1.OUT/"physical_scope_v2";CACHE={}


def corrected_episode(f,held,kind):
    key=v1.episode_key(f,held,kind)
    if key in CACHE:return CACHE[key]
    source=v1.OUT/"episodes"/(key+".json")
    ep=json.loads(source.read_text());changes=[]
    if kind in ("stock","Tf"):
        for r in ep["results"]:
            if r["shift"]!=0:continue
            invalid=[row for row in r["records"] if row.get("physical_valid") is False]
            if not invalid:continue
            anchor_invalid=any(row["role"]=="anchor" for row in invalid)
            for row in r["records"]:
                if row.get("physical_valid") is False or anchor_invalid:
                    if row["predicted"] is not None:
                        changes.append(dict(budget=r["budget"],lam=r["lam"],obs_id=row["obs_id"],
                            original_prediction=row["predicted"],reason="invalid cached physical anchor" if anchor_invalid else "cached physical Hall/domain refusal"))
                    row["predicted"]=None
                    row["error"]="physical_zero_update_anchor_refusal" if anchor_invalid else "physical_zero_update_domain_refusal"
            if anchor_invalid:r["anchor_complete"]=False
            q=[row for row in r["records"] if row["role"]=="query"]
            r["MAE"]=(float(np.mean([abs(row["predicted"]-row["observed"]) for row in q]))
                if all(row["predicted"] is not None for row in q) else None)
    ep["scope_correction"]=dict(original_path=str(source.relative_to(ROOT)),original_sha256=v1.energy.sha(source),changes=changes)
    if changes:v1.energy.write(OUT/"episodes"/(key+".json"),ep)
    CACHE[key]=ep
    return ep


def main():
    assert (v1.OUT/"prediction_execution.json").exists(),"Wait for the original job to finish"
    protocol=dict(version=2,created_utc=v1.energy.now(),code_sha256=v1.energy.sha(Path(__file__)),
        original_prediction_protocol_sha256=v1.energy.sha(v1.OUT/"prediction_protocol.json"),
        reason="finite raw free-electron predictions may violate the single-electron Hall condition;cached zero update must preserve physical refusals",
        policy="stock/Tf zero physicalfalse -> explicitrefusal;invalid physicalanchor invalidates episode;empirical outputs retain separate nonphysical scope",
        invariants="allbasefits,profiles,nonzero evaluations,roles and grids unchanged;reselect using corrected training-source scores only")
    p=OUT/"scope_protocol.json"
    if p.exists():assert json.loads(p.read_text())["code_sha256"]==protocol["code_sha256"]
    else:v1.energy.write(p,protocol)
    write=v1.energy.write;csvwrite=v1.energy.csvwrite
    def route(p):
        try:r=Path(p).relative_to(v1.OUT/"models")
        except ValueError:return p
        return OUT/"models"/r
    def routed_write(p,x):return write(route(p),x)
    def routed_csv(p,x):return csvwrite(route(p),x)
    v1.energy.write=routed_write;v1.energy.csvwrite=routed_csv;v1.episode=corrected_episode
    before=time.perf_counter()
    for model in v1.MODELS:v1.selected(model)
    changes=[dict(episode_key=ep["key"],**change) for ep in CACHE.values() for change in ep["scope_correction"]["changes"]]
    write(OUT/"changes.json",changes)
    write(OUT/"execution.json",dict(created_utc=v1.energy.now(),models=len(v1.MODELS),
        cached_episodes=len(CACHE),changed_episodes=sum(bool(ep["scope_correction"]["changes"]) for ep in CACHE.values()),
        changed_prediction_entries=len(changes),seconds=time.perf_counter()-before,
        new_physical_fits=0,new_scalar_fits=0,original_artifacts_preserved=True))
    print("physical scope V2 completed",len(changes),"entries",flush=True)


if __name__=="__main__":main()
