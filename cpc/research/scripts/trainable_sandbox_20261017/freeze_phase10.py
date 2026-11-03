"""Freeze local-adaptation prediction evidence, retained stalls and runtime."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy

BASE=energy.BASE;OUT=BASE/"phase10_internal_adaptation"
DEST=BASE/"artifact_manifest_phase10.json"


def verify(path):
    x=json.loads(path.read_text());entries=x["files"] if isinstance(x,dict) else x
    changed=[r["path"] for r in entries if not (ROOT/r["path"]).is_file() or energy.sha(ROOT/r["path"])!=r["sha256"]]
    if changed:raise RuntimeError("Protected artifacts changed: "+str(changed))
    return len(entries)


def main():
    if DEST.exists():print("Verified phase10",verify(DEST));return
    parents=("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json",
        "artifact_manifest_phase34.json","artifact_manifest_phase56.json","artifact_manifest_phase7.json",
        "artifact_manifest_phase8.json","artifact_manifest_phase9.json")
    for name in parents:verify(BASE/name)
    a=json.loads((OUT/"contract_audit.json").read_text())
    assert a["passed"] and a["methods"]==11 and a["saved_outer_rows"]==5874
    assert json.loads((OUT/"numerical_gates.json").read_text())["passed"]
    assert json.loads((OUT/"solver_budget_sensitivity/stalled_refinement_audit.json").read_text())["passed"]
    assert json.loads((OUT/"research_predictor_v2/verification.json").read_text())["passed"]
    assert json.loads((OUT/"rendering_check.json").read_text())["passed"]
    for name in ("protocol.json","research_predictor_v2/protocol.json"):
        p=json.loads((OUT/name).read_text())
        for group in ("code_hashes","input_hashes"):
            for path,sha in p.get(group,{}).items():assert energy.sha(ROOT/path)==sha,path
    api=json.loads((OUT/"research_predictor_v2/verification.json").read_text())
    assert api["API_code_sha256"]==energy.sha(ROOT/"src/sandbox/thermal_predictor.py")
    snap=BASE/"code_snapshot_phase10"
    sources=list((ROOT/"src/sandbox").glob("*.py"))
    sources+=list((ROOT/"scripts/trainable_sandbox_20261017").glob("phase10*.py"))
    sources+=[ROOT/"scripts/trainable_sandbox_20261017"/p for p in (
        "freeze_phase10.py","phase9_joint_prediction.py","phase9_temperature_gates.py",
        "phase2_property.py","phase2_energy.py","csv_io_v2.py","csv_io.py","export_research_pdf.py")]
    sources+=[ROOT/"docs"/p for p in ("ML_SANDBOX_PARAMETER_ADAPTATION_20261017.md",
        "ML_SANDBOX_PARENT_ALLOCATION_NEXT_20261017.md","ML_SANDBOX_GOAL_20261017.md",
        "ML_SANDBOX_GOAL_AUDIT_20261017.md")]
    sources.append(BASE/"CONTINUE_PHASE10_HERE.md")
    for p in sources:
        dest=snap/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert energy.sha(dest)==energy.sha(p)
        else:shutil.copyfile(p,dest)
    logdir=OUT/"logs";logdir.mkdir(exist_ok=True)
    for name in ("phase10_run.log","phase10_audit_run.log","phase10_solver_budget.log","phase10_stalled_refinement.log"):
        p=BASE/name
        if p.is_file():shutil.copyfile(p,logdir/name)
    paths=sorted(p for directory in (OUT,snap) for p in directory.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc")
    payload=dict(stage="phase10_internal_parameter_adaptation_prediction_and_standalone_runtime",
        created_utc=energy.now(),goal_status="active",goal_complete=False,
        main="unchanged actualdose-to-RT29specimens5DOI;temperature7oldSi specimens2DOI supplement",
        methods=11,new_base_fits=92,calibrations=11277,outer_prediction_rows=5874,
        numerical_checks=19,zero_update_replays=129,runtime_replays=447,
        scope="trained shared E(C)+full defect IFT+0/1/2anchor constrained physical adaptation;not new MLalgorithm/microscopic identification",
        outcome="conditional2temperatureanchor improvement,small beyondsame-sharedoffset and not inbothsources;originalRTmain no consistentgain",
        unfavorable="1anchor overfitting;allTquery near tie;V1 nonconverged starts retained;V2 allconvergence failure retained and targeted KKT correction saved",
        runtime="research_predictor_v2,explicitpressure/material/energeticchecksum,no experimental table access;v4unchanged",
        next="gauge-checked shared dopantparent stock allocation,then primary cross-source predictive comparison",
        new_material_specimens=0,new_DFT_MD=0,GPU_training=False,
        parent_manifests={p:energy.sha(BASE/p) for p in parents},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in paths])
    energy.write(DEST,payload);print("Frozen phase10",len(paths));print("Manifest SHA256",energy.sha(DEST))


if __name__=="__main__":main()
