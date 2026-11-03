"""Freeze complete-temperature source-excluded prediction development evidence."""
from pathlib import Path
import sys,json,shutil

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy

BASE=energy.BASE
OUT=BASE/"phase9_complete_temperature"
DEST=BASE/"artifact_manifest_phase9.json"


def verify(path):
    x=json.loads(path.read_text());entries=x["files"] if isinstance(x,dict) else x
    changed=[r["path"] for r in entries if not (ROOT/r["path"]).is_file() or
             energy.sha(ROOT/r["path"])!=r["sha256"]]
    if changed:raise RuntimeError("Protected artifacts changed: "+str(changed))
    return len(entries)


def main():
    if DEST.exists():
        print("Verified phase9",verify(DEST));return
    parents=("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json",
        "artifact_manifest_phase34.json","artifact_manifest_phase56.json","artifact_manifest_phase7.json",
        "artifact_manifest_phase8.json")
    for file in parents:verify(BASE/file)
    audit=json.loads((OUT/"contract_audit.json").read_text())
    gates=json.loads((OUT/"numerical_gates.json").read_text())
    assert audit["passed"] and not audit["goal_complete"]
    assert audit["fits"]==227 and audit["nested_candidate_evaluations"]==3240
    assert gates["passed"] and len(gates["checks"])==165
    assert json.loads((OUT/"gpu1_bridge_verification.json").read_text())["passed"]
    assert json.loads((OUT/"rendering_check.json").read_text())["passed"]
    protocol=json.loads((OUT/"prediction_protocol.json").read_text())
    for group in ("code_hashes","input_hashes"):
        for file,sha in protocol[group].items():assert energy.sha(ROOT/file)==sha,file
    snap=BASE/"code_snapshot_phase9"
    files=list((ROOT/"src/sandbox").glob("*.py"))
    files+=list((ROOT/"scripts/trainable_sandbox_20261017").glob("phase9*.py"))
    files+=[ROOT/"scripts/trainable_sandbox_20261017"/file for file in (
        "freeze_phase9.py","phase2_energy.py","phase2_property.py","csv_io_v2.py","export_research_pdf.py")]
    files+=[ROOT/"docs"/file for file in ("ML_SANDBOX_FULL_TEMPERATURE_20261017.md",
        "ML_SANDBOX_GOAL_20261017.md","ML_SANDBOX_GOAL_AUDIT_20261017.md",
        "ML_SANDBOX_GOAL_CHECK_PHASE9_20261017.md")]
    for p in files:
        dest=snap/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert energy.sha(dest)==energy.sha(p)
        else:shutil.copyfile(p,dest)
    logdir=OUT/"logs";logdir.mkdir(exist_ok=True)
    for name in ("phase9_numerical_run.log","phase9_joint_run.log"):
        p=BASE/name
        if p.exists():shutil.copyfile(p,logdir/name)
    paths=sorted(p for directory in (OUT,snap) for p in directory.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc")
    payload=dict(stage="phase9_complete_temperature_internal_relation_prediction",
        created_utc=energy.now(),goal_status="active",goal_complete=False,
        scope="immutable profile,full supplied hot/frozen variable-temperature solver,real-label shared E(C),eight controls",
        evidence=dict(specimens=29,source_groups=5,temperature_specimens=7,temperature_sources=2,
            new_physical_specimens=0,numerical_checks=165,cached_fits=227,optimizer_starts=150,
            nonconverged_starts=1,selected_nonconverged_fits=0,state_failures=0),
        results="temperature conditional gain from shared E(C),no consistent originalRT gain,strongtwoanchor line remains better",
        predictive_learning="E0,s trained through full high-temperature stocks and cold charge IFT;effective states,not newHSE labels or newML algorithm",
        physical_limits="localized donor statistics not Mott/metallic limit;unitHall/hydrogen/carbon/pressure assumptions explicit",
        GPU="GPU1 first-orderbridge verified;roots/training CPU;GPU0unused",
        current_production_inference="v4 unchanged,not validated variable-temperature mobility",
        next="constrained internal-parameter adaptation with matching anchor budgets,plus required state/Hall limiting-case audit;retain original29/5endpoint",
        new_DFT_MD=0,new_material_measurements=0,
        parent_manifests={file:energy.sha(BASE/file) for file in parents},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in paths])
    energy.write(DEST,payload)
    print("Frozen phase9",len(paths));print("Manifest SHA256",energy.sha(DEST))


if __name__=="__main__":main()
