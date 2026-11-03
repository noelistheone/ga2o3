"""Freeze the prediction checkpoint while leaving the scientific goal active."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy

BASE=energy.BASE;OUT=BASE/"phase11_parent_allocation";DEST=BASE/"artifact_manifest_phase11.json"


def verify(path):
    p=json.loads(path.read_text());entries=p["files"] if isinstance(p,dict) else p
    changed=[r["path"] for r in entries if not (ROOT/r["path"]).is_file() or energy.sha(ROOT/r["path"])!=r["sha256"]]
    assert not changed,changed
    return len(entries)


def main():
    if DEST.exists():print("Verified phase11",verify(DEST));return
    parents=("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json",
        "artifact_manifest_phase34.json","artifact_manifest_phase56.json","artifact_manifest_phase7.json",
        "artifact_manifest_phase8.json","artifact_manifest_phase9.json","artifact_manifest_phase10.json")
    checked={p:verify(BASE/p) for p in parents}
    for rel in ("numerical_gates.json","learning_gates.json","zero_replays.json","contract_audit.json",
                "state_diagnostic_audit.json","solver_budget_sensitivity/contract_audit.json",
                "solver_budget_sensitivity/retained_stall_refinement/audit.json","solver_budget_sensitivity/state_diagnostic_audit.json",
                "external_PRB2024/contract_audit.json","rendering_check.json"):
        assert json.loads((OUT/rel).read_text())["passed"],rel
    for rel in ("prediction_protocol.json","numerical_protocol.json","learning_protocol.json"):
        p=json.loads((OUT/rel).read_text())
        for group in ("code_hashes","input_hashes"):
            for path,sha in p.get(group,{}).items():assert energy.sha(ROOT/path)==sha,path
    retained=json.loads((OUT/"solver_budget_sensitivity/sensitivity_audit.json").read_text())
    assert retained["passed"] is False and retained["objective_worse_records"]==1
    snap=BASE/"code_snapshot_phase11"
    sources=list((ROOT/"src/sandbox").glob("*.py"))
    sources+=list((ROOT/"scripts/trainable_sandbox_20261017").glob("phase11*.py"))
    sources += [ROOT/"scripts/trainable_sandbox_20261017"/p for p in
        ("freeze_phase11.py","phase9_joint_prediction.py","phase9_temperature_gates.py","phase2_property.py",
         "phase2_energy.py","csv_io_v2.py","csv_io.py","export_research_pdf.py")]
    sources += [ROOT/"docs"/p for p in ("ML_SANDBOX_PARENT_LEARNING_20261017.md",
        "ML_SANDBOX_PARENT_ALLOCATION_THEORY_20261017.md","ML_SANDBOX_PARENT_ALLOCATION_THEORY_20261017.pdf",
        "ML_SANDBOX_GOAL_20261017.md","ML_SANDBOX_GOAL_AUDIT_20261017.md")]
    sources += [BASE/p for p in ("CONTINUE_PHASE11_HERE.md","goal_requirements_audit.json","progress.json")]
    for p in sources:
        dst=snap/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert energy.sha(dst)==energy.sha(p)
        else:shutil.copyfile(p,dst)
    logs=OUT/"logs";logs.mkdir(exist_ok=True)
    for p in BASE.glob("phase11*.log"):
        dst=logs/p.name
        if dst.exists():assert energy.sha(dst)==energy.sha(p)
        else:shutil.copyfile(p,dst)
    paths=sorted(p for folder in (OUT,snap) for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc")
    audit=json.loads((OUT/"solver_budget_sensitivity/contract_audit.json").read_text())
    payload=dict(created_utc=energy.now(),stage="shared_parent_inventory_learning_and_doped_property_prediction",
        goal_status="active",goal_complete=False,primary="original29Si/Sn specimens5DOI actualdose-to-RT Hall,0/1/2anchors andcommon18",
        supplementary="7sameSi specimens/2DOI temperature curves;3new source SIMS/Hall conditions with missingprocessinputs",
        models=14,original_cached_fits=540,original_new_optimizer_starts=660,
        shared_state_learning=True,inventory_and_learning_checks=175,zero_update_replays=86,
        common_cap_refinement="all75records with any failedstart,60to180;allV1failurespreserved",
        remaining_best_nonconverged_fits=audit["selected_nonconverged_fits"],
        remaining_outer_selected_nonconverged_fits=audit["selected_outer_nonconverged_fits"],
        retained_failed_gate="commoncap objective-not-worse fails1case by4.79e-8;targeted same-objective correction with IFT check keeps allouter selections/predictions unchanged",
        outcome="no stable original-RT gain established for shared inventory field;do not infer all internalphysicallearning lacksvalue",
        scope="retrospective conditional research checkpoint,not a newmicroscopicmechanism,full-transport orrareearth validation",
        new_physical_experiments=0,new_DFT_MD=0,GPU_training=False,authorized_GPU=1,
        previous_checked=checked,parent_manifests={p:energy.sha(BASE/p) for p in parents},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in paths])
    energy.write(DEST,payload)
    print("Frozen phase11",len(paths));print("Manifest SHA256",energy.sha(DEST))


if __name__=="__main__":main()
