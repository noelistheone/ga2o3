"""Freeze the shared-donor prediction evidence, retaining the active goal."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/"phase13_donor_mixture";DEST=BASE/"artifact_manifest_phase13.json"


def verify(path):
    p=json.loads(path.read_text());rows=p["files"] if isinstance(p,dict) else p
    bad=[r["path"] for r in rows if not (ROOT/r["path"]).is_file() or energy.sha(ROOT/r["path"])!=r["sha256"]]
    assert not bad,bad
    return len(rows)


def main():
    if DEST.exists():print("Verified phase13",verify(DEST));return
    names=("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json",
        "artifact_manifest_phase34.json","artifact_manifest_phase56.json","artifact_manifest_phase7.json",
        "artifact_manifest_phase8.json","artifact_manifest_phase9.json","artifact_manifest_phase10.json",
        "artifact_manifest_phase11.json","artifact_manifest_phase12.json")
    checked={p:verify(BASE/p) for p in names}
    for rel in ("numerical_gates.json","model_gates.json","validation/contract_audit.json",
        "solver_budget_v2/refinement_gate.json","solver_budget_v2/validation/contract_audit.json",
        "research_runtime/verification.json","selection_sensitivity.json","rendering_check.json"):
        assert json.loads((OUT/rel).read_text())["passed"],rel
    for rel in ("numerical_protocol.json","prediction_protocol.json","solver_budget_v2/protocol.json",
        "solver_budget_v2/metadata_finish_protocol.json","research_runtime/protocol.json"):
        p=json.loads((OUT/rel).read_text())
        for group in ("code_hashes","input_hashes"):
            for filename,sha in p.get(group,{}).items():assert energy.sha(ROOT/filename)==sha,(rel,filename)
    for filename,sha in json.loads((OUT/"model_gates.json").read_text())["code_hashes"].items():
        assert energy.sha(ROOT/filename)==sha,filename
    r=json.loads((OUT/"research_runtime/protocol.json").read_text())
    assert energy.sha(ROOT/"scripts/trainable_sandbox_20261017/phase13_runtime.py")==r["code_sha256"]
    assert energy.sha(ROOT/"src/sandbox/donor_mixture_model.py")==r["inference_code_sha256"]
    finished=json.loads((OUT/"solver_budget_v2/metadata_finish_protocol.json").read_text())
    assert energy.sha(ROOT/"scripts/trainable_sandbox_20261017/phase13_refinement_finish.py")==finished["code_sha256"]
    a=json.loads((OUT/"solver_budget_v2/validation/contract_audit.json").read_text())
    assert a["selected_outer_nonconverged"]==0 and a["missing_primary_queries"]==0
    assert a["new_fit_records"]==405 and a["actual_total_optimizer_starts"]==333
    assert "duplicate" in (BASE/"phase13_refinement.log").read_text().lower() or "multiple values" in (BASE/"phase13_refinement.log").read_text()
    assert json.loads((OUT/"goal_direction_check.json").read_text())["system_goal_status"]=="active"
    sources=list((ROOT/"src/sandbox").glob("*.py"))+list((ROOT/"scripts/trainable_sandbox_20261017").glob("*.py"))
    sources+=[ROOT/p for p in ("README.md","AGENTS.md","docs/ML_SANDBOX_GOAL_20261017.md",
        "docs/ML_SANDBOX_GOAL_AUDIT_20261017.md","docs/ML_SANDBOX_DONOR_MIXTURE_THEORY_20261017.md",
        "docs/ML_SANDBOX_TRANSPORT_UNIT_DERIVATION_20261017.md")]
    sources+=[BASE/p for p in ("CONTINUE_PHASE13_HERE.md","progress.json","goal_requirements_audit.json")]
    snap=BASE/"code_snapshot_phase13"
    for path in sources:
        dst=snap/path.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert energy.sha(dst)==energy.sha(path),dst
        else:shutil.copyfile(path,dst)
    logs=OUT/"logs";logs.mkdir(exist_ok=True)
    for path in BASE.glob("phase13*.log"):
        dst=logs/path.name
        if dst.exists():assert energy.sha(dst)==energy.sha(path),dst
        else:shutil.copyfile(path,dst)
    files=sorted(p for folder in (OUT,snap) for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc")
    payload=dict(created_utc=energy.now(),stage="shared_donor_fraction_ML_Sandbox_property_prediction",
        goal_status="active",goal_complete=False,primary="original29Si/Sn5DOI,0/1/2sampleanchors,common18",
        supplementary="7existingSi specimens/2DOI;no new independent samples",methods=17,fit_records=405,
        physical_fits=165,initial_starts=330,additional_refinement_starts=3,actual_total_starts=333,
        numerical_checks=742,inner_scores=7620,primary_saved_rows=1479,temperature_saved_rows=7599,
        independent_bundles=165,independent_carrier_rows=957,independent_paired_transport_rows=121,
        zero_anchor_signal="fraction+constantbinding .263339 vsoriginal .266289;not maintained1/2;oldE(C)bettertemperature",
        stable_primary_increment_established=False,new_joint_EC_fraction_trained=False,
        new_transport_model_trained=False,new_physical_measurements=0,new_DFT_MD=0,authorized_GPU=1,GPU_training=False,
        retained_failures="original2unconvergedstarts/onefit;refinement metadataduplicatekey,finish wrapperdoesnotmodifynumericfit",
        paired_mobility_specimens=5,paired_mobility_DOIs=2,independent_sigma_labels=0,
        parents_checked=checked,parent_sha256={p:energy.sha(BASE/p) for p in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in files])
    energy.write(DEST,payload)
    print("Frozen phase13",len(files));print("Manifest SHA256",energy.sha(DEST))


if __name__=="__main__":main()
