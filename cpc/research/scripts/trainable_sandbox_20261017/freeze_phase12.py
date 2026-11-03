"""Freeze the source-adaptation prediction checkpoint; keep the goal active."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy

BASE=energy.BASE;OUT=BASE/"phase12_stock_adaptation";DEST=BASE/"artifact_manifest_phase12.json"


def verify(path):
    p=json.loads(path.read_text());rows=p["files"] if isinstance(p,dict) else p
    changed=[r["path"] for r in rows if not (ROOT/r["path"]).is_file() or energy.sha(ROOT/r["path"])!=r["sha256"]]
    assert not changed,changed
    return len(rows)


def main():
    if DEST.exists():print("Verified phase12",verify(DEST));return
    parents=("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json",
        "artifact_manifest_phase34.json","artifact_manifest_phase56.json","artifact_manifest_phase7.json",
        "artifact_manifest_phase8.json","artifact_manifest_phase9.json","artifact_manifest_phase10.json",
        "artifact_manifest_phase11.json")
    checked={p:verify(BASE/p) for p in parents}
    checks=("base_gates.json","adaptation_gates.json","physical_scope_v2/contract_audit.json",
        "physical_scope_v2/affine_control_audit.json","full_property_chain/audit.json",
        "full_property_chain/head_capacity_audit.json","full_property_chain/head_capacity_replay_audit.json",
        "research_runtime/verification.json","research_runtime/transport_replay_audit.json","rendering_check.json")
    for rel in checks:assert json.loads((OUT/rel).read_text())["passed"],rel
    for pp in OUT.rglob("*protocol.json"):
        p=json.loads(pp.read_text())
        for group in ("code_hashes","input_hashes"):
            for path,sha in p.get(group,{}).items():assert energy.sha(ROOT/path)==sha,(pp,path)
    s=json.loads((OUT/"physical_scope_v2/scope_protocol.json").read_text())
    assert energy.sha(ROOT/"scripts/trainable_sandbox_20261017/phase12_physical_scope_v2.py")==s["code_sha256"]
    assert energy.sha(OUT/"prediction_protocol.json")==s["original_prediction_protocol_sha256"]
    r=json.loads((OUT/"research_runtime/protocol.json").read_text())
    for filename,key in (("scripts/trainable_sandbox_20261017/phase12_runtime.py","code_sha256"),
        ("src/sandbox/source_stock_model.py","inference_code_sha256"),
        ("src/sandbox/scalar_profile_adaptation.py","adaptation_code_sha256")):
        assert energy.sha(ROOT/filename)==r[key],filename
    a=json.loads((OUT/"physical_scope_v2/contract_audit.json").read_text())
    assert a["V2_selection_changes"]==0 and a["scope_corrected_prediction_entries"]==25
    assert a["new_base_fits"]==150 and a["models"]==20 and a["failed_outer_queries"]==0
    d=json.loads((OUT/"goal_direction_check.json").read_text())
    assert d["system_goal_status"]=="active" and d["direction_correct"] and not d["sufficient_to_complete_goal"]
    snap=BASE/"code_snapshot_phase12"
    sources=list((ROOT/"src/sandbox").glob("*.py"))
    sources+=list((ROOT/"scripts/trainable_sandbox_20261017").glob("*.py"))
    sources+=[ROOT/p for p in ("README.md","AGENTS.md",
        "docs/ML_SANDBOX_SOURCE_STOCK_ADAPTATION_20261017.md","docs/ML_SANDBOX_TWO_CLOSURES_NEXT_20261017.md",
        "docs/ML_SANDBOX_GOAL_20261017.md","docs/ML_SANDBOX_GOAL_AUDIT_20261017.md")]
    sources+=[BASE/p for p in ("CONTINUE_PHASE12_HERE.md","goal_requirements_audit.json","progress.json")]
    for p in sources:
        target=snap/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert energy.sha(target)==energy.sha(p),target
        else:shutil.copyfile(p,target)
    logs=OUT/"logs";logs.mkdir(exist_ok=True)
    for p in BASE.glob("phase12*.log"):
        target=logs/p.name
        if target.exists():assert energy.sha(target)==energy.sha(p),target
        else:shutil.copyfile(p,target)
    files=sorted(p for folder in (OUT,snap) for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc")
    payload=dict(created_utc=energy.now(),stage="trainable_ML_Sandbox_internal_source_adaptation_and_full_property_chain",
        goal_status="active",goal_complete=False,
        primary="original29Si/Sn specimens5DOI,0/1/2specimenanchors,29/23/18queries andcommon18",
        full_direction="dopant/dose/process->learned internal relation->conservative Sandbox->properties",
        models=20,new_base_fits=150,new_base_starts=300,nonconverged_starts=73,best_nonconverged=18,
        selected_outer_nonconverged=0,episodes=900,inner_candidate_evaluations=12420,outer_saved_rows=1740,
        gates=228,zero_update_replays=1305,retained_scope_bug="V1zero-cacheHallflag;V2corrects3episodes25entries,noouterchoice/predictionchange",
        standalone_bundles=135,carrier_replay_rows=783,paired_transport_replay_rows=99,
        paired_mobility_specimens=5,paired_mobility_DOIs=2,independent_sigma_labels=0,
        outcome="conditional1anchorinternalstocksignal mainlyTEOS;not stable0/2,no stableRBFincrement;fixedCT headcapacity limitsfullmu",
        next_candidate_implemented=False,new_DFT_MD=0,new_physical_experiments=0,GPU_training=False,authorized_GPU=1,
        parents_checked=checked,parent_manifest_sha256={p:energy.sha(BASE/p) for p in parents},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in files])
    energy.write(DEST,payload)
    print("Frozen phase12",len(files));print("Manifest SHA256",energy.sha(DEST))


if __name__=="__main__":main()
