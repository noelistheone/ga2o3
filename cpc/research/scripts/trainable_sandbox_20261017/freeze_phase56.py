"""Freeze new reference-learning evidence and predictor deliverables."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy

BASE=energy.BASE;DEST=BASE/"artifact_manifest_phase56.json"
def main():
    if DEST.exists():
        entries=json.loads(DEST.read_text())["files"]
        changed=[r["path"] for r in entries if not (ROOT/r["path"]).exists() or energy.sha(ROOT/r["path"])!=r["sha256"]]
        if changed:raise RuntimeError("Frozen phase56 artifacts changed: "+str(changed))
        print("Existing phase56 freeze verified",len(entries));return
    for file in ("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json","artifact_manifest_phase34.json"):
        data=json.loads((BASE/file).read_text());entries=data["files"] if isinstance(data,dict) else data
        changed=[r["path"] for r in entries if not (ROOT/r["path"]).exists() or energy.sha(ROOT/r["path"])!=r["sha256"]]
        if changed:raise RuntimeError("Earlier protected artifacts changed: "+str(changed))
    audit=json.loads((BASE/"phase56_validation/contract_audit.json").read_text())
    assert audit["nested_inner_fits"]==dict(phase5=10080,phase6=336)
    assert audit["direct_nested_source_memberships_checked"]==320
    assert not audit["element_leakage_cases"] and not audit["source_leakage_cases"]
    for p in ("phase5_reference_features/feature_gates.json","phase6_native_reference/feature_gates.json",
              "predictor_bundle_v3/verification.json","predictor_bundle_v4/verification.json"):
        assert json.loads((BASE/p).read_text())["passed"]
    snap=BASE/"code_snapshot_phase56"
    files=list((ROOT/"scripts/trainable_sandbox_20261017").glob("*.py"))
    files += [ROOT/"src/sandbox"/p for p in (
        "implicit_defects.py","energy_learning.py","property_metric_learning.py","nonlinear_property_learning.py",
        "reference_features.py","native_reference_features.py","goal_acquisition.py","kroger_db.py",
        "thermo.py","transport.py","trainable_predictor.py","trainable_predictor_v2.py",
        "trainable_predictor_v3.py","trainable_predictor_v4.py")]
    files += [ROOT/"docs"/p for p in ("ML_SANDBOX_GOAL_20261017.md","ML_SANDBOX_GOAL_AUDIT_20261017.md",
        "ML_SANDBOX_TARGET_LOSS_20261017.md","ML_SANDBOX_TRAINING_THEORY_20261017.md",
        "ML_SANDBOX_REFERENCE_FEATURES_20261017.md","ML_SANDBOX_NATIVE_REFERENCE_20261017.md",
        "ML_SANDBOX_INVENTORY_CLOSURE_NEXT_20261017.md")]
    for p in files:
        target=snap/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert energy.sha(target)==energy.sha(p)
        else:shutil.copyfile(p,target)
    artifacts=[]
    for folder in ("phase5_reference_features","phase6_native_reference","phase56_direct_controls",
                   "phase56_validation","predictor_bundle_v3","predictor_bundle_v4","code_snapshot_phase56"):
        artifacts += [p for p in (BASE/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc"]
    artifacts += [p for p in BASE.glob("*.log") if p.name.startswith(("phase5","phase6","phase56","build_predictor_v3","build_predictor_v4","verify_predictor_v3","verify_predictor_v4"))]
    items=sorted(set(artifacts))
    payload=dict(stage="phase5_phase6_reference_learning_and_prediction",created_utc=energy.now(),
        goal_status="active",next_research="process-to-constrained-inventory learning; not implemented in this freeze",
        scope="two reference input attempts,10416nested fits,common directcontrols,energy/process counterfactual and v3/v4runtime",
        actual_data="29carrier/5sources,52conditionalmu/11sources,5full-chainpairs/2sources;related-study2roundedchemicalpairs",
        new_DFT_or_MD=0,new_material_measurements=0,
        parent_manifests={p:energy.sha(BASE/p) for p in ("artifact_manifest_phase1.json","artifact_manifest_phase2.json","artifact_manifest_phase34.json")},
        notes="Private third-party-derived data/models; original and applicability-corrected direct metadata retained. Mutable root progress/continuations excluded.",
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in items])
    energy.write(DEST,payload)
    print("Frozen new artifacts",len(items));print("Manifest SHA256",energy.sha(DEST))

if __name__=="__main__":main()
