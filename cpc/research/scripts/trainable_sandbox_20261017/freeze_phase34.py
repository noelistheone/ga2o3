"""Freeze new prediction-stage evidence without altering earlier manifests."""
from pathlib import Path
import sys,json,shutil,platform,importlib.metadata
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/"src"))
import phase2_energy as energy

BASE=energy.BASE
DEST=BASE/"artifact_manifest_phase34.json"

def main():
    if DEST.exists():
        x=json.loads(DEST.read_text())
        changed=[r["path"] for r in x["files"] if not (ROOT/r["path"]).exists()
                 or energy.sha(ROOT/r["path"])!=r["sha256"]]
        if changed:raise RuntimeError("Frozen stage34 artifacts changed: "+str(changed))
        print("Existing stage34 freeze verified:",len(x["files"]));return
    for filename in ("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json"):
        x=json.loads((BASE/filename).read_text());files=x["files"] if isinstance(x,dict) else x
        changed=[r["path"] for r in files if not (ROOT/r["path"]).exists()
                 or energy.sha(ROOT/r["path"])!=r["sha256"]]
        if changed:raise RuntimeError("Earlier protected files changed: "+str(changed))
    assert json.loads((BASE/"phase3_target_loss/numerical_gates.json").read_text())["passed"]
    assert json.loads((BASE/"phase4_nonlinear/gradient_gates.json").read_text())["passed"]
    assert json.loads((BASE/"predictor_bundle_v1/verification.json").read_text())["passed"]
    assert json.loads((BASE/"predictor_bundle_v2/verification.json").read_text())["passed"]
    audit=json.loads((BASE/"phase34_validation/contract_audit.json").read_text())
    assert not audit["source_leakage_cases"] and not audit["element_leakage_cases"]
    assert not audit["accepted_loss_increases"]
    scripts=ROOT/"scripts/trainable_sandbox_20261017"
    sources=[p for p in scripts.glob("*.py") if p.name.startswith((
        "phase3","phase4","phase34","build_predictor","predict_properties","verify_predictor",
        "freeze_phase34","export_research_pdf","csv_io","transport_chain"))]
    sources += [ROOT/"src/sandbox"/f for f in (
        "implicit_defects.py","energy_learning.py","property_metric_learning.py",
        "nonlinear_property_learning.py","trainable_predictor.py","trainable_predictor_v2.py",
        "kroger_db.py","transport.py","thermo.py")]
    sources += [ROOT/"docs"/f for f in ("ML_SANDBOX_GOAL_20261017.md",
        "ML_SANDBOX_GOAL_AUDIT_20261017.md","ML_SANDBOX_TARGET_LOSS_20261017.md",
        "ML_SANDBOX_TRAINING_THEORY_20261017.md")]
    snap=BASE/"code_snapshot_phase34"
    for p in sources:
        target=snap/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert energy.sha(target)==energy.sha(p)
        else:shutil.copyfile(p,target)
    versions={}
    for package in ("numpy","scipy","pandas","scikit-learn","matplotlib","torch"):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    energy.write(BASE/"phase34_validation/environment.json",dict(
        python=sys.version,executable=sys.executable,platform=platform.platform(),
        packages=versions,cpu_threads=1,authorized_gpu=1,this_stage_gpu_training=False))
    folders=("phase3_target_loss","phase4_nonlinear","new_property_sources",
             "predictor_bundle_v1","predictor_bundle_v2","phase34_validation","code_snapshot_phase34")
    files=[]
    for folder in folders:
        for p in sorted((BASE/folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc":
                files.append(p)
    logs=[p for p in BASE.glob("*.log") if p.name.startswith(("phase3","phase4","phase34","build_predictor"))]
    files+=logs
    paths=sorted(set(files))
    payload=dict(stage="phase3_phase4_prediction_training_and_runtime",
        created_utc=energy.now(),goal_status="active",
        direction="ML+Sandbox prediction of doped material properties",
        status="implemented/tested current training variants, no stable nonlinear material advantage",
        new_DFT_MD_or_material_measurements=0,
        parent_manifests={f:energy.sha(BASE/f) for f in (
            "artifact_manifest_phase1.json","artifact_manifest_phase2.json")},
        notes="Private derived catalogue/models; mutable root progress/README/continuation excluded. Older failed logs and v1 input-check limitation retained.",
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in paths])
    energy.write(DEST,payload)
    print("Frozen artifacts:",len(paths));print("Manifest SHA256:",energy.sha(DEST))

if __name__=="__main__":main()
