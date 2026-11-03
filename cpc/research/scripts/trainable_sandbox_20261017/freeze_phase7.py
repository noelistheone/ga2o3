"""Freeze the constrained-inventory prediction attempt and its evidence."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy

BASE=energy.BASE;OUT=BASE/"phase7_inventory";DEST=BASE/"artifact_manifest_phase7.json"

def verify(path):
    x=json.loads(path.read_text());items=x["files"] if isinstance(x,dict) else x
    changed=[v["path"] for v in items if not (ROOT/v["path"]).exists()
             or energy.sha(ROOT/v["path"])!=v["sha256"]]
    if changed:raise RuntimeError("Protected files changed: "+str(changed))
    return len(items)

def main():
    if DEST.exists():print("Verified frozen phase7",verify(DEST));return
    parents=("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json",
             "artifact_manifest_phase34.json","artifact_manifest_phase56.json")
    for p in parents:verify(BASE/p)
    a=json.loads((OUT/"contract_audit.json").read_text())
    assert a["passed"] and a["fits"]==90 and a["trials"]==180 and a["all_primary_numeric"]
    assert json.loads((OUT/"numerical_gates.json").read_text())["passed"]
    assert json.loads((OUT/"rendering_check.json").read_text())["pages"]==5
    snap=BASE/"code_snapshot_phase7"
    sources=[ROOT/"scripts/trainable_sandbox_20261017"/p for p in (
        "phase7_inventory_gates.py","phase7_inventory_property.py","phase7_inventory_analyze.py",
        "phase7_inventory_report.py","freeze_phase7.py","phase2_property.py","phase2_energy.py",
        "csv_io.py","csv_io_v2.py","export_research_pdf.py")]
    sources += [ROOT/"src/sandbox"/p for p in ("inventory_closure.py","implicit_defects.py",
        "property_metric_learning.py","energy_learning.py","kroger_db.py","thermo.py","transport.py")]
    sources += [ROOT/"docs"/p for p in ("ML_SANDBOX_INVENTORY_PHASE7_20261017.md",
        "ML_SANDBOX_GOAL_CHECK_PHASE7_20261017.md","ML_SANDBOX_GOAL_20261017.md")]
    for p in sources:
        target=snap/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert energy.sha(target)==energy.sha(p)
        else:shutil.copyfile(p,target)
    paths=[p for folder in (OUT,snap) for p in folder.rglob("*")
           if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc"]
    energy.write(DEST,dict(stage="phase7_constrained_inventory_prediction",created_utc=energy.now(),
        goal_status="active",scope="six models,90source-excluded fits,180starts,54numerical gates,5page report",
        conclusion="current learned stock family has no stable incremental carrier accuracy; preserve all results",
        actual_data="same29carrier samples/5DOI groups, no new material measurements or energetic labels",
        next="per-specimen process and variable-temperature Hall/SIMS input audit for predictive internal learning",
        parent_manifests={p:energy.sha(BASE/p) for p in parents},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p))
               for p in sorted(paths)]))
    print("Frozen phase7 files",len(paths));print("Manifest SHA256",energy.sha(DEST))

if __name__=="__main__":main()
