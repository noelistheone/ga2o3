"""Preserve temperature data/controls and material-input version experiments."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/"phase8_process_temperature";DEST=BASE/"artifact_manifest_phase8.json"

def verify(path):
    x=json.loads(path.read_text());entries=x["files"] if isinstance(x,dict) else x
    changed=[r["path"] for r in entries if not (ROOT/r["path"]).exists() or energy.sha(ROOT/r["path"])!=r["sha256"]]
    if changed:raise RuntimeError("Protected files changed: "+str(changed))
    return len(entries)

def main():
    if DEST.exists():print("Verified phase8",verify(DEST));return
    parents=("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json",
        "artifact_manifest_phase34.json","artifact_manifest_phase56.json","artifact_manifest_phase7.json")
    for file in parents:verify(BASE/file)
    assert json.loads((OUT/"extraction_v2/extraction_checks.json").read_text())["readings"]==196
    assert json.loads((OUT/"contract_audit.json").read_text())["passed"]
    assert json.loads((OUT/"band_profiles_v2/contract_audit.json").read_text())["material_fits"]==45
    assert json.loads((OUT/"postpublication_protocol_audit.json").read_text())["passed"]
    assert json.loads((OUT/"rendering_check.json").read_text())["pages"]==5
    snap=BASE/"code_snapshot_phase8"
    scripts=list((ROOT/"scripts/trainable_sandbox_20261017").glob("phase8*.py"))
    scripts += [ROOT/"scripts/trainable_sandbox_20261017"/p for p in (
        "freeze_phase8.py","phase2_energy.py","phase2_property.py","csv_io.py","csv_io_v2.py","export_research_pdf.py")]
    scripts += [ROOT/"src/sandbox"/p for p in ("donor_ionization.py","implicit_defects.py",
        "thermo.py","kroger_db.py","transport.py","energy_learning.py")]
    scripts += [ROOT/"docs"/p for p in ("ML_SANDBOX_TEMPERATURE_THEORY_20261017.md","ML_SANDBOX_GOAL_20261017.md")]
    for p in scripts:
        target=snap/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert energy.sha(target)==energy.sha(p)
        else:shutil.copyfile(p,target)
    paths=sorted(p for folder in (OUT,snap) for p in folder.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc")
    payload=dict(stage="phase8_temperature_prediction_inputs_and_material_versions",
        created_utc=energy.now(),goal_status="active",
        scope="142n+54mu curve reads/7previous specimens,2DOSversions reduced controls,3bandscenarios45primaryrefits",
        inference_bundle="v4unchanged,legacydeclaredscenario;nonewfulltemperatureproductionmodel",
        boundary="reducedcontrollearningnotfullKROGER;temperatureanchorsper specimen differfromoriginalperDOI",
        results="nostableincrementfromtestedE(C);twoanchorline retained;publishedinput mismatch quantified withfixed/refittedprocess",
        next="immutablematerialprofile and fullvariabletemperature energetic/frozenstock layer,thenrealsharedlearning",
        actual_new_physical_samples=0,new_DFT_MD=0,
        parent_manifests={p:energy.sha(BASE/p) for p in parents},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in paths])
    energy.write(DEST,payload);print("Frozen phase8",len(paths));print("Manifest SHA256",energy.sha(DEST))

if __name__=="__main__":main()
