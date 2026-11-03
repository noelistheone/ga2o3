"""Upgrade known-input checks without refitting/replacing frozen v1 artifacts."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.kroger_db import load
from sandbox.trainable_predictor_v2 import structure_fingerprint,supplied_energy_fingerprint
import phase2_energy as energy

SOURCE=energy.BASE/"predictor_bundle_v1"
OUT=energy.BASE/"predictor_bundle_v2"

def main():
    original=json.loads((SOURCE/"manifest.json").read_text());db=load()
    protocol=dict(version=2,created_utc=energy.now(),
        parent_manifest_sha256=energy.sha(SOURCE/"manifest.json"),
        code_sha256=energy.sha(Path(__file__)),
        runtime_sha256=energy.sha(ROOT/"src/sandbox/trainable_predictor_v2.py"),
        base_runtime_sha256=energy.sha(ROOT/"src/sandbox/trainable_predictor.py"),
        change="structure plus per-request known-energy checks; hidden target entries may be absent/NaN",
        fits="exact version1 saved learners/process/transport/anchor rules, no retraining",
        reference_mode="full_library still requires all energetic entries",
        predictions="must reproduce version1 when its complete database is supplied")
    path=OUT/"input_contract.json"
    if path.exists():
        old=json.loads(path.read_text())
        for k in ("parent_manifest_sha256","code_sha256","runtime_sha256","base_runtime_sha256"):
            assert old[k]==protocol[k]
    else:energy.write(path,protocol)
    for entry in original["files"]:
        source=SOURCE/entry["path"];dest=OUT/entry["path"]
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert energy.sha(dest)==entry["sha256"]
        else:shutil.copyfile(source,dest)
    # Process parameters are all in the manifest; retained v1 training files
    # resolve through the explicitly recorded parent, never new query labels.
    manifest=dict(original)
    manifest["version"]=2
    manifest["parent_manifest_sha256"]=energy.sha(SOURCE/"manifest.json")
    manifest["structure_fingerprint"]=structure_fingerprint(db)
    manifest["supplied_energy_fingerprints"]={e:{} for e in original["supported_dopants"]}
    for e in original["supported_dopants"]:
        for m in original["methods"]:
            hidden=[] if m=="full_library" else [int(r["index"]) for r in original["target_records"][e]]
            manifest["supplied_energy_fingerprints"][e][m]=supplied_energy_fingerprint(db,hidden)
    manifest["inference_input_contract"]="known energetic entries checked; hidden nonreference target energies not read as model inputs"
    energy.write(OUT/"manifest.json",manifest)
    print(OUT)

if __name__=="__main__":main()
