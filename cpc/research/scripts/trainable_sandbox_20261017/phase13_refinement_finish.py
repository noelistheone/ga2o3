"""Finish metadata/selection after the retained duplicate-key driver failure."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase13_refinement as ref
import phase2_energy as energy


def main():
    gate=json.loads((ref.OUT/"refinement_gate.json").read_text());assert gate["passed"]
    fits=list((ref.OUT/"fits").glob("*.json"));assert len(fits)==1
    records=[]
    for path in fits:
        f=json.loads(path.read_text());source=ROOT/f["refinement"]["original_fit_path"]
        assert energy.sha(source)==f["refinement"]["original_fit_sha256"]
        assert min(t["loss"] for t in f["trials"] if t["loss"] is not None)==f["refinement"]["gate"]["new_objective"]
        records.append(dict(key=f["key"],seconds=f["seconds"],counters=f["counters"],gate=f["refinement"]["gate"]))
    protocol=dict(created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in [ref.OUT/"protocol.json",ref.OUT/"refinement_gate.json",*fits]},
        repair="initial driver saved numerical fit then failed on duplicatekey in metadata;thiswrapper does no numerical refit",
        initial_driver_failed=True,numerical_fits_modified=False)
    energy.write(ref.OUT/"metadata_finish_protocol.json",protocol)
    energy.write(ref.OUT/"refinement_execution_complete.json",dict(created_utc=energy.now(),affected_records=1,records=records,
        numerical_fits_complete=True,initial_driver_failed=True,metadata_repaired=True,goal_complete=False))
    ref.select()
    print("Retained numerical fits;metadata and complete nestedselection finished")


if __name__=="__main__":main()
