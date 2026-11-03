"""Reuse the audited full-chain transport calculation on phase5 predictions."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
import phase4_transport as driver
import phase5_property as study

OUT=study.training.OUT/"transport"

def main():
    p=OUT/"wrapper_protocol.json"
    payload=dict(created_utc=energy.now(),wrapper_code_sha256=energy.sha(Path(__file__)),
        driver_code_sha256=energy.sha(Path(driver.__file__)),
        source_predictions_sha256=energy.sha(study.OUT/"predictions.csv"),
        change="same specimen joins and source-excluded CT/median calculation on new nH predictions",
        no_new_measured_nH_in_transport=True,conductivity_is_derived=True)
    if p.exists():
        old=json.loads(p.read_text())
        for k in ("wrapper_code_sha256","driver_code_sha256","source_predictions_sha256"):
            assert old[k]==payload[k]
    else:energy.write(p,payload)
    driver.OUT=OUT;driver.study=SimpleNamespace(OUT=study.OUT)
    driver.main()

if __name__=="__main__":main()
