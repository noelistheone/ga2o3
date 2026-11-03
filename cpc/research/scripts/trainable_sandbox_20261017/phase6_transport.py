"""Audited complete transport chain for native-reference predictions."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
import phase4_transport as driver
import phase6_property as study

OUT=study.training.OUT/"transport"
if __name__=="__main__":
    path=OUT/"wrapper_protocol.json"
    payload=dict(wrapper_sha256=energy.sha(Path(__file__)),driver_sha256=energy.sha(Path(driver.__file__)),
        predictions_sha256=energy.sha(study.OUT/"predictions.csv"),
        rule="same5paired specimens and source-excluded CT/median; no measured query nH in complete chain")
    if path.exists():assert json.loads(path.read_text())==payload
    else:energy.write(path,payload)
    driver.OUT=OUT;driver.study=SimpleNamespace(OUT=study.OUT)
    driver.main()
