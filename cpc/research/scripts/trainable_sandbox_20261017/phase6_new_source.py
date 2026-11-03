"""Same explicit-refusal related-study protocol for native-reference models."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
import phase5_new_source as driver
import phase6_property as study

OUT=study.training.OUT/"new_property_source"
if __name__=="__main__":
    study.freeze()
    path=OUT/"wrapper_protocol.json"
    payload=dict(wrapper_sha256=energy.sha(Path(__file__)),
        driver_sha256=energy.sha(Path(driver.__file__)),
        property_protocol_sha256=energy.sha(study.OUT/"protocol.json"),
        grade="same exposed related study and rounding scenarios; no target tuning")
    if path.exists():assert json.loads(path.read_text())==payload
    else:energy.write(path,payload)
    driver.OUT=OUT;driver.study=study
    driver.main()
