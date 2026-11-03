"""Exact phase5 real-source protocol for the native-anchored descriptor."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import pandas as pd
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as fitter
import phase5_property as driver
import phase6_native_reference as training

OUT=training.OUT/"real_property"
METHODS=training.METHODS
CONTROLS=("oracle_full","global_mean","ridge","kernel_energy_select_energy","kernel_nonlinear_l1",
          "rbf_base_energy","rbf_base_property","rbf_closed_energy","rbf_closed_property")
PREVIOUS=energy.BASE/"phase5_reference_features/real_property"
ENERGY={};CONFIGURED=False

def energy_frame(element):
    if element not in ENERGY:
        files=sorted((training.OUT/"energy_outer").glob(f"{element}__*.csv"))
        if len(files)!=2:raise RuntimeError("Native target-element training incomplete")
        ENERGY[element]=pd.concat([read_result_csv(p) for p in files],ignore_index=True)
    return ENERGY[element]

def configure_driver():
    global CONFIGURED
    if CONFIGURED:return
    driver.training=training;driver.OUT=OUT;driver.METHODS=METHODS;driver.CONTROLS=CONTROLS
    driver.previous=SimpleNamespace(OUT=PREVIOUS)
    driver.energy_frame=energy_frame;driver.freeze=freeze
    driver.DBS.clear();driver.ENERGY.clear();CONFIGURED=True

def freeze():
    training.frozen();configure_driver()
    mapping=driver.aliases()
    paths=[p for e in ("Si","Sn") for p in sorted((training.OUT/"energy_outer").glob(f"{e}__*.csv"))]
    controls=sorted(p for p in (PREVIOUS/"outer").glob("*.csv")
                    if p.name.split("__",1)[1][:-4] in CONTROLS)
    if len(controls)!=5*len(CONTROLS):raise RuntimeError("Missing previous controls")
    payload=dict(version=1,created_utc=energy.now(),methods=METHODS,controls=CONTROLS,
        exact_energy_aliases=mapping,primary=["rbf_base_energy","rbf_native_energy"],
        comparison="same frozen phase5 process fitting and source/anchor validation; no new hyperparameter selection",
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (
            Path(__file__),Path(driver.__file__),Path(fitter.__file__))},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},
        control_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in controls},
        energy_protocol_sha256=energy.sha(training.OUT/"protocol.json"),
        source_exclusion="same29records/5sources, inner shrinkage and0/1/2anchors; fixedcommon18",
        energy_budget="same Si844/Sn850 known including243native; no newly revealed target energies",
        pressure="same unreported1e-5atm scenario",grade="retrospective development")
    p=OUT/"protocol.json"
    if p.exists():
        old=json.loads(p.read_text())
        for k in ("code_hashes","input_hashes","control_hashes","exact_energy_aliases"):
            assert old[k]==payload[k]
    else:energy.write(p,payload)
    return mapping,controls

def database(element,method):
    configure_driver();return driver.database(element,method)

def worker(method):
    configure_driver();return driver.worker(method)

def run(workers):
    configure_driver();mapping=driver.prepare()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for method in pool.map(worker,list(dict.fromkeys(mapping.values()))):
            print("native property complete",method,flush=True)
    driver.configure();driver.copy_aliases(mapping);driver.ORIGINAL_SUMMARY()

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=4)
    run(ap.parse_args().workers)
