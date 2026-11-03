"""Real source-held-out dopant-property validation of nonlinear energy learning.

Reuses the frozen phase2 process fit and phase3 exact absent-impurity catalogue.
Copied phase3 controls have identical cases and are not retrained or overwritten.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os, sys, json, shutil, argparse
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.special import expit
from sandbox.energy_learning import reconstruct
from sandbox.property_metric_learning import element_database
from sandbox.implicit_defects import ImplicitDefectLayer
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as fitter
import phase3_metric as metric
import phase4_nested as training

OUT = training.OUT / "real_property"
METHODS = ("kernel_nonlinear_l1", "kernel_nonlinear_or_energy")
CONTROLS = ("oracle_full", "global_mean", "ridge", "kernel_energy",
            "kernel_energy_select_energy", "kernel_diagonal_l1", "kernel_full_l1")
DBS = {}
ORIGINAL_SUMMARY = fitter.summarize

def freeze():
    training.freeze()
    model_paths = [training.OUT / "energy_outer" / f"{e}.csv" for e in ("Si", "Sn")]
    control_paths = sorted(p for p in (metric.OUT / "real_property/outer").glob("*.csv")
                           if p.name.split("__", 1)[1][:-4] in CONTROLS)
    if len(control_paths) != 5 * len(CONTROLS):
        raise RuntimeError("Missing frozen phase3 controls")
    payload = dict(version=1, created_utc=energy.now(), methods=METHODS,
        controls=CONTROLS,
        code_hashes={str(p.relative_to(ROOT)): energy.sha(p) for p in
            (Path(__file__), Path(fitter.__file__),
             ROOT / "src/sandbox/property_metric_learning.py",
             ROOT / "scripts/trainable_sandbox_20261017/csv_io_v2.py")},
        energy_input_hashes={str(p.relative_to(ROOT)): energy.sha(p) for p in model_paths},
        control_hashes={str(p.relative_to(ROOT)): energy.sha(p) for p in control_paths},
        training_protocol_sha256=energy.sha(training.OUT / "nested_protocol.json"),
        process_fit="frozen phase2 equations: family-equal log-squared loss, two starts, max_nfev60",
        selection="inner source exclusion for 0/1/2-anchor shrinkage; target query labels excluded",
        energy_training="Si/Sn outer element excluded; remaining14 elements nested leave-one-element-out",
        comparison="nonlinear and ordinary six-candidate selection; extra actual root/time cost reported",
        secondary="nonlinear-or-energy has12 candidates, not the primary six-candidate contrast",
        exact_absence=True,
        coverage="same29 samples/5 DOI families and common18 queries; physical refusals retained",
        missing_input="all pO2 unreported; common1e-5atm scenario; same historical growth proxies",
        grade="retrospective real-property comparison; virtual training targets are not material labels")
    path = OUT / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        for key in ("code_hashes", "energy_input_hashes", "control_hashes"):
            assert old[key] == payload[key]
    else:
        energy.write(path, payload)
    return control_paths

def prepare_controls():
    for p in freeze():
        dest = OUT / "outer" / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert energy.sha(dest) == energy.sha(p)
        else:
            shutil.copyfile(p, dest)

def database(element, method):
    key = (element, method)
    if key not in DBS:
        db, records, _ = energy.records()
        target = [v for v in records if v["element"] == element]
        data = read_result_csv(training.OUT / "energy_outer" / f"{element}.csv")
        data = data[data.method.eq(method)]
        lookup = dict(zip(data.label_id, data.predicted_crossing_eV))
        if set(lookup) != {v["label_id"] for v in target}:
            raise RuntimeError("Missing nonlinear energy predictions")
        DBS[key] = reconstruct(db, target, [lookup[v["label_id"]] for v in target])
    return DBS[key]

def point(row, w, target_element, method):
    key = (target_element, method, row.element)
    if key not in fitter.LAYERS:
        db, _ = element_database(database(target_element, method), row.element)
        fitter.LAYERS[key] = ImplicitDefectLayer(db, (row.element,))
    layer = fitter.LAYERS[key]
    s = expit(w[0])
    logT = fitter.L0 + (fitter.L1 - fitter.L0) * s
    pred = layer.predict(layer.parameters(T=10**logT, Nd=10**w[1],
                         doses=(row.incorporated_cm3,), pO2=1e-5))
    chain = np.zeros((layer.np, 2))
    chain[0, 0] = (fitter.L1 - fitter.L0) * s * (1-s)
    chain[1, 1] = 1.
    return pred, pred.jacobian @ chain

def configure():
    fitter.OUT = OUT
    fitter.METHODS = METHODS
    fitter.freeze = freeze
    fitter.database = database
    fitter.point = point
    fitter.read_result_csv = read_result_csv
    fitter.DATA = read_result_csv(energy.BASE / "dataset_v1.csv")
    fitter.ROLES = read_result_csv(energy.BASE / "episodes_v1.csv")
    fitter.FITS.clear(); fitter.DBS.clear(); fitter.LAYERS.clear(); DBS.clear()

def worker(method):
    configure()
    fitter.summarize = lambda: None
    fitter.run(method)
    return method

def run(workers):
    prepare_controls()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for method in pool.map(worker, METHODS):
            print("real nonlinear complete", method, flush=True)
    configure()
    ORIGINAL_SUMMARY()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    run(ap.parse_args().workers)
