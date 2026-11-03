"""Focused Si/Sn test of a native-anchored, reference-consistent descriptor."""
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os,sys,json,argparse
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.native_reference_features import NativeReferenceBank,NativeReferenceLearner
from sandbox.energy_learning import macro_error
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase3_metric as metric
import phase5_reference as driver

OUT=energy.BASE/"phase6_native_reference"
METHODS=tuple(f"{l}_native_{s}" for l in driver.GRIDS for s in driver.SELECTORS)

def data():
    db,r,oracle=metric.blocks()
    return db,r,oracle,NativeReferenceBank(db)

def frozen():
    p=json.loads((OUT/"protocol.json").read_text())
    for category in ("code_hashes","input_hashes"):
        for name,expected in p[category].items():
            if energy.sha(ROOT/name)!=expected:raise RuntimeError("Frozen native-reference input changed: "+name)
    return p

def initialize():
    db,r,oracle,bank=data()
    p=OUT/"protocol.json"
    payload=dict(version=1,created_utc=energy.now(),
        objective="retain known chemical energy context while respecting atomic/electronic reference changes",
        empirical_motivation="phase5closed-energy property failure and loss of global energy gain despite resolving33/35input collisions; raw energy context improves energies but not robust properties",
        primary="rbf_native_energy vs frozen phase5rbf_base_energy, same six candidates and missing-energy budget",
        auxiliary="Ridge and coverage-first property selectors, not pooled with primary",
        outer=["Si","Sn"],inner="leave each of remaining14 supervised elements out",
        scope="the two carrier-cohort dopants; not a new15-element energy-accuracy claim",
        native_basis="weighted least squares A_native = [dmGa,dmO,q] beta + residual; equal native-parent weights",
        basis_interpretation="coordinate reference, not actual physical chemical potentials",
        dopant_anchor="metadata only: Ga-substitution, interstitial, O-substitution, other; then smallest host atom count and parentID",
        descriptor="A_ref(i)-A_ref(anchor)-[b(i)-b(anchor)] beta",
        reference_proof="rank3nativeD implies beta'=beta+delta; dopant-anchor subtraction cancels dopant reference",
        additional_supplied_energy_labels=0,
        native_inputs=f"{bank.diagnostics['native_states']} native states already supplied in every earlier benchmark",
        budget="Si844known29missing/Sn850known23missing; same parent references, no newDFT",
        kernel_and_selection="exact phase5grid, training weights/scaling and separate energy/coverage-first property selectors",
        methods=METHODS,grid=driver.GRIDS,design=metric.DESIGN,evaluation=metric.EVAL,
        growth_inputs="same unknown-pressure and effective-quench assumptions; no new material protocols",
        grade="retrospective hypothesis after phase5, not blinded or preregistered",
        code_hashes={str(q.relative_to(ROOT)):energy.sha(q) for q in (
            Path(__file__),Path(driver.__file__),ROOT/"src/sandbox/reference_features.py",
            ROOT/"src/sandbox/native_reference_features.py")},
        input_hashes={str(q.relative_to(ROOT)):energy.sha(q) for q in (
            ROOT/"results/phase0/kroger_ga2o3_092724_numeric.npz",
            energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv")},
        parent_phase5_protocol_sha256=energy.sha(driver.OUT/"protocol.json"))
    if p.exists():
        old=json.loads(p.read_text())
        for k in ("code_hashes","input_hashes"):assert old[k]==payload[k]
    else:energy.write(p,payload)

def gates():
    initialize();db,r,oracle,bank=data();rng=np.random.default_rng(617)
    delta=rng.normal(0,2.,len(db.elements));de=.37
    altered=NativeReferenceBank(replace(db,dHo=db.dHo+db.dm@delta+db.charge*de))
    tests={}
    tests["all_reference_descriptor_error_eV"]=float(abs(
        bank.descriptor(r)-altered.descriptor(r)).max())
    tests["native_coefficient_covariance_error_eV"]=float(abs(
        altered.native_coefficients-bank.native_coefficients-np.r_[delta[:2],de]).max())
    a=db.dHo.copy();a[[v["index"] for v in r]]=np.nan
    masked=NativeReferenceBank(replace(db,dHo=a))
    stripped=[{k:v for k,v in x.items() if k not in ("crossing_eV","target_energy_eV","reference_energy_eV")} for x in r]
    tests["hidden_energy_erasure_error"]=float(abs(bank.descriptor(r)-masked.descriptor(stripped)).max())
    tr=[v for v in r if v["element"]!="Si"];te=[v for v in r if v["element"]=="Si"]
    m=NativeReferenceLearner(alpha=.01,length=.5).fit(tr,bank)
    tests["model_reference_invariance_error_eV"]=float(abs(m.predict(te,bank)-m.predict(te,altered)).max())
    tests["model_hidden_energy_erasure_error_eV"]=float(abs(m.predict(te,bank)-m.predict(te,masked)).max())
    assert max(tests.values())<5e-8
    energy.write(OUT/"feature_gates.json",dict(passed=True,tests=tests,
        native_diagnostics=bank.diagnostics,metadata_anchors=bank.anchors,
        guarantee="reference consistency and no hidden-label use; no property improvement guarantee"))
    print(json.dumps(tests,indent=2),flush=True)

def configure():
    driver.OUT=OUT;driver.data=data;driver.frozen=frozen
    driver.ReferenceEnergyLearner=NativeReferenceLearner

def candidate(task):
    configure();return driver.candidate(task)

def outer(task):
    configure();return driver.outer(task)

def run(workers):
    frozen();assert json.loads((OUT/"feature_gates.json").read_text())["passed"]
    tasks=[(e,l,"native",i) for e,l in product(("Si","Sn"),driver.GRIDS)
           for i in range(len(driver.GRIDS[l]))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(candidate,tasks):pass
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(outer,product(("Si","Sn"),driver.GRIDS,("native",))):pass
    import pandas as pd
    ep=pd.concat([read_result_csv(p) for p in sorted((OUT/"energy_outer").glob("*.csv"))],ignore_index=True)
    ep.to_csv(OUT/"energy_predictions.csv",index=False)
    vp=pd.concat([read_result_csv(p) for p in sorted((OUT/"property_outer").glob("*.csv"))],ignore_index=True)
    vp.to_csv(OUT/"virtual_predictions.csv",index=False)
    rows=[]
    for (m,e),g in ep.groupby(["method","element"]):
        v=vp[vp.method.eq(m)&vp.element.eq(e)&vp.reference_eligible]
        rows.append(dict(method=m,element=e,energy_MAE_eV=macro_error(g.to_dict("records"),g.predicted_crossing_eV.to_numpy(float)),
            eligible=len(v),numeric=int(v.predicted_log_nH.notna().sum()),physical=int(v.predicted_physical_valid.sum()),
            mathematical_property_MAE_dex=float(abs(v.predicted_log_nH-v.reference_log_nH).mean())))
    energy.csvwrite(OUT/"summary.csv",rows)
    print("Native-anchored whole-element fitting complete",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--initialize",action="store_true")
    ap.add_argument("--gates",action="store_true");ap.add_argument("--train",action="store_true")
    ap.add_argument("--workers",type=int,default=4);a=ap.parse_args()
    if a.initialize:initialize()
    if a.gates:gates()
    if a.train:run(a.workers)
