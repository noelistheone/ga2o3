"""Separate repository-default and publication-declared band input scenarios.

Old source/energetic/anchor protocols are replayed unchanged. Each profile runs
in its own Python process; in-memory thermo constants never interleave profiles.
No frozen source files or energetic values are edited.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os,sys,json,argparse
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv
from sandbox import thermo
from sandbox.implicit_defects import ImplicitDefectLayer
from sandbox.kroger_db import load
import phase2_energy as energy
import phase2_property as fitter

OUT=energy.BASE/"phase8_process_temperature/band_profiles_v2"
PROFILES={
    "repository_default":dict(NC_REF=1e19,NV_REF=5e20,EG0=5.0),
    "published_DOS_only":dict(NC_REF=3.33e18,NV_REF=1.71e20,EG0=5.0),
    "published_DOS_gap":dict(NC_REF=3.33e18,NV_REF=1.71e20,EG0=5.1),
}
DATA=read_result_csv(energy.BASE/"dataset_v1.csv")
ROLES=read_result_csv(energy.BASE/"episodes_v1.csv")

def freeze():
    p=OUT/"protocol.json"
    payload=dict(created_utc=energy.now(),version=2,profiles=PROFILES,
        code_hashes={str(q.relative_to(ROOT)):energy.sha(q) for q in (
            Path(__file__),Path(fitter.__file__),ROOT/"src/sandbox/implicit_defects.py",
            ROOT/"src/sandbox/thermo.py",ROOT/"src/sandbox/kroger_db.py",
            ROOT/"scripts/trainable_sandbox_20261017/csv_io_v2.py")},
        input_hashes={str(q.relative_to(ROOT)):energy.sha(q) for q in (
            energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv",
            ROOT/"results/phase0/kroger_ga2o3_092724_numeric.npz",
            ROOT/"external/KROGER/Ga2O3/KROGER_Set_Ga2O3_Material_Conditions.m")},
        publication="10.1039/D4CP04817B final RSC fulltext,Temperature dependence of bandgap",
        publication_values="NC3.33e18,NV1.71e20 at300K,Eg0=5.1;also cached arxiv selectedmethods",
        reference_values="localrepository material defaults NC1e19,NV5e20,Eg0=5.0",
        fixed_inputs="same defect formation energies,charge catalogue,thermal functions,EcT_fraction.4,pO2 unmeasured1e-5atm",
        not_claimed="not a complete reproduction of published bulk experiment; energetic alignment and supplied thin-film process assumptions remain",
        primary="same29actualdose/5DOI,full energetic input,0/1/2anchors per DOI,guard/common18 unchanged",
        fit="same2source-excluded parameters,effectiveT600..1950,logNd14..18,two starts,max_nfev60;innerLOFOlambda.1/1/10/inf",
        frozen_parameter_diagnostic="same old source-excluded fitted process parameters+old shrinkage,change only profile;no refit",
        gradient_gate="four oldinput partials at regularSi state per profile",
        interpretation="input-version sensitivity,not selection of most favorable physical constants",
        process_isolation="oneprofile perworker,patch loadedthermo constants in memory;do not edit any frozen file",
        old_manifest=energy.sha(energy.BASE/"artifact_manifest_phase7.json"))
    if p.exists():
        d=json.loads(p.read_text())
        for k in ("profiles","code_hashes","input_hashes"):assert d[k]==energy.clean(payload[k]),k
    else:energy.write(p,payload)
    for name,values in PROFILES.items():
        local=OUT/name/"refit/protocol.json"
        context=dict(parent_protocol_sha256=energy.sha(p),profile_name=name,band_values=values,
                     method="oracle_full",fitter="unchanged phase2 equations/source and anchor rules")
        if local.exists():assert json.loads(local.read_text())==context
        else:energy.write(local,context)

def configure(name):
    for k,v in PROFILES[name].items():setattr(thermo,k,v)
    fitter.OUT=OUT/name/"refit";fitter.METHODS=("oracle_full",);fitter.freeze=freeze
    fitter.DATA=DATA;fitter.ROLES=ROLES;fitter.read_result_csv=read_result_csv
    fitter.FITS.clear();fitter.DBS.clear();fitter.LAYERS.clear()

def gradient_gate(name):
    layer=ImplicitDefectLayer(load(),("Si",))
    p=layer.parameters(T=1400.,Nd=1e16,doses=(1e18,));a=layer.predict(p);rows=[]
    for j in range(len(p)):
        h=1e-5;x=p.copy();y=p.copy();x[j]+=h;y[j]-=h
        fd=(layer.predict(x).values-layer.predict(y).values)/(2*h)
        err=float(np.max(abs(fd-a.jacobian[:,j])/(1+abs(fd))))
        rows.append(dict(parameter=a.parameter_names[j],scaled_error=err,passed=err<3e-5))
    energy.write(OUT/name/"gradient_gate.json",dict(passed=all(v["passed"] for v in rows),checks=rows))
    assert all(v["passed"] for v in rows)

def frozen_process(name):
    oldpred=read_result_csv(energy.BASE/"phase56_validation/all_primary_predictions.csv")
    oldpred=oldpred[oldpred.origin.eq("phase2")&oldpred.method.eq("oracle_full")]
    result=[]
    for held in sorted(DATA.family.unique()):
        subset=oldpred[oldpred.family.eq(held)];keys=subset.fit_key.unique();assert len(keys)==1
        path=energy.OUT/"property/fits"/(keys[0]+".json")
        f0=json.loads(path.read_text());rows=[]
        for row in DATA[DATA.family.eq(held)].itertuples():
            p,_=fitter.point(row,f0["parameters"],"Si","oracle_full");dg=p.diagnostics
            rows.append(dict(sample_id=row.sample_id,predicted_log_nH=float(p.values[0]),numeric=True,
                physical_valid=bool(dg["gradient_valid"] and dg["single_electron_Hall_valid"] and dg["above_carrier_detection_floor"]),
                diagnostics=dg))
        fixed=dict(key=f0["key"],predictions=rows)
        for b in (0,1,2):
            lam=subset[subset.budget.eq(b)].shrinkage_lambda.unique();assert len(lam)==1
            lam=np.inf if str(lam[0])=="infinity" else float(lam[0])
            score,rs=fitter.evaluate(fixed,held,b,lam)
            for r in rs:r.update(profile=name,method="frozen_old_process",old_fit_key=f0["key"])
            result+=rs
    energy.csvwrite(OUT/name/"frozen_old_process.csv",result)

def worker(name):
    freeze();configure(name);gradient_gate(name);frozen_process(name);fitter.run("oracle_full")
    return name

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--profile",choices=PROFILES)
    args=ap.parse_args();freeze()
    if args.profile:worker(args.profile)
    else:
        with ProcessPoolExecutor(max_workers=3) as pool:
            for p in pool.map(worker,PROFILES):print("Bandprofile complete",p,flush=True)

if __name__=="__main__":main()
