"""Compare fixed-process and recalibrated band input scenarios; no new fits."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.special import expit
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase8_band_profiles_v2 as study

OUT=study.OUT

def family_mean(d,column):return float(d.groupby("family")[column].mean().mean())

def main():
    study.freeze();allrows=[];fits=[];contexts=0
    for name,values in study.PROFILES.items():
        p=OUT/name
        gate=json.loads((p/"gradient_gate.json").read_text());assert gate["passed"]
        a=read_result_csv(p/"refit/predictions.csv");a["profile"]=name;a["process_mode"]="refit"
        b=read_result_csv(p/"frozen_old_process.csv");b["process_mode"]="frozen"
        allrows.extend([a,b])
        fs={q.stem:json.loads(q.read_text()) for q in (p/"refit/fits").glob("*.json")}
        assert len(fs)==15
        for f in fs.values():
            for t in f["trials"]:
                assert t["loss"] is not None
            row=dict(profile=name,fit_key=f["key"],train_families=";".join(f["train_families"]),
                train_source_count=len(f["train_families"]),seconds=f["seconds"],parameters=f["parameters"],
                starts=len(f["trials"]),nonconverged=sum(not v["success"] for v in f["trials"]),
                T_effective_K=10**(study.fitter.L0+(study.fitter.L1-study.fitter.L0)*expit(f["parameters"][0])),
                background_cm3=10**f["parameters"][1])
            fits.append(row)
        for held in sorted(study.DATA.family.unique()):
            o=a[a.family.eq(held)]
            key=o.fit_key.unique();assert len(key)==1 and held not in fs[key[0]]["train_families"]
            training=sorted(set(study.DATA.family)-{held})
            for inner in training:
                expected=set(training)-{inner}
                f=[v for v in fs.values() if set(v["train_families"])==expected]
                assert len(f)==1;contexts+=1
    d=pd.concat(allrows,ignore_index=True);d.to_csv(OUT/"all_predictions.csv",index=False)
    common=set(study.ROLES[study.ROLES.budget.eq(2)&study.ROLES.role.eq("query")].sample_id)
    by=[]
    for (profile,mode,b,f),g in d[d.role.eq("query")].groupby(["profile","process_mode","budget","family"]):
        for scope in ("all_query","common18"):
            h=g if scope=="all_query" else g[g.sample_id.isin(common)]
            e=abs(h.predicted_log_nH-h.observed_log_nH)
            by.append(dict(profile=profile,process_mode=mode,budget=int(b),family=f,scope=scope,n=len(h),
                numeric=int(e.notna().sum()),physical=int(h.physical_valid.sum()),
                MAE=float(e.mean()) if e.notna().all() else None))
    by=pd.DataFrame(by);by.to_csv(OUT/"by_family.csv",index=False)
    s=by.groupby(["profile","process_mode","budget","scope"]).agg(
        source_groups=("family","nunique"),reads=("n","sum"),numeric=("numeric","sum"),
        physical=("physical","sum"),MAE=("MAE","mean"),worst=("MAE","max")).reset_index()
    s.to_csv(OUT/"summary.csv",index=False);energy.write(OUT/"fit_audit.json",fits)
    base=d[d.profile.eq("repository_default")&d.process_mode.eq("refit")&d.budget.eq(0)][
        ["sample_id","family","predicted_log_nH"]].rename(columns={"predicted_log_nH":"baseline"})
    changes=[]
    for name in study.PROFILES:
        for mode in ("frozen","refit"):
            g=d[d.profile.eq(name)&d.process_mode.eq(mode)&d.budget.eq(0)].merge(base,on=["sample_id","family"])
            g["absolute_change"]=abs(g.predicted_log_nH-g.baseline)
            changes.append(dict(profile=name,process_mode=mode,source_equal_abs_prediction_change=family_mean(g,"absolute_change"),
                                maximum_change=float(g.absolute_change.max())))
    energy.write(OUT/"process_compensation_diagnostic.json",changes)
    old=read_result_csv(energy.BASE/"phase7_inventory/predictions.csv")
    old=old[old.method.eq("eq")]
    r=d[d.profile.eq("repository_default")&d.process_mode.eq("refit")].merge(old,on=["sample_id","budget"],suffixes=("_new","_old"))
    replay=float(abs(r.predicted_log_nH_new-r.predicted_log_nH_old).max());assert len(r)==87 and replay<2e-7
    # Direct/chemical controls are exactly unchanged under a DOS-only change.
    v1=read_result_csv(study.energy.BASE/"phase8_process_temperature/temperature_baselines/predictions.csv")
    v2=read_result_csv(study.energy.BASE/"phase8_process_temperature/temperature_baselines_published_DOS/predictions.csv")
    k=["method","budget","read_id"];merged=v1.merge(v2,on=k,suffixes=("_old","_published"))
    unchanged=merged[merged.method.isin(["chemical_constant","direct_ridge"])]
    unchanged_error=float(abs(unchanged.predicted_old-unchanged.predicted_published).max());assert unchanged_error<1e-12
    pubfits=[json.loads(p.read_text()) for p in (study.energy.BASE/"phase8_process_temperature/temperature_baselines_published_DOS/fits").glob("*.json")]
    source_audit=dict(
        final_publication=dict(doi="10.1039/D4CP04817B",
            url="https://pubs.rsc.org/en/content/articlehtml/2025/cp/d4cp04817b",
            read_section="Temperature dependence of bandgap;verified finalpublisher and cached authorpreprintp31",
            NC_300_cm3=3.33e18,NV_300_cm3=1.71e20,Eg0_eV=5.1),
        local_repository=dict(path="external/KROGER/Ga2O3/KROGER_Set_Ga2O3_Material_Conditions.m",
            sha256=energy.sha(ROOT/"external/KROGER/Ga2O3/KROGER_Set_Ga2O3_Material_Conditions.m"),
            NC_300_cm3=1e19,NV_300_cm3=5e20,Eg0_eV=5.0),
        python_prior=dict(path="src/sandbox/thermo.py",NC_REF=1e19,NV_REF=5e20,EG0=5.0),
        source_difference_is_proven=True,publication_code_reproduction_claim="prior runs reproduce chosen repository-default scenario,not verified finalpublication settings",
        not_claimed="no author-code error,DFT realignment,completepublishedcurve reproduction or universaltrue material parameters is asserted",
        mass_note="donorhydrogenic mass and DOS reference require separate provenance;NC1e19 was not derived from m*.28 here",
        failed_initial_runner="missing perprofile refit/protocol.json before any refit;initial files/log retained;v2 creates perprofile context",
        frozen_scientific_inputs_changed=False)
    energy.write(OUT/"band_input_source_audit.json",source_audit)
    audit=dict(passed=True,profiles=3,material_fits=len(fits),optimizer_starts=sum(v["starts"] for v in fits),
        nonconverged_starts=sum(v["nonconverged"] for v in fits),inner_source_contexts=contexts,
        summed_fit_seconds=sum(v["seconds"] for v in fits),legacy_replay_max_dex=replay,
        unchanged_direct_temperature_prediction_error=unchanged_error,
        published_temperature_fit_records=len(pubfits),
        published_temperature_optimizer_starts=sum(len(v["trials"]) for v in pubfits),
        published_temperature_nonconverged=sum(not t["success"] for v in pubfits for t in v["trials"]),
        all_numeric=bool(s.numeric.eq(s.reads).all()),all_declared_domain=bool(s.physical.eq(s.reads).all()),
        goal_status="active",primary_benchmark_unchanged=True,
        next="explicit material-profile injection and full variable-temperature frozen-stock layer,not another default-parameter grid")
    energy.write(OUT/"contract_audit.json",audit)
    print(s[s.scope.eq("all_query")].to_string(index=False));print(json.dumps(changes,indent=2));print(json.dumps(audit,indent=2))

if __name__=="__main__":main()
