"""Audit primary comparisons and repair the secondary label eligibility contract.

V1 nH fitting/selection is unchanged. Secondary mu fits in the initial script
accepted one row whose mobility_eligible flag is false. Preserve the original
files and recompute eligible secondary properties under a separately named audit.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_benchmark as b
OUT=b.OUT

def main():
    p=pd.read_csv(OUT/"predictions_v1.csv")
    q=p[p.role.eq("query")].copy()
    assert p.shape[0]==29*3*11
    assert q.shape[0]==(29+23+18)*11
    assert q.groupby(["method","budget"]).family.nunique().eq(5).all()
    assert all(row.family not in row.train_families.split("|") for row in p.itertuples())
    for path in (OUT/"selections").glob("*.json"):
        for choice in json.loads(path.read_text()):
            held=choice["outer_family"]
            fit=json.loads((OUT/"fits"/(choice["fit_key"]+".json")).read_text())
            assert held not in fit["train_families"]
    protected=json.loads((OUT/"protected_inputs_before.json").read_text())
    changed=[x["path"] for x in protected
        if hashlib.sha256((ROOT/x["path"]).read_bytes()).hexdigest()!=x["sha256"]]
    assert not changed,changed
    old=pd.read_csv(ROOT/"results/chain_diagnosis_20261015/executed/carrier_ML_predictions.csv")
    legacy=set(old[old.model.eq("physics_global")&old.comparable].sample_id)
    assert len(legacy)==24
    covered=[]
    for keys,g in q[q.sample_id.isin(legacy)].groupby(["method","budget","family"]):
        covered.append(dict(method=keys[0],budget=keys[1],family=keys[2],n=len(g),
            mae=float(abs(g.predicted_log_nH-g.observed_log_nH).mean())))
    covered=pd.DataFrame(covered);covered.to_csv(OUT/"legacy_common24_by_family_v1.csv",index=False)
    covered.groupby(["method","budget"]).agg(families=("family","nunique"),n=("n","sum"),
        family_equal_mae=("mae","mean")).reset_index().to_csv(OUT/"legacy_common24_summary_v1.csv",index=False)
    # Use inner validation to select within a class, never aggregate outer errors
    # to choose a class representative.
    choices={}
    for path in (OUT/"selections").glob("*.json"):
        for x in json.loads(path.read_text()):
            choices[(x["outer_family"],x["budget"],x["method"])]=x["best"]["inner_mae"]
    class_records=[]
    for (family,budget),g in q.groupby(["family","budget"]):
        for name,prefix in [("empirical_inner_selected",False),("physics_inner_selected",True)]:
            names=[m for m in g.method.unique() if m.startswith("physics")==prefix]
            selected=min(names,key=lambda m:(choices[(family,int(budget),m)],m))
            z=g[g.method.eq(selected)].copy();z["selected_method"]=selected;z["method_class"]=name
            class_records.append(z)
    sel=pd.concat(class_records,ignore_index=True)
    sel.to_csv(OUT/"class_selected_predictions_v1.csv",index=False)
    cs=[]
    for keys,g in sel.groupby(["method_class","budget","family"]):
        cs.append(dict(method_class=keys[0],budget=keys[1],family=keys[2],
            selected_method=g.selected_method.iloc[0],n=len(g),mae=float(abs(g.predicted_log_nH-g.observed_log_nH).mean())))
    cs=pd.DataFrame(cs);cs.to_csv(OUT/"class_selected_by_family_v1.csv",index=False)
    cs.groupby(["method_class","budget"]).agg(families=("family","nunique"),n=("n","sum"),
        family_equal_mae=("mae","mean"),worst_family_mae=("mae","max")).reset_index().to_csv(OUT/"class_selected_summary_v1.csv",index=False)
    # Secondary labels: every mu eligibility flag is honored. No carrier fit,
    # carrier hyperparameter or nH prediction changes.
    repaired=[];transport_cache={}
    source=b.DATA.set_index("sample_id")
    for row in q.itertuples():
        s=source.loc[row.sample_id]
        if not (bool(s.mobility_eligible) and np.isfinite(s.mu_cm2Vs) and s.mu_cm2Vs>0):continue
        fit=json.loads((OUT/"fits"/(row.fit_key+".json")).read_text())
        if row.fit_key not in transport_cache:
            tr=b.DATA[b.DATA.family.isin(fit["train_families"])].copy()
            tr=tr[tr.mobility_eligible.fillna(False).astype(bool)&tr.mu_cm2Vs.gt(0)].copy()
            if fit["kind"].startswith("physics"):
                raw=[b.physical_point(x,fit["parameters"],fit["kind"])[0].values[1] for x in tr.itertuples()]
                offset=float(np.average(np.log10(tr.mu_cm2Vs.to_numpy())-np.array(raw),weights=b.family_weights(tr)))
                transport_cache[row.fit_key]=dict(offset=offset,labels=len(tr),families=tr.family.nunique())
            else:
                X=b.features(tr);mu=np.log10(tr.mu_cm2Vs.to_numpy())
                center=X.mean(axis=0);scale=X.std(axis=0);scale[scale<1e-12]=1.
                alpha=float(fit["spec"].split(":")[1]) if ":" in fit["spec"] else 1.
                m=Ridge(alpha=alpha).fit((X-center)/scale,mu,sample_weight=b.family_weights(tr))
                transport_cache[row.fit_key]=dict(coef=m.coef_,intercept=m.intercept_,center=center,scale=scale,
                    labels=len(tr),families=tr.family.nunique())
        fix=transport_cache[row.fit_key]
        if fit["kind"].startswith("physics"):
            mu=row.predicted_log_muH-fit["transport_offset_dex"]+fix["offset"]
        else:
            X=b.features(pd.DataFrame([s]))
            mu=float((fix["intercept"]+((X-fix["center"])/fix["scale"])@fix["coef"]).item())
        repaired.append(dict(method=row.method,budget=row.budget,family=row.family,sample_id=row.sample_id,
            observed_log_muH=np.log10(s.mu_cm2Vs),predicted_log_muH=mu,
            predicted_log_nH=row.predicted_log_nH,
            observed_log_sigma=np.log10(1.602176634e-19)+row.observed_log_nH+np.log10(s.mu_cm2Vs),
            predicted_log_sigma=np.log10(1.602176634e-19)+row.predicted_log_nH+mu,
            training_mu_labels=fix["labels"],training_mu_families=fix["families"],
            note="nH-frozen secondary refit on eligible mu labels; sigma is derived, not independent validation"))
    aux=pd.DataFrame(repaired);aux.to_csv(OUT/"eligible_secondary_predictions_v1a.csv",index=False)
    ar=[]
    for keys,g in aux.groupby(["method","budget","family"]):
        ar.append(dict(method=keys[0],budget=keys[1],family=keys[2],n=len(g),
            mu_mae=float(abs(g.predicted_log_muH-g.observed_log_muH).mean()),
            sigma_mae=float(abs(g.predicted_log_sigma-g.observed_log_sigma).mean())))
    pd.DataFrame(ar).to_csv(OUT/"eligible_secondary_by_family_v1a.csv",index=False)
    comparisons=[]
    fam=pd.read_csv(OUT/"by_family_v1.csv")
    for budget in (0,1,2):
        a=fam[(fam.budget==budget)&fam.method.eq("physics_shared_process_input_offset")].set_index("family")
        z=fam[(fam.budget==budget)&fam.method.eq("physics_global_process_offset")].set_index("family")
        diff=a.mae-z.mae
        comparisons.append(dict(comparison="shared_input_minus_global_process",budget=budget,
            mean_difference=float(diff.mean()),better_families=int((diff< -1e-8).sum()),
            worse_families=int((diff>1e-8).sum()),ties=int((abs(diff)<=1e-8).sum())))
    pd.DataFrame(comparisons).to_csv(OUT/"paired_comparisons_v1.csv",index=False)
    all_fits=[json.loads(x.read_text()) for x in (OUT/"fits").glob("*.json")]
    fits=pd.DataFrame([dict(key=f["key"],model=f["kind"],train_families="|".join(f["train_families"]),
        parameters=json.dumps(f["parameters"]),seconds=f.get("seconds"),
        fit_success=f.get("fit",{}).get("success"),loss=f.get("fit",{}).get("loss"),
        energy_bound=(abs(f["parameters"][0])>.49999 if f["kind"]=="physics_shared_energy" else None))
        for f in all_fits])
    fits.to_csv(OUT/"fit_audit_v1.csv",index=False)
    audit=dict(primary_prediction_rows=len(p),query_prediction_rows=len(q),outer_families=5,methods=11,
        source_exclusion_checked=True,protected_files=len(protected),changed_protected_files=changed,
        legacy_common_records=len(legacy),primary_nH_results_valid=True,
        initial_secondary_label_issue="one noneligible TEOS mu was accepted; initial mu/sigma scores are superseded by eligible_secondary_*_v1a.csv",
        eligible_mu_labels=int((b.DATA.mobility_eligible.fillna(False).astype(bool)&b.DATA.mu_cm2Vs.gt(0)).sum()),
        eligible_mu_families=int(b.DATA[b.DATA.mobility_eligible.fillna(False).astype(bool)&b.DATA.mu_cm2Vs.gt(0)].family.nunique()),
        secondary_scope="Only two sources support paired dose/mu labels; insufficient for a general full-chain mobility claim",
        variance_scope="conditional finite-grid Gaussian model, not conformal coverage")
    b.write(OUT/"integrity_v1.json",audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
