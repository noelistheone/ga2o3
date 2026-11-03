"""Frozen related-study transfer, separating full-chain and conditional transport."""
from pathlib import Path
from itertools import product
import sys,json,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.special import expit
from sandbox.energy_learning import reconstruct
from sandbox.property_metric_learning import element_database
from sandbox.implicit_defects import ImplicitDefectLayer
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase3_metric as metric
import phase3_property as models
import phase3_new_source as source
import transport_chain as transport_head
from phase2_property import digest

OUT=source.OUT
RELATED=source.RELATED
METHODS=list(models.METHODS)+["P2_ET_goal_k2","P2_Ridge_goal_k2","P2_mean_goal_k2",
                             "identity_dose","unit_dose","line_dose"]

def freeze():
    source.initialize();metric.frozen()
    p=OUT/"azizie2023_prediction_protocol.json"
    payload=dict(version=1,created_utc=energy.now(),methods=METHODS,
        source_protocol_sha256=energy.sha(OUT/"azizie2023_validation_protocol.json"),
        code_sha256=energy.sha(Path(__file__)),
        related_study_exclusion=RELATED,
        model_selection="reuse each old held5.0101132 source fit and shrinkage, trained on the other4 sources; no new-study tuning",
        empirical_controls="identity, source-equal unit-dose offset and source-equal dose line; shrinkage by inner exclusion within same4 sources",
        paired_input_scenarios="primary rounded text C2e18/3e19; fixed3x3 rounding resolution scenarios, not confidence bounds",
        primary_pO2_atm=1e-5,oxygen_status="unmeasured effective-O2 scenario, not actual distilled-ozone chemical potential",
        full_chain="predict nH from chemical input; transport head uses predicted nH only; no target mu anchors",
        transport="same CT+family-weighted median rule trained on original52 pairs excluding related old study, not the new12",
        conditional="all12 pairs use measured nH only in a separate conditional transport table",
        sigma="derived from nH*mu, not independent experiment",
        numeric_and_physical_coverage_separate=True,
        grade="retrospective corrected-label related-study transfer, not blind independent-lab validation",
        input_hashes={str(q.relative_to(ROOT)):energy.sha(q) for q in
            (OUT/"azizie2023_audited_samples.csv",energy.BASE/"dataset_v1.csv",
             ROOT/"results/chain_diagnosis_20261015/executed/mobility_bounds_and_inputs.csv")})
    if p.exists():
        assert json.loads(p.read_text())["code_sha256"]==payload["code_sha256"]
        assert json.loads(p.read_text())["input_hashes"]==payload["input_hashes"]
    else:energy.write(p,payload)

def library_and_fold(method):
    if method.startswith("P2_"):
        import phase2_acquired_property as ap
        import phase2_acquisition as aq
        db,rr,_=energy.records();targets=[r for r in rr if r["element"]=="Si"]
        if method=="P2_ET_goal_k2":
            name="ET_goal_k2";folder=energy.OUT/"acquired_property";db=ap.acquired_database("Si",name)
        elif method=="P2_mean_goal_k2":
            name="mean_goal_k2";folder=energy.OUT/"acquired_property";db=ap.acquired_database("Si",name)
        else:
            name="Ridge_diagonal_xi_k2";folder=energy.OUT/"acquired_property_addendum"
            mean=np.load(energy.OUT/"acquisition_ablation/states/Si__ridge__diagonal_xi.npz")["means"][2]
            db=reconstruct(db,targets,mean)
        reveals=2
    else:
        name=method;folder=metric.OUT/"real_property";db=models.database("Si",method)
        reveals=0
    raw=read_result_csv(folder/"outer"/f"{digest(RELATED)}__{name}.csv")
    key=raw.fit_key.iloc[0]
    fit=json.loads((folder/"fits"/f"{key}.json").read_text())
    assert RELATED not in fit["train_families"] and source.DOI not in fit["train_families"]
    choices=json.loads((folder/"selections"/f"{digest(RELATED)}__{name}.json").read_text())
    lam={int(x["budget"]):x["selected"]["lam"] if x["selected"]["lam"] is not None else np.inf for x in choices}
    active,indices=element_database(db,"Si")
    return ImplicitDefectLayer(active,("Si",)),fit,lam,reveals

def main():
    freeze()
    new=read_result_csv(OUT/"azizie2023_audited_samples.csv")
    pair=new[new.paired_dose_eligible].copy().sort_values("incorporated_cm3")
    assert list(pair.sample_label)==["e","k"]
    mobility=read_result_csv(ROOT/"results/chain_diagnosis_20261015/executed/mobility_bounds_and_inputs.csv")
    mobility=mobility[mobility.family.ne(RELATED)&mobility.family.ne(source.DOI)]
    weights=1/mobility.family.map(mobility.family.value_counts()).to_numpy(float)
    base=np.array([transport_head.logCT(x) for x in np.log10(mobility.carrier_cm3)])
    offset=transport_head.median(np.log10(mobility.mu_cm2Vs).to_numpy()-base,weights)
    conditional=[]
    for row in new.itertuples():
        pred=transport_head.logCT(np.log10(row.measured_nH_cm3))+offset
        conditional.append(dict(sample_id=row.sample_id,observed_log_mu=np.log10(row.measured_muH_cm2Vs),
            predicted_log_mu=pred,scope="given measured nH, held related study excluded"))
    energy.csvwrite(OUT/"azizie2023_conditional_transport.csv",conditional)
    old=read_result_csv(energy.BASE/"dataset_v1.csv")
    tr=old[old.family.ne(RELATED)]
    def empirical_fit(d,kind):
        w=1/d.family.map(d.family.value_counts()).to_numpy(float);w/=w.sum()
        x=d.log_dose.to_numpy(float);y=d.observed_log_nH.to_numpy(float)
        if kind=="identity_dose":return np.array([0.,1.])
        if kind=="unit_dose":return np.array([float(w@(y-x)),1.])
        X=np.c_[np.ones(len(x)),x-18.]
        b=np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        return np.array([b[0]-18*b[1],b[1]])
    roles=read_result_csv(energy.BASE/"episodes_v1.csv")
    def select_empirical_lam(kind):
        scores=[]
        for lam in (.1,1.,10.,np.inf):
            errs=[]
            for held in sorted(tr.family.unique()):
                train=tr[tr.family.ne(held)];test=tr[tr.family.eq(held)]
                w=empirical_fit(train,kind);p=w[0]+w[1]*test.log_dose.to_numpy(float)
                role=roles[roles.family.eq(held)&roles.budget.eq(1)]
                rmap=dict(zip(role.sample_id,role.role))
                a=np.array([rmap[s]=="anchor" for s in test.sample_id])
                q=np.array([rmap[s]=="query" for s in test.sample_id])
                delta=0. if np.isinf(lam) else float((test.observed_log_nH.to_numpy(float)[a]-p[a]).sum()/(a.sum()+lam))
                errs.append(float(abs(p[q]+delta-test.observed_log_nH.to_numpy(float)[q]).mean()))
            scores.append((float(np.mean(errs)),lam))
        return min(scores,key=lambda x:(round(x[0],10),-x[1]))[1]
    chemical=[[pair.iloc[j].rounded_C_low,pair.iloc[j].incorporated_cm3,pair.iloc[j].rounded_C_high] for j in range(2)]
    rows=[];model_provenance={}
    for method in METHODS:
        if method in ("identity_dose","unit_dose","line_dose"):
            coeff=empirical_fit(tr,method);lam={0:np.inf,1:select_empirical_lam(method)};reveals=None
            layer=None;fit=dict(train_families=sorted(tr.family.unique()),coefficients=coeff)
        else:layer,fit,lam,reveals=library_and_fold(method)
        model_provenance[method]=dict(training_families=fit["train_families"],
            fit_key=fit.get("key"),shrinkage_lambda={str(k):v if np.isfinite(v) else "infinity" for k,v in lam.items()},
            energy_reveals=reveals)
        for scenario in product(range(3),repeat=2):
            C=np.array([chemical[j][scenario[j]] for j in range(2)])
            midpoint=scenario==(1,1)
            if layer is None:
                raw=coeff[0]+coeff[1]*np.log10(C);physical=[None,None]
            else:
                s=expit(fit["parameters"][0]);logT=models.fitter.L0+(models.fitter.L1-models.fitter.L0)*s
                pred=[layer.predict(layer.parameters(T=10**logT,Nd=10**fit["parameters"][1],
                        doses=(v,),pO2=1e-5)) for v in C]
                raw=np.array([v.values[0] for v in pred])
                physical=[v.diagnostics["single_electron_Hall_valid"] and v.diagnostics["above_carrier_detection_floor"] for v in pred]
            for budget in (0,1):
                delta=0. if budget==0 or np.isinf(lam[1]) else float((np.log10(pair.iloc[0].measured_nH_cm3)-raw[0])/(1+lam[1]))
                final=raw+delta
                for j,row in enumerate(pair.itertuples()):
                    pm=transport_head.logCT(final[j])+offset
                    rows.append(dict(method=method,budget=budget,scenario=f"{scenario[0]}_{scenario[1]}",
                        primary_midpoint=midpoint,sample_id=row.sample_id,sample_label=row.sample_label,
                        role="anchor" if budget==1 and j==0 else "query",chemical_cm3=C[j],
                        observed_log_nH=np.log10(row.measured_nH_cm3),predicted_log_nH=final[j],
                        observed_log_mu=np.log10(row.measured_muH_cm2Vs),predicted_log_mu=pm,
                        observed_log_sigma=np.log10(1.602176634e-19*row.measured_nH_cm3*row.measured_muH_cm2Vs),
                        predicted_log_sigma=np.log10(1.602176634e-19)+final[j]+pm,
                        physical_valid=physical[j],offset_dex=delta,extra_energy_labels=reveals))
    data=pd.DataFrame(rows);data.to_csv(OUT/"azizie2023_full_chain_predictions.csv",index=False)
    summary=[]
    for (m,b),g in data[data.role.eq("query")].groupby(["method","budget"]):
        primary=g[g.primary_midpoint]
        entry=dict(method=m,budget=int(b),query_samples=len(primary))
        for name in ("nH","mu","sigma"):
            errs=abs(g["predicted_log_"+name]-g["observed_log_"+name])
            byscenario=errs.groupby(g.scenario).mean()
            entry[name+"_MAE"]=float(abs(primary["predicted_log_"+name]-primary["observed_log_"+name]).mean())
            entry[name+"_scenario_min"]=float(byscenario.min())
            entry[name+"_scenario_max"]=float(byscenario.max())
        summary.append(entry)
    pd.DataFrame(summary).to_csv(OUT/"azizie2023_full_chain_summary.csv",index=False)
    energy.write(OUT/"azizie2023_prediction_audit.json",dict(
        model_provenance=model_provenance,transport_offset_dex=offset,
        transport_training_records=len(mobility),transport_training_families=sorted(mobility.family.unique()),
        conditional_pairs=12,full_chain_pairs=2,
        conditional_mobility_MAE=float(np.mean([abs(r["predicted_log_mu"]-r["observed_log_mu"]) for r in conditional])),
        no_query_measured_nH_used_in_full_chain=True,no_target_mu_used_in_training=True,
        current_cohort_or_related_source_not_pooled=True,
        chemical_inputs_are_rounded_prose_not_Hall_reverse_calculation=True,
        intervals_are_input_resolution_scenarios_not_confidence_bounds=True,
        full_chain_prediction_rows=len(rows),two_anchor_queries=0))
    print(pd.DataFrame(summary)[["method","budget","nH_MAE","mu_MAE","sigma_MAE"]].to_string(index=False))
if __name__=="__main__":main()
