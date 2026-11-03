"""Matched full property chains; empirical Hall argument is explicit.

The legacy head's heuristic Hall conversion is retained as a feature transform.
The primary head uses the Hall argument directly, consistent with the unit-Hall
carrier scenario. Neither head infers microscopic scattering from charge counts.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
from sandbox import transport
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import transport_chain as legacy

OUT=energy.BASE/"phase12_stock_adaptation";FINAL=OUT/"physical_scope_v2";DEST=OUT/"full_property_chain"


def logmu(logn,head):
    if head=="direct_Hall_argument":return float(np.log10(transport.mu_caughey_thomas(10**logn,300.)))
    return float(legacy.logCT(logn))


def main():
    assert json.loads((FINAL/"contract_audit.json").read_text())["passed"]
    files=[Path(__file__),ROOT/"src/sandbox/transport.py",ROOT/"scripts/trainable_sandbox_20261017/transport_chain.py"]
    inputs=[FINAL/"predictions.csv",energy.BASE/"dataset_v1.csv",
        ROOT/"results/chain_diagnosis_20261015/executed/mobility_bounds_and_inputs.csv"]
    protocol=dict(created_utc=energy.now(),version=1,code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in files},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        primary_head="CT measured-Hall argument +source-excluded weightedmedianoffset;declared empirical Hall curve,not true ionized-density label",
        legacy_head="reproduce old heuristic Hall inversion as separate empirical argumenttransform;not the rH1 physical state",
        paired_validation="exact5original specimenIDs/2DOI,all0-1-2sourceanchor budgets;no nearest joining,no new mobility labels",
        training="52conditional nH/mu pairs11families;exclude targetDOI;all20carrier methods receive same head",
        prediction="query measurednH/mu neverinput;only forecastnH and saved non-target-headparameters",
        sigma="q*nH*muH derived on both sides,not independent measurement or validation of microscopic scattering")
    pp=DEST/"protocol.json"
    if pp.exists():assert json.loads(pp.read_text())["code_hashes"]==protocol["code_hashes"]
    else:energy.write(pp,protocol)
    pred=read_result_csv(inputs[0]);data=read_result_csv(inputs[1]).set_index("sample_id");mu=read_result_csv(inputs[2])
    eligible=set(data[data.mobility_eligible.fillna(False)&data.mu_cm2Vs.gt(0)].index)
    assert len(mu)==52 and mu.family.nunique()==11 and len(eligible)==5
    assert eligible<=set(mu.sample_id)
    fits={};conditional=[]
    for head in ("direct_Hall_argument","legacy_heuristic_argument"):
        baseline=np.array([logmu(x,head) for x in np.log10(mu.carrier_cm3)])
        for family in sorted(set(mu.family)|set(data.family)):
            tr=mu[mu.family.ne(family)];w=1/tr.family.map(tr.family.value_counts()).to_numpy(float)
            idx=mu.family.ne(family).to_numpy()
            offset=legacy.median(np.log10(tr.mu_cm2Vs).to_numpy(float)-baseline[idx],w)
            fits[head,family]=dict(head=head,held=family,offset_dex=float(offset),
                training_records=len(tr),training_families=sorted(tr.family.unique()),training_specimens=tr.sample_id.tolist())
            assert family not in fits[head,family]["training_families"]
        for i,row in enumerate(mu.itertuples()):
            f=fits[head,row.family]
            conditional.append(dict(head=head,family=row.family,sample_id=row.sample_id,
                observed=np.log10(row.mu_cm2Vs),predicted=baseline[i]+f["offset_dex"]))
    rows=[]
    for row in pred[pred.role.eq("query")&pred.sample_id.isin(eligible)].itertuples():
        for head in ("direct_Hall_argument","legacy_heuristic_argument"):
            fit=fits[head,row.family];valid=bool(np.isfinite(row.predicted))
            prediction=logmu(row.predicted,head)+fit["offset_dex"] if valid else np.nan
            target_mu=float(np.log10(data.loc[row.sample_id,"mu_cm2Vs"]))
            rows.append(dict(model=row.model,budget=int(row.budget),head=head,family=row.family,sample_id=row.sample_id,
                predicted_log_nH=row.predicted,observed_log_nH=row.observed,predicted_log_mu=prediction,observed_log_mu=target_mu,
                predicted_log_sigma=np.log10(transport.E)+row.predicted+prediction,
                observed_log_sigma=np.log10(transport.E)+row.observed+target_mu,
                head_training_records=fit["training_records"],target_source_excluded=True,
                target_measured_nH_not_used=True,conductivity_is_derived=True))
    result=pd.DataFrame(rows);by=[]
    for keys,g in result.groupby(["model","head","budget","family"]):
        r=dict(model=keys[0],head=keys[1],budget=keys[2],family=keys[3],queries=len(g))
        for endpoint in ("nH","mu","sigma"):
            err=abs(g["predicted_log_"+endpoint]-g["observed_log_"+endpoint])
            r[endpoint+"_MAE"]=float(err.mean()) if np.isfinite(err).all() else np.inf
        by.append(r)
    by=pd.DataFrame(by);summary=by.groupby(["model","head","budget"]).agg(
        nH_MAE=("nH_MAE","mean"),mu_MAE=("mu_MAE","mean"),sigma_MAE=("sigma_MAE","mean"),
        sources=("family","nunique"),queries=("queries","sum")).reset_index()
    conditional=pd.DataFrame(conditional)
    cs=[]
    for head,g in conditional.groupby("head"):
        score=float(abs(g.predicted-g.observed).groupby(g.family).mean().mean())
        cs.append(dict(head=head,MAE=score,records=len(g),sources=g.family.nunique()))
        if head=="legacy_heuristic_argument":assert abs(score-.3666089)<2e-6
    for d,name in ((result,"predictions"),(by,"by_source"),(summary,"summary"),(conditional,"conditional_predictions"),(pd.DataFrame(cs),"conditional_summary")):
        d.to_csv(DEST/(name+".csv"),index=False)
    energy.write(DEST/"head_parameters.json",list(fits.values()))
    audit=dict(passed=True,created_utc=energy.now(),eligible_samples=sorted(eligible),chemical_paired_specimens=5,
        independent_DOI_sources=2,conditional_training_pairs=52,conditional_sources=11,
        full_prediction_rows=len(result),query_measured_nH_not_input=True,target_source_mu_labels_excluded=True,
        independent_sigma_labels=0,old_heuristic_conditional_reference_reproduced=True,
        primary_convention="empirical Hall argument,unit-Hall carrier scenario;not a microscopic Ni-to-mu validation",
        no_new_mobility_data=True,goal_complete=False)
    energy.write(DEST/"audit.json",audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2));print(pd.DataFrame(cs).to_string(index=False))


if __name__=="__main__":main()
