"""Complete dose->nH->mu->derived sigma on explicitly matched old specimens."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase4_property as study
import transport_chain as head

OUT=study.training.OUT/"transport"

def main():
    inputs=(study.OUT/"predictions.csv",energy.BASE/"dataset_v1.csv",
        ROOT/"results/chain_diagnosis_20261015/executed/mobility_bounds_and_inputs.csv")
    protocol=dict(version=1,created_utc=energy.now(),
        code_sha256=energy.sha(Path(__file__)),
        helper_sha256=energy.sha(Path(head.__file__)),
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        task="dose/element -> predicted nH -> CT+source-excluded weighted-median transport -> derived sigma",
        eligibility="same explicit specimen IDs and mobility_eligible as corrected phase1, no nearest joining",
        no_target_measured_nH_in_transport=True,
        sigma="derived q*nH*mu, not independent measured conductivity",
        scope="empirical conditional transport head, not new microscopic scattering information")
    p=OUT/"protocol.json"
    if p.exists():
        old=json.loads(p.read_text())
        assert old["code_sha256"]==protocol["code_sha256"]
        assert old["input_hashes"]==protocol["input_hashes"]
    else:energy.write(p,protocol)
    primary=read_result_csv(inputs[0]);data=read_result_csv(inputs[1]).set_index("sample_id")
    mobility=read_result_csv(inputs[2])
    valid=set(data[data.mobility_eligible.fillna(False).astype(bool)&data.mu_cm2Vs.gt(0)].index)
    assert len(valid)==5 and valid.issubset(set(mobility.sample_id))
    fits={}
    for family in sorted(data.loc[list(valid)].family.unique()):
        tr=mobility[mobility.family.ne(family)]
        w=1/tr.family.map(tr.family.value_counts()).to_numpy(float)
        residual=np.log10(tr.mu_cm2Vs).to_numpy()-np.array([head.logCT(x) for x in np.log10(tr.carrier_cm3)])
        fits[family]=dict(offset_dex=head.median(residual,w),training_records=len(tr),
                          training_families=sorted(tr.family.unique()))
        assert family not in fits[family]["training_families"]
    rows=[]
    for r in primary[primary.role.eq("query")&primary.sample_id.isin(valid)].itertuples():
        f=fits[r.family]
        numeric=bool(np.isfinite(r.predicted_log_nH))
        mu=head.logCT(r.predicted_log_nH)+f["offset_dex"] if numeric else np.nan
        observed_mu=float(np.log10(data.loc[r.sample_id,"mu_cm2Vs"]))
        rows.append(dict(method=r.method,budget=int(r.budget),family=r.family,sample_id=r.sample_id,
            predicted_log_nH=r.predicted_log_nH,observed_log_nH=r.observed_log_nH,
            predicted_log_mu=mu,observed_log_mu=observed_mu,
            predicted_log_sigma=np.log10(1.602176634e-19)+r.predicted_log_nH+mu,
            observed_log_sigma=np.log10(1.602176634e-19)+r.observed_log_nH+observed_mu,
            numeric=numeric,physical_valid=bool(r.physical_valid),
            no_target_measured_nH_in_transport=True))
    result=pd.DataFrame(rows);result.to_csv(OUT/"full_chain_predictions.csv",index=False)
    by=[]
    for (m,b,f),g in result.groupby(["method","budget","family"]):
        entry=dict(method=m,budget=int(b),family=f,n=len(g),numeric=int(g.numeric.sum()),
                   physical=int(g.physical_valid.sum()))
        for q in ("nH","mu","sigma"):
            err=abs(g["predicted_log_"+q]-g["observed_log_"+q])
            entry[q+"_MAE"]=float(err.mean()) if err.notna().all() else None
        by.append(entry)
    family=pd.DataFrame(by);family.to_csv(OUT/"by_family.csv",index=False)
    summary=family.groupby(["method","budget"]).agg(families=("family","nunique"),
        records=("n","sum"),numeric=("numeric","sum"),physical=("physical","sum"),
        nH_MAE=("nH_MAE","mean"),mu_MAE=("mu_MAE","mean"),sigma_MAE=("sigma_MAE","mean")).reset_index()
    summary.to_csv(OUT/"summary.csv",index=False)
    energy.write(OUT/"audit.json",dict(eligible_specimens=sorted(valid),specimen_count=len(valid),
        independent_DOI_families=int(data.loc[list(valid)].family.nunique()),transport_fits=fits,
        predicted_rows=len(rows),query_nH_not_used_as_input=True,derived_sigma_only=True))
    print(summary.to_string(index=False))

if __name__=="__main__":main()
