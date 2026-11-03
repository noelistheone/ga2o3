"""Independently replay the fixed-head lower bound, without altering forecasts."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
from sandbox import transport
from csv_io_v2 import read_result_csv
import phase2_energy as energy


def main():
    out=energy.BASE/"phase12_stock_adaptation/full_property_chain"
    heads=json.loads((out/"head_parameters.json").read_text())
    offsets={h["held"]:h["offset_dex"] for h in heads if h["head"]=="direct_Hall_argument"}
    pred=read_result_csv(out/"predictions.csv")
    pred=pred[pred.model.eq("original_offset")&pred["head"].eq("direct_Hall_argument")]
    bounds=read_result_csv(out/"head_capacity_bounds.csv").set_index(["sample_id","budget"])
    summary=read_result_csv(out/"head_capacity_summary.csv").set_index("budget")
    errors=[];rows=[]
    assert transport.CT_MU_MIN==82. and transport.CT_MU_MAX==205.
    for r in pred.itertuples():
        lo=np.log10(transport.CT_MU_MIN)+offsets[r.family]
        hi=np.log10(transport.CT_MU_MAX)+offsets[r.family]
        lower=max(lo-r.observed_log_mu,0.,r.observed_log_mu-hi)
        actual=abs(r.predicted_log_mu-r.observed_log_mu)
        # Oracle input is used only in this diagnostic, never in the method.
        oracle=abs(np.log10(transport.mu_caughey_thomas(10**r.observed_log_nH,300.))+
            offsets[r.family]-r.observed_log_mu)
        saved=bounds.loc[r.sample_id,r.budget]
        for key,value in (("log_mu_min",lo),("log_mu_max",hi),("observed_log_mu",r.observed_log_mu),
            ("minimum_possible_log_error",lower),("actual_log_error",actual)):
            errors.append(abs(value-saved[key]))
        assert lower<=actual+1e-12
        assert lo-1e-12<=r.predicted_log_mu<=hi+1e-12
        rows.append(dict(budget=r.budget,family=r.family,lower_bound=lower,actual_error=actual,
            same_query_perfect_n_error=oracle))
    import pandas as pd
    by=pd.DataFrame(rows).groupby(["budget","family"])[
        ["lower_bound","actual_error","same_query_perfect_n_error"]].mean()
    for budget,r in by.groupby(level=0).mean().iterrows():
        for key in r.index:errors.append(abs(r[key]-summary.loc[budget,key]))
    assert len(pred)==11 and max(errors)<1e-12
    inputs=("head_parameters.json","predictions.csv","head_capacity_bounds.csv","head_capacity_summary.csv")
    energy.write(out/"head_capacity_replay_audit.json",dict(passed=True,created_utc=energy.now(),
        code_sha256=energy.sha(Path(__file__)),transport_code_sha256=energy.sha(ROOT/"src/sandbox/transport.py"),
        input_hashes={f:energy.sha(out/f) for f in inputs},rows=len(pred),
        maximum_numeric_reconstruction_error=max(errors),no_forecasts_modified=True,
        oracle_scope="diagnostic-only perfectcarrier argument;never available to the evaluated predictor",
        interpretation="fixed calibrated empirical-head lower bound;not arbitrarytransport impossibility or unique physical mechanism"))
    print("Fixed-head capacity replay passed",len(pred),"rows; max error",max(errors))


if __name__=="__main__":main()
