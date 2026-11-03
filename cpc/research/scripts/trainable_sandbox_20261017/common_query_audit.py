from pathlib import Path
import sys
import pandas as pd
from csv_io import read_result_csv

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/trainable_sandbox_20261017"

def main():
    p=read_result_csv(OUT/"predictions_v1.csv")
    v=read_result_csv(OUT/"v2_complex/predictions_v2.csv")
    q2=set(p[(p.budget==2)&p.role.eq("query")].sample_id)
    assert len(q2)==18
    p=p[p.sample_id.isin(q2)];v=v[v.sample_id.isin(q2)]
    records=[]
    for keys,g in pd.concat([p,v],ignore_index=True).groupby(["method","budget","family"]):
        records.append(dict(method=keys[0],budget=int(keys[1]),family=keys[2],n=len(g),
            mae=float(abs(g.predicted_log_nH-g.observed_log_nH).mean())))
    by=pd.DataFrame(records);by.to_csv(OUT/"common_query18_by_family.csv",index=False)
    summary=by.groupby(["method","budget"]).agg(families=("family","nunique"),query_records=("n","sum"),
        family_equal_mae=("mae","mean")).reset_index()
    summary.to_csv(OUT/"common_query18_summary.csv",index=False)
    print(summary[summary.method.isin(["physics_global_process_offset","physics_complex_energy_offset",
        "physics_shared_energy_offset","line_dose_offset","unit_dose_offset"])].to_string(index=False))

if __name__=="__main__":main()
