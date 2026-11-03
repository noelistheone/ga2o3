"""Aggregate the completed V2 predictions using Python's CSV reader.

The local pandas C CSV parser segfaulted on the already-written outer files.
This changes only file ingestion, preserving the learned fits and predictions.
"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/trainable_sandbox_20261017/v2_complex"

def main():
    records=[];inputs=[]
    for p in sorted((OUT/"outer").glob("*.csv")):
        with p.open(newline="") as f:rows=list(csv.DictReader(f))
        records+=rows
        inputs.append(dict(path=str(p.relative_to(ROOT)),rows=len(rows),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    assert len(inputs)==5 and len(records)==29*3*2
    d=pd.DataFrame.from_records(records)
    textual={"method","family","sample_id","role","fit_key","train_families"}
    for c in d.columns:
        if c not in textual:
            d[c]=pd.to_numeric(d[c].replace({"True":"1","False":"0"}),errors="raise")
    assert all(r.family not in r.train_families.split("|") for r in d.itertuples())
    d.to_csv(OUT/"predictions_v2.csv",index=False)
    q=d[d.role.eq("query")].copy();rows=[]
    assert len(q)==(29+23+18)*2
    for keys,g in q.groupby(["method","budget","family"]):
        e=g.predicted_log_nH-g.observed_log_nH
        rows.append(dict(method=keys[0],budget=int(keys[1]),family=keys[2],n=len(g),
            mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),
            physically_covered=int(g.physical_valid_mass.ge(.95).sum())))
    by=pd.DataFrame(rows);by.to_csv(OUT/"by_family_v2.csv",index=False)
    summary=by.groupby(["method","budget"]).agg(families=("family","nunique"),n=("n","sum"),
        family_equal_mae=("mae","mean"),worst_family_mae=("mae","max"),
        physically_covered=("physically_covered","sum")).reset_index()
    summary.to_csv(OUT/"summary_v2.csv",index=False)
    old=pd.read_csv(ROOT/"results/trainable_sandbox_20261017/by_family_v1.csv")
    comparisons=[]
    for budget in (0,1,2):
        for method in by.method.unique():
            new=by[by.budget.eq(budget)&by.method.eq(method)].set_index("family")
            base=old[old.budget.eq(budget)&old.method.eq("physics_global_process_offset")].set_index("family")
            diff=new.mae-base.mae
            comparisons.append(dict(budget=budget,method=method,comparison="minus V1 global process + offset",
                delta_mae=float(diff.mean()),better_families=int((diff < -1e-8).sum()),
                worse_families=int((diff > 1e-8).sum()),ties=int((abs(diff)<=1e-8).sum())))
    pd.DataFrame(comparisons).to_csv(OUT/"paired_vs_global_v2.csv",index=False)
    note=dict(completed_outer_families=5,primary_prediction_rows=len(d),query_prediction_rows=len(q),
        inputs=inputs,source_exclusion_checked=True,
        io_incident="Two exit-139 failures in local pandas C read_csv during aggregation; faulthandler stack retained.",
        resolution="Python csv.DictReader + explicit numeric conversion, including boolean valid-mass strings as 0/1; no fits or predictions changed.",
        C_parser_log="../benchmark_v2_resume.log",fit_and_protocol_version_unchanged=True)
    (OUT/"aggregation_integrity_v2.json").write_text(json.dumps(note,ensure_ascii=False,indent=2)+"\n")
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))

if __name__=="__main__":main()
