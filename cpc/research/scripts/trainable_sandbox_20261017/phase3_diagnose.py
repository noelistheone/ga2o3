"""Post-evaluation Taylor and coverage diagnostics; never used in frozen fits."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase3_metric as metric

def main():
    metric.frozen()
    db,rr,blocks=metric.blocks()
    data=read_result_csv(metric.OUT/"energy_predictions.csv")
    p=read_result_csv(metric.OUT/"virtual_property_predictions.csv")
    scope=[]
    for (method,e),g in p.groupby(["method","element"]):
        q=g[g.reference_eligible]
        scope.append(dict(method=method,element=e,reference_cases=len(q),
            numeric=int(q.predicted_log_nH.notna().sum()),
            physical=int(q.predicted_physical_valid.sum()),
            mathematical_MAE=float(abs(q.predicted_log_nH-q.reference_log_nH).mean()) if len(q) else None))
    by=pd.DataFrame(scope);by.to_csv(metric.OUT/"virtual_coverage_by_element.csv",index=False)
    summary=by.groupby("method").agg(reference_cases=("reference_cases","sum"),
        numeric=("numeric","sum"),physical=("physical","sum"),
        reference_elements=("mathematical_MAE","count"),
        mathematical_macro_MAE=("mathematical_MAE","mean")).reset_index()
    summary["full_physical_coverage"]=summary.physical.eq(summary.reference_cases)
    summary.to_csv(metric.OUT/"virtual_coverage_summary.csv",index=False)
    rows=[]
    methods=("kernel_energy","kernel_diagonal_l1","kernel_full_l1")
    for e in sorted(blocks):
        rec=[v for v in rr if v["element"]==e];truth=np.array([v["crossing_eV"] for v in rec])
        if not blocks[e]["eval_eligible"].any():continue
        y,J,valid,dg=metric.forward(db,rec,truth,metric.EVAL,True,True)
        for method in methods:
            d=data[data.element.eq(e)&data.method.eq(method)]
            lookup=dict(zip(d.label_id,d.predicted_crossing_eV))
            pred=np.array([lookup[v["label_id"]] for v in rec]);h=pred-truth
            q=p[p.element.eq(e)&p.method.eq(method)]
            old={(r.Tf_K,r.dose_cm3):r for r in q.itertuples()}
            for i,(T,C) in enumerate(metric.EVAL):
                if not valid[i]:continue
                r=old[(T,C)]
                linear=float(J[i]@h);actual=float(r.predicted_log_nH-y[i])
                rows.append(dict(element=e,method=method,Tf_K=T,dose_cm3=C,
                    crossing_error_L2_eV=float(np.linalg.norm(h)),
                    crossing_max_error_eV=float(abs(h).max()),
                    linearized_property_error=linear,actual_property_error=actual,
                    Taylor_remainder_dex=actual-linear,
                    predicted_physical=bool(r.predicted_physical_valid)))
    energy.csvwrite(metric.OUT/"posthoc_Taylor_diagnostics.csv",rows)
    t=pd.DataFrame(rows)
    agg=t.groupby("method").agg(cases=("element","count"),
        mean_absolute_linear_error=("linearized_property_error",lambda s:float(abs(s).mean())),
        mean_absolute_actual_error=("actual_property_error",lambda s:float(abs(s).mean())),
        mean_absolute_Taylor_remainder=("Taylor_remainder_dex",lambda s:float(abs(s).mean())),
        worst_Taylor_remainder=("Taylor_remainder_dex",lambda s:float(abs(s).max()))).reset_index()
    agg.to_csv(metric.OUT/"posthoc_Taylor_summary.csv",index=False)
    audit=dict(timing="post-evaluation diagnostic, never imported by training/selection",
        frozen_training_unchanged=True,
        interpretation="all finite numerical logn/rH values are mathematical extensions; outside single-electron/detection validity they cannot be reported as usable Hall predictions",
        next_hypothesis="test full nonlinear training loss with analytic implicit gradients and line-search checks; do not infer success from Taylor algebra")
    energy.write(metric.OUT/"posthoc_diagnostic_contract.json",audit)
    print(summary.to_string(index=False));print(agg.to_string(index=False))
if __name__=="__main__":main()
