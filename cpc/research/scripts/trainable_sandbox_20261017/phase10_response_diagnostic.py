"""Post-result physical response diagnosis; no labels used to update models."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import phase10_internal_adaptation as study
import phase2_energy as energy


def main():
    rows=[]
    for held in study.FAMILIES:
        _,base=study.base_fit([f for f in study.FAMILIES if f!=held],"learned_screen_joint")
        data=study.PRED[study.PRED.family.eq(held)&study.PRED.kind.eq("RT")]
        extra=[]
        for sample in study.prior.SPECS[study.prior.SPECS.family.eq(held)].index:
            q=study.PRED[study.PRED.sample_id.eq(sample)&study.PRED.kind.eq("curve")]
            extra.append(q.loc[q.Tobs_K.idxmin()])
        if extra:data=pd.concat([data,pd.DataFrame(extra)],ignore_index=True)
        for r in data.itertuples():
            theta=study.initial_coordinates(base);a,J=study.physical_point(r,base,theta)
            rows.append(dict(sample_id=r.sample_id,family=held,element=r.element,obs_id=r.obs_id,
                kind=r.kind,Tobs_K=r.Tobs_K,chemical_dose_cm3=r.incorporated_cm3,
                observed=r.target,predicted=float(a.values[0]),
                dlogn_dlogTf=float(J[0,0]),dlogn_dlogE0=float(J[0,1]),
                absolute_charge_cm3=a.diagnostics["ionized_absolute_charge_cm3"],
                q2_charge_cm3=a.diagnostics["charge_squared_scatter_budget_cm3"],
                background_cm3=a.diagnostics["Nd_cm3"],
                above_reference_floor=a.diagnostics["above_carrier_detection_floor"]))
    d=pd.DataFrame(rows);d.to_csv(study.OUT/"physical_response_diagnostic.csv",index=False)
    rt=d[d.kind.eq("RT")].set_index("sample_id")
    ratios=[]
    for r in d[d.kind.eq("curve")].itertuples():
        ratios.append(dict(sample_id=r.sample_id,family=r.family,
            RT_energy_sensitivity=abs(float(rt.loc[r.sample_id,"dlogn_dlogE0"])),
            cold_energy_sensitivity=abs(r.dlogn_dlogE0),
            cold_over_RT=abs(r.dlogn_dlogE0)/max(abs(float(rt.loc[r.sample_id,"dlogn_dlogE0"])),1e-20)))
    energy.write(study.OUT/"paired_temperature_sensitivity.json",ratios)
    adjacent=[]
    for family,g in d[d.kind.eq("RT")].groupby("family"):
        g=g.sort_values("chemical_dose_cm3")
        for i in range(1,len(g)):
            a,b=g.iloc[i-1],g.iloc[i]
            denom=np.log10(b.chemical_dose_cm3/a.chemical_dose_cm3)
            if abs(denom)<1e-8:continue
            adjacent.append(dict(family=family,sample_low=a.sample_id,sample_high=b.sample_id,
                dose_ratio=b.chemical_dose_cm3/a.chemical_dose_cm3,
                observed_between_specimen_slope=(b.observed-a.observed)/denom,
                predicted_between_specimen_slope=(b.predicted-a.predicted)/denom,
                caveat="not controlled dose derivative;unreported process/history may differ"))
    energy.csvwrite(study.OUT/"between_specimen_dose_diagnostic.csv",adjacent)
    protocol=dict(created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),
        parent_protocol_sha256=energy.sha(study.OUT/"protocol.json"),
        timing="post primary outcomes,diagnosis only,no retraining/reselection",
        states=len(d),scope="all29RTsourceexcluded base states plus lowestT7oldspecimens",
        RT_median_abs_energy_response=float(abs(d[d.kind.eq("RT")].dlogn_dlogE0).median()),
        paired_ratios=ratios,interpretation="model sensitivities conditional on supplied inputs,not unique real failure attribution")
    energy.write(study.OUT/"physical_response_diagnostic.json",protocol)
    print(json.dumps({k:v for k,v in protocol.items() if k!="paired_ratios"},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
