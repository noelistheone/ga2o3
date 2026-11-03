"""Scenario interface for the historical hybrid API.

Review repair 2026-10-15: corpus-unit/pairing errors invalidate the former hard-coded
accuracy bars. No experimental certification or interval coverage is claimed here.
Use reviewed_transport.predict for explicit incorporated-density/300 K comparisons.
"""
from __future__ import annotations
import math
from . import full_forward, kroger_db

TF_GLOBAL = 1500.0  # assumed effective temperature, not a universal measured constant
KROGER = set(kroger_db.ELEMENTS)
PROPERTY_GRADES = {
    "hall_mu_cm2Vs":"hall_mu", "hall_n_cm3":"hall_n", "hall_density_cm3":"hall_n",
    "free_electron_cm3":"hall_n", "sigma_S_cm":"sigma", "Eg_optical_eV":"Eg",
    "dEg_eV":"dEg", "V_O_cm3_quench":"V_O", "tau_score":"tau",
    "dark_activation_eV":"dark", "PDR_score":"PDR",
}
CERT = {key:{"grade":full_forward.FIDELITY[grade],"status":"unvalidated"}
        for key,grade in PROPERTY_GRADES.items()}


def predict(dopant, conc_frac, T_anneal=TF_GLOBAL, pO2=1e-5, film=False,
            mode="A", lab_anchor_offset_mu=0.0, db=None):
    """Scenario calculation with no invented accuracy bars.

    Mode B applies a caller-supplied additive log10 mobility offset. Conductivity
    receives the same multiplier so sigma=e*n_H*mu_H is preserved. Accuracy must
    be tested on observations excluded from fitting the offset.
    """
    if mode not in {"A","B"}: raise ValueError("mode must be A or B")
    if not math.isfinite(float(lab_anchor_offset_mu)):
        raise ValueError("Mobility offset must be finite and expressed in log10 units")
    if mode=="A" and lab_anchor_offset_mu!=0:
        raise ValueError("Use mode B to apply an anchor-derived mobility offset")
    db=db or kroger_db.load()
    if dopant not in set(db.elements)-{"Ga","O"} and dopant!="undoped":
        raise ValueError("Dopant has no states in the supplied defect database")
    fwd=full_forward.forward(dopant,conc_frac,T_anneal=T_anneal,pO2=pO2,film=film,db=db)
    factor=10.**lab_anchor_offset_mu if mode=="B" else 1.
    properties={}
    for key,grade in PROPERTY_GRADES.items():
        adjusted=mode=="B" and key in {"hall_mu_cm2Vs","sigma_S_cm"}
        properties[key]={"value":fwd[key]*factor if adjusted else fwd[key],
            "grade":full_forward.FIDELITY[grade],"validation_status":"unvalidated",
            "bar_factor":None,"range":None,"empirical_anchor_adjustment":adjusted}
    return {"input":{"dopant":dopant,"conc_frac":conc_frac,"T_freeze":T_anneal,
                    "pO2":pO2,"film":film,"mode":mode,"mobility_offset_log10":lab_anchor_offset_mu},
            "properties":properties,"measurement_temperature_K":300.,
            "carrier_definition":fwd['carrier_definition'],
            "comparison_flags":{k:fwd[k] for k in ['compensation_abstain','site_blocking_limited','above_reported_incorporation_ceiling']},
            "headline":"Scenario calculation; historical accuracy bars are withdrawn. "
                       "No validated new-dopant ranking or device prediction is implied.",
            "evidence_manifest":"paper/npj/revision_20261015/issue_register.json"}


if __name__=="__main__":
    import json
    print(json.dumps(predict('Si',1e18/full_forward.N_CAT_SITE,film=False),indent=2))
