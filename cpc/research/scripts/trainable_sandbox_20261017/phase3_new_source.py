"""Audit a held-study full-property check without pooling its labels into training.

The publication was already in the historical archive with erroneous rows.
These are newly re-audited same-sample measurements, not a blind new publication.
"""
from pathlib import Path
import sys,json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.BASE/"new_property_sources"
DOI="10.1063/5.0139622"
RELATED="10.1063/5.0101132"

def initialize():
    path=OUT/"azizie2023_validation_protocol.json"
    if path.exists():return
    pdf=OUT/"azizie2023_final.pdf"
    assert pdf.exists()
    data=read_result_csv(energy.BASE/"dataset_v1.csv")
    mobility=read_result_csv(ROOT/"results/chain_diagnosis_20261015/executed/mobility_bounds_and_inputs.csv")
    # All current numerical learners use audited cohorts, not old augmented rows.
    assert not data.family.astype(str).str.contains("5.0139622",regex=False).any()
    assert not mobility.family.astype(str).str.contains("5.0139622",regex=False).any()
    names=list("abcdefghijkl")
    mu=[124,119,111,129,104,86,98,91,75,79,62,68]
    n=[2.7e17,2.9e17,3.5e17,3.5e17,9.8e17,1.3e18,1.5e18,2.4e18,5.5e18,1.1e19,3e19,4.4e19]
    cell=[1286,1134,1115,1119,1187,1225,1171,1176,1262,1286,1319,1350]
    low_flux={"b","c","e","g"}
    rows=[]
    for name,m,nn,cs in zip(names,mu,n,cell):
        paired=name in ("e","k")
        # Primary concentration is directly the rounded value in p6 text.
        # Do not derive a chemical input from held-out carrier labels/activation.
        C={"e":2e18,"k":3e19}.get(name)
        bounds={"e":(1.5e18,2.5e18),"k":(2.5e19,3.5e19)}.get(name,(None,None))
        rows.append(dict(sample_id=DOI+"::"+name,family=DOI,sample_label=name,element="Si",
            measured_nH_cm3=nn,measured_muH_cm2Vs=m,measurement_T_K=300.,
            growth_T_C=525.,SiO2_cell_T_C=cs,nominal_thickness_nm=1000.,
            Ga2O_flux_molecules_cm2s=6e14 if name in low_flux else 1e15,
            reported_ozone_pressure_Torr=2.5e-6,reported_ozone_fraction=.8,
            incorporated_cm3=C,rounded_C_low=bounds[0],rounded_C_high=bounds[1],
            concentration_kind="same_sample_SIMS_approximate_prose" if paired else "not_independently_reported_for_this_sample",
            paired_dose_eligible=paired,table_locator="journal printedp5 TableII, PDFpage5(0-index)",
            concentration_locator="journal printedp6 paired-SIMS paragraph, approximate values; Fig2 consistency check" if paired else "",
            caveat="nominal/estimated thickness and rounded chemical concentration; growth-rate and reactive-ozone chemical-potential limitations"))
    energy.csvwrite(OUT/"azizie2023_audited_samples.csv",rows)
    proto=dict(version=1,created_utc=energy.now(),source_DOI=DOI,source_pdf_sha256=energy.sha(pdf),
        source_url="https://schlom.mse.cornell.edu/sites/default/files/research%20pdfs/5.0139622.pdf",
        source_status="existing historical archive source; corrected paired-label extraction, held from current29/52 cohorts, not blind",
        team_relationship="shared Cornell/Jena/AFRL authors with old5.0101132; exclude that entire source from carrier and transport fitting",
        related_excluded_DOI=RELATED,
        paired_labels=2,conditional_mobility_labels=12,
        primary_chemical_inputs="rounded p6 prose C_e=2e18,C_k=3e19cm-3; same-sample SIMS explicitly stated; no carrier-derived C",
        rounding_scenarios="e[1.5,2.0,2.5]e18 and k[2.5,3.0,3.5]e19, Cartesian; resolution/sensitivity scenarios, not measured confidence bounds",
        figure_role="Fig2 matched SIMS markers can cross-check the rounding; do not use arrowhead endpoints as data or tune C after predictions",
        flux_warning="right Fig2 axis assumes1um/h;0.86/1.3um/h rates describe SIMS stacks, not automatically the paired Hall-film rates",
        timing="freeze before evaluating any new-study model predictions",
        pressure="primary same unmeasured1e-5atm effective-O2 scenario as training; reported2.5e-6Torr distilled ozone is not an equilibrium O2 chemical potential",
        observed_properties="room-temperature nH and muH; sigma=q*nH*muH is derived, not a new measurement",
        tasks=dict(
            zero_anchor="both dose-paired samples are queries",
            one_anchor="lower-dose e is nH anchor, k query; prior shrinkage selected only from old training sources",
            two_anchor="zero queries; do not report zero error or claim validation",
            full_chain="chemical dose -> predicted nH -> training-only CT median mobility head -> derived sigma; no query measured nH/mu in the chain",
            conditional_transport="all12 measured nH -> mu, kept separate from full chain"),
        methods="phase3 all fixed/nested energy methods and full-energy reference; simple unit-dose/line controls; phase2 selected low-label paths as separate budget block",
        carrier_training_records=int(data.family.ne(RELATED).sum()),
        carrier_training_families=sorted(data[data.family.ne(RELATED)].family.unique()),
        transport_training_records=int(mobility.family.ne(RELATED).sum()),
        transport_training_families=sorted(mobility[mobility.family.ne(RELATED)].family.unique()),
        no_training_pool_augmentation=True,
        limitations=["2 approximate chemical-dose points","one related study, not independent lab","unknown reactive-oxygen reservoir","nominal thickness/per-sample metrology uncertainty"],
        code_sha256=energy.sha(Path(__file__)),
        dataset_sha256=energy.sha(OUT/"azizie2023_audited_samples.csv"))
    energy.write(path,proto)
    print(json.dumps(proto,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":initialize()
