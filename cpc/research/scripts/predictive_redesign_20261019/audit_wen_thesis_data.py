"""Transcribe actual Hall table; retain missing chemical and nitrogen inputs."""
from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

SOURCE=c.BASE/'additional_sources_v2/repository_followup_v4'
OUT=c.BASE/'wen_thesis_data_audit_v1'


def main():
    assert not (OUT/'protocol_before_transcription.json').exists()
    old=c.ROOT/'results/next_study_20261015/target_correction_audit.json'
    c.write(OUT/'protocol_before_transcription.json',dict(created_utc=c.e.now(),
        code_sha256=c.e.sha(__file__),source_sha256=c.e.sha(SOURCE/'wen_thesis2024.pdf'),
        legacy_exposure_file_sha256=c.e.sha(old),thesis_cover_author='Zhuoqun Wen',thesis_cover_year=2024,
        source_scope='chapter2 related toAPL2023 DOI5.0142107;exact table source is thesisTable2',
        table_location='printedpage43/PDFpage63;original page actually viewed',
        SIMS_figure_location='printedpage37/PDFpage57;original page actually viewed',
        no_current_training_changes=True,no_fabricated_SIMS_perHall_sample=True,
        no_N_concentration_to_active_acceptor_truth_substitution=True,
        highresistive_or_missing_Hall_not_zero=True,goal_complete=False))
    flows=[.012,.029,.059,.147,.294,.588]
    data={525.:[(3.4e17,135.),(9.6e17,88.),(1.7e18,83.),(4.3e18,45.),(8.1e18,30.),(1.8e19,20.)],
          550.:[(2.1e17,100.),(5.8e17,65.),(1.2e18,62.),(2.5e18,60.),(7.7e18,55.),(1.4e19,34.)],
          600.:[(2.e17,20.),(4.5e17,33.),(1.1e18,79.),(4.5e18,38.),(7.e18,48.),(1.1e19,66.)]}
    rows=[]
    for T,values in data.items():
        for Q,(n,mu) in zip(flows,values):
            rows.append(dict(sample_id=f'WenThesis2024::Ch2::T{T:g}::Q{Q:g}',
                source_family='WenThesis2024_Ch2_related_APL2023_5.0142107',
                element='Si',growth_T_C=T,disilane_flow_sccm=Q,disilane_dilution_wtpercent=10.,
                Ga_BEP_Torr=1e-8,oxygen_flow_sccm=2.,RF_power_W=410.,growth_rate_nm_h=110.,
                active_layer_thickness_nm=440.,UID_buffer_thickness_nm=220.,
                nH_cm3=n,mu_cm2Vs=mu,derived_sigma_S_cm=c.learning.E_CHARGE*n*mu,
                derived_sigma_not_third_independent_label=True,Tobs_K=300.,Tobs_scope='roomtemperature model convention',
                incorporated_cm3=None,chemical_input_status='separate SIMS calibration structures;no exact individual dose assigned',
                H_cm3=None,H_status='SIMS stack H~1e17;not exact perHall sample',
                batch_N_total_cm3_approx=1e17,N_exact_perHall_sample=False,N_active_acceptor_fraction_known=False,
                substrate='Fe-doped betaGa2O3(010),UID buffer separates layer',
                source_page='thesisprinted43/PDF63 Table2',dose_property_main_eligible=False,
                conditional_Hall_table_usable=True,current_model_N_input_missing=True,
                observation_status='reported quantitative Hall condition',
                independence_from_old_groups='pending sample/equipment lineage;not a blind newsource claim'))
    chemical=[dict(observation_id='WenFig13_T'+str(int(T)),temperature_C=T,flow_sccm=.588,
                   chemical_Si_cm3_approx=2.3e19 if T==525 else 1.8e19,
                   active_layer_nm=220.,measurement='SIMS,independent uniform calibration structure',
                   source='thesisprinted37-38 Fig13 andtext;not each Hallfilm') for T in [525.,600.,650.,700.]]
    missing=[dict(temperature_C=T,flow_sccm=.029,
                  status='no Hall result reported' if T==500 else 'reported highlyresistive/difficultohmiccontact',
                  nH_cm3=None,mu_cm2Vs=None,no_numeric_detection_threshold_given=True)
             for T in [500.,650.,700.]]
    assert len(rows)==18 and len({r['sample_id'] for r in rows})==18
    assert all(r['incorporated_cm3'] is None for r in rows)
    c.e.csvwrite(OUT/'Hall_conditions18.csv',rows)
    c.e.csvwrite(OUT/'chemical_calibration_points4.csv',chemical)
    c.e.csvwrite(OUT/'unmeasured_or_censored_conditions.csv',missing)
    legacy_count=old.read_text().count('10.1063/5.0142107')
    c.write(OUT/'qualification.json',dict(created_utc=c.e.now(),reported_Hall_conditions=18,
        per_Hall_independently_measured_chemical_doses_assigned=0,
        current35_condition_main_admissions=0,current_external3_condition_admissions=0,
        legacy_corpus_already_contains_related_DOI=legacy_count>0,legacy_occurrences=legacy_count,
        not_blind=True,source_year_from_cover2024=True,
        PDF_internal_author_property_is_template_metadata_not_cover_author=True,
        minimum_needed=['explicit calibrated incorporation operator with separate SIMS stacks andtransfer uncertainty',
                        'nitrogen-compatible charge budget or explicit out-of-composition scope',
                        'growth/equipment/sample lineage audit before independent-group count'],
        SIMS_stack_minflow_body_sccm=.024,SIMS_stack_minflow_fig12_rectangle_sccm=.012,
        minflow_source_discrepancy_unresolved_not_corrected=True,
        highestflow_SIMS_stack_Si_approx=2e19,highestflow_uniform525C_Si_approx=2.3e19,
        separate_calibration_structures_not_merged_into_one_truth=True,
        source_donor_or_neutral_impurity_curvefits_are_not_chemical_truth=True,
        source_read_depth='cover,chapter2methods/results/table2/fig12;chapter3not yet completed',
        goal_complete=False,evidence_sha256=c.file_hashes([
            OUT/'Hall_conditions18.csv',OUT/'chemical_calibration_points4.csv',
            OUT/'unmeasured_or_censored_conditions.csv',SOURCE/'wen_thesis2024_page057.png',
            SOURCE/'wen_thesis2024_page063.png'])))
    print('Wen thesis:18actual Hall rows retained;0fabricated individualchemicaldoses;Nscope andpriorcorpus exposure recorded',flush=True)


if __name__=='__main__':main()
