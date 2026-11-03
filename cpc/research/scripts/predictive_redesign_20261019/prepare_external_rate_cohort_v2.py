"""Explicit retrospective external conditions, kept outside frozen training."""
from pathlib import Path
from itertools import product
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

OUT=c.BASE/'external_rate_cohort_v2'
GROUP='related_OSU_Zhao_LPCVD_MOCVD_2018_2019'


def main():
    assert not (OUT/'protocol_before_transcription.json').exists()
    sources=[c.BASE/'additional_sources_v3/rafique2018.pdf',
             c.BASE/'additional_sources_v4/feng2019.pdf']
    code=Path(__file__)
    c.write(OUT/'protocol_before_transcription.json',dict(created_utc=c.e.now(),
        code_sha256=c.e.sha(code),source_hashes=c.file_hashes(sources),
        target='additional conditional external prediction from existing full-five-group presets only',
        no_new_model_or_hyperparameter_training_on_these_labels=True,
        primary_prespecified='ChargeRatioSpectrum__chain_safe;all parents and rate controls retained',
        primary_external='zero anchor,all eligible three conditions;Feng explicit caption andRafique context-linked pair grades separate',
        group_count=1,papers=2,physical_conditions=3,
        pairing_grade_and_missing_inputs_reported=True,
        prior_snippet_and_fulltext_labels_exposed=True,not_blind=True,
        baseline_models_may_have_prior_corpus_exposure=True,
        source_roles='sort actual chemical dose;first1/2conditions anchor where available;insufficient2anchor query scope left empty',
        metric='equal scenarios percondition then equalDOI;also worst scenario and eachcondition;not a calibrated probability',
        H_flag='backend H_known means explicitly provided positive hypothetical H for sensitivity;H_measured_exact is False for every case',
        censored_H_not_an_exact_label=True,carbon_not_modelled_scope_preserved=True,
        old35conditions_and49scenarios_remain_unchanged=True,
        prediction_protocol_hashes=c.file_hashes([
            c.BASE/'phase31_rate_redesign/prediction/design_before_prediction.json',
            c.BASE/'phase32_rate_controls/prediction/design_before_prediction.json']),
        training_protocol_hashes=c.file_hashes([
            c.BASE/'phase31_rate_redesign/training_protocol.json',
            c.BASE/'phase32_rate_controls/training_protocol.json']),goal_complete=False))
    records=[
        dict(sample_id='10.1063/1.5017616::001',family='10.1063/1.5017616',
            element='Si',incorporated_cm3=9.6e17,nH_cm3=9.5e17,mu_cm2Vs=42.,
            orientation='001',thickness_um=1.2,growth_T_C=900.,
            pO2_atm=.048*4./760.,growth_T_status='reported',oxygen_status='idealmixture partial pressure from4Torr and4.8percentO2',
            pairing_grade='orientation/context-linked Fig5+Fig6,not explicit physical specimen ID',
            read_location='PDFpage3 Methods;PDFpage5 Fig5/6 andtext',
            H_status='visible SIMSprofile,no unique reported bulk-average H',
            C_status='visible SIMSprofile;C omitted by current Si/Sn/H physics;not carbon-free',
            source_note='retain reported chemical/Hall values;paper phrase negligiblecompensation does not change their ratio'),
        dict(sample_id='10.1063/1.5017616::010',family='10.1063/1.5017616',
            element='Si',incorporated_cm3=2.4e18,nH_cm3=1.2e18,mu_cm2Vs=72.,
            orientation='010',thickness_um=1.9,growth_T_C=900.,
            pO2_atm=.048*4./760.,growth_T_status='reported',oxygen_status='idealmixture partial pressure from4Torr and4.8percentO2',
            pairing_grade='orientation/context-linked Fig5+Fig6,not explicit physical specimen ID',
            read_location='PDFpage3 Methods;PDFpage5 Fig5/6 andtext',
            H_status='visible SIMSprofile,no unique reported bulk-average H',
            C_status='visible SIMSprofile;C omitted by current Si/Sn/H physics;not carbon-free',
            source_note='chemical/Hall ratio exactly2 asreported;not repaired'),
        dict(sample_id='10.1063/1.5109678::Fig2',family='10.1063/1.5109678',
            element='Si',incorporated_cm3=3.e17,nH_cm3=1.01e17,mu_cm2Vs=152.,
            orientation='010',thickness_um=1.25,growth_T_C=None,pO2_atm=None,
            growth_T_status='samepaper800-880C range;Fig2 specimen T not uniquely assigned',
            oxygen_status='Fig2 specimen pO2 missing;VI/III ratio is not gas partial pressure',
            pairing_grade='explicit same specimen inFig2caption',
            read_location='authorPDFpage6 chemical Si~3e17;page14 caption;page17 Fig2',
            H_status='below2e16 detectionlimit;not measured zero',
            C_status='below2e16 detectionlimit;C omitted by physics;not measured zero',
            source_note='TableI doping meansHall density andTableIII donor populations arefitted;neither supplies additional chemical doses')]
    old=c.cases()
    old_oxygen=sorted(float(v) for v in old.loc[old.family.eq(c.data.GOGOVA),'scenario_pO2_atm'].unique())
    cases=[];roles=[]
    for record in records:
        record.update(related_cluster=GROUP,Tobs_K=300.,temperature_scope='room temperature model convention',
                      chemical_dose_status='reported approximate bulk SIMS',source_verification='primaryPDFtext+visualfigure/caption',
                      all_dopant_measurements_retained=True)
        if record['family']=='10.1063/1.5109678':
            temps=[800.,840.,880.];ox=old_oxygen;hydrogen=[0.,1e16,2e16]
            hscope='censored H sensitivity at0,half limit,upperlimit;not posterior or exact measuredH'
        else:
            temps=[900.];ox=[record['pO2_atm']];hydrogen=[0.,1e17,1e18]
            hscope='hypothetical H sensitivity,not a measured confidenceinterval;unknown bulk H remains'
        scenarios=list(product(temps,ox,hydrogen))
        for i,(T,O,H) in enumerate(scenarios):
            cases.append(dict(record,case_id=record['sample_id']+f'::scenario{i}',
                growth_T_C=T,scenario_pO2_atm=O,oxygen_input_kind='conditional_external_scenario' if len(ox)>1 else 'reported molecular O2 nominal ideal-mixture partial pressure',
                H_cm3=H,H_known=H>0,H_measured_exact=False,H_scenario_scope=hscope,
                carbon_cm3=0.,carbon_known=False,carbon_zero_is_only_omitted_input_scenario=True,
                scenario_index=i,scenario_count=len(scenarios),scenario_weight=1./len(scenarios),
                target=float(np.log10(record['nH_cm3'])),mobility_eligible=True,
                primary_common20=False,new_Sn=False))
    for family in sorted({r['family'] for r in records}):
        selected=sorted([r for r in records if r['family']==family],key=lambda r:r['incorporated_cm3'])
        for budget in range(min(2,len(selected))+1):
            anchors={r['sample_id'] for r in selected[:budget]}
            for r in selected:roles.append(dict(family=family,sample_id=r['sample_id'],budget=budget,
                role='anchor' if r['sample_id'] in anchors else 'query'))
    assert len(records)==3 and len(cases)==24
    assert not set(r['family'] for r in records)&set(old.family)
    for r in records:
        q=[s for s in cases if s['sample_id']==r['sample_id']]
        assert abs(sum(s['scenario_weight'] for s in q)-1)<1e-12
    c.e.csvwrite(OUT/'conditions3.csv',records)
    c.e.csvwrite(OUT/'scenarios24.csv',cases)
    c.e.csvwrite(OUT/'episodes.csv',roles)
    c.write(OUT/'qualification.json',dict(created_utc=c.e.now(),physical_conditions=3,DOIs=2,
        conservative_related_groups=1,scenario_rows=24,
        explicit_caption_pair_conditions=1,context_linked_orientation_pair_conditions=2,
        full_exact_composition_conditions=0,
        conditional_external_only=True,not_blind=True,no_current_training_label_overlap_byDOI=True,
        figures_actually_viewed=['RafiquePDFpage5','FengPDFpage14','FengPDFpage17'],
        missing_or_censored_inputs_kept_as_scenarios=True,
        no_tableI_fitted_donor_or_flow_substitution=True,goal_complete=False,
        evidence_sha256=c.file_hashes([OUT/'conditions3.csv',OUT/'scenarios24.csv',OUT/'episodes.csv'])))
    print('Prepared separate external cohort:3conditions/2papers/1group/24conditionalscenarios;training unchanged',flush=True)


if __name__=='__main__':main()
