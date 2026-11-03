"""Version35 paired observations and normalized missing-process scenarios.

Original33 rows/roles are never rewritten. Condition and scenario counts stay
separate; unknown per-sample inputs remain missing in the condition table.
"""
from pathlib import Path
from itertools import product
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE=e.BASE;OUT=BASE/'phase29_joint_scattering'
OLD=BASE/'phase25_shared_process/data'
SOURCE=BASE/'phase28_paired_transport'
GOGOVA='10.1039/c5ce01106j'
IKZ='10.1007/s10853-015-9693-6'
TEMPERATURES=(775.,800.,825.,850.)
OXYGEN=(1e-5,1e-2)


def load_conditions():return read(OUT/'data/conditions35.csv')
def load_cases():return read(OUT/'data/scenarios49.csv')
def load_roles():return read(OUT/'data/episodes35.csv')


def main():
    target=OUT/'data/qualification.json'
    if target.exists():
        d=json.loads(target.read_text())
        for kind in ('input_hashes','code_hashes','output_hashes'):
            for name,digest in d[kind].items():assert e.sha(ROOT/name)==digest,name
        print('Verified35condition/49scenario data version;no writes',flush=True)
        return
    assert e.sha(BASE/'artifact_manifest_phase28.json')=='8be1a180910f7fc42f020bc599af23beddfe76eedc1483acd607e5b3dca625b3'
    assert json.loads((SOURCE/'source_qualification_v1/audit.json').read_text())['passed']
    inputs=[OLD/'conditions33.csv',OLD/'episodes33.csv',SOURCE/'source_observations_v1.json',
        SOURCE/'source_qualification_v1/summary.json',SOURCE/'source_qualification_v1/audit.json',
        BASE/'artifact_manifest_phase28.json']
    code={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))}
    protocol=dict(created_utc=e.now(),purpose='35reported conditions and11paired labels;49normalized input scenarios',
        original_rows_and_roles_unchanged=True,retrospective=True,new_material_measurements=0,
        scenario_temperature_C=TEMPERATURES,scenario_oxygen_activity=OXYGEN,
        scenario_weight='1/8pernewcondition,1peroldcondition;not measured input distribution',
        IKZ_association='newGogova/oldBaldini same exclusion group;no unique wafer IDs claimed',
        new_roles='B(lower chemical dose) firstHall anchor,A second;bothquery atbudget0',
        primary_query='originalcommon20unchanged;newSn supplement separately',
        inputs={str(p.relative_to(ROOT)):e.sha(p) for p in inputs},code_hashes=code,goal_complete=False)
    before=OUT/'data/protocol_before_construction.json'
    assert not before.exists(),'preserve prior construction attempt before a versioned correction'
    e.write(before,protocol)
    old=read(OLD/'conditions33.csv');roles=read(OLD/'episodes33.csv')
    assert len(old)==33 and len(roles)==99
    source=json.loads((SOURCE/'source_observations_v1.json').read_text())
    candidates={x['sample']:x for x in source['conditions'] if x['source_id']=='Gogova2015'}
    assert set(candidates)=={'A','B','C'}
    added=[]
    for label in ('A','B'):
        r=candidates[label];v=r['reported'];sid=GOGOVA+'::'+label
        added.append(dict(sample_id=sid,family=GOGOVA,related_cluster=IKZ,sample_label=label,element='Sn',
            incorporated_cm3=float(v['Sn_cm3']),carrier_cm3=float(v['nH_cm3']),mu_cm2Vs=float(v['muH_cm2Vs']),
            target=float(np.log10(float(v['nH_cm3']))),observed_log_nH=float(np.log10(float(v['nH_cm3']))),
            growth_T_C=np.nan,scenario_pO2_atm=np.nan,pO2_atm=np.nan,Tobs_K=300.,measurement_T_K=300.,
            oxygen_input_kind='unknown individual H2O redox;declared scenario set,not measured O2',
            H_known=False,H_cm3=0.,carbon_known=False,carbon_cm3=0.,carbon_present_qualitative=True,
            mobility_eligible=True,kind='RT',endpoint='nH',obs_id='RT::'+sid,
            method='MOVPE',substrate='(100)Mg-doped SI beta-Ga2O3',
            concentration_kind='Table1 labelled SIMS concentration;individual direct/calibrated path unreported',
            incorporated_reading_status='central printed value;quantitative error covariance not given',
            observation_T_status='RT explicit;300K comparison convention,exactT absent',
            condition_pairing_verified=True,independent_physical_ID_verified=False,not_blind=True,
            included_primary=False,new_phase29_condition=True,source_record_id=r['condition_id'],
            source_locator='author-uploaded primary9pagePDF,p8Table1;methods p2,electrical p7',
            source_file='public authorPDF via web;localdownload403 retained;see source_observations_v1.json',
            scope='conditional input scenarios;unquantified C-H and thermal/pressure history;relatedIKZ',
            full_process_input_measured=False))
    extra=pd.DataFrame(added)
    data=pd.concat([old,extra],ignore_index=True)
    data['new_phase29_condition']=data.family.eq(GOGOVA)
    data['source_record_id']=data.source_record_id.fillna('original33')
    data['phase29_training_eligible']=True
    data['primary_carrier_benchmark_condition']=data.sample_id.isin(old.sample_id)
    assert len(data)==35 and data.sample_id.is_unique
    assert data.family.nunique()==7 and data.related_cluster.nunique()==5
    pd.testing.assert_frame_equal(data.iloc[:33][old.columns].reset_index(drop=True),old,check_dtype=False)
    assert int(data.mobility_eligible.sum())==11
    new_roles=[]
    for b in (0,1,2):
        for label in ('A','B'):
            anchor=b==2 or (b==1 and label=='B')
            new_roles.append(dict(sample_id=GOGOVA+'::'+label,family=GOGOVA,budget=b,role='anchor' if anchor else 'query'))
    episodes=pd.concat([roles,pd.DataFrame(new_roles)],ignore_index=True)
    pd.testing.assert_frame_equal(episodes.iloc[:99][roles.columns].reset_index(drop=True),roles,check_dtype=False)
    assert len(episodes)==105 and not episodes.duplicated(['sample_id','budget']).any()
    for b,count in ((0,35),(1,27),(2,20)):
        assert len(episodes[episodes.budget.eq(b)&episodes.role.eq('query')])==count
        assert len(episodes[episodes.budget.eq(b)&episodes.role.eq('anchor')])==7*b
    common=set.intersection(*[set(episodes.loc[episodes.budget.eq(b)&episodes.role.eq('query'),'sample_id']) for b in (0,1,2)])
    old_common=set.intersection(*[set(roles.loc[roles.budget.eq(b)&roles.role.eq('query'),'sample_id']) for b in (0,1,2)])
    assert common==old_common and len(common)==20
    rows=[]
    for r in data.to_dict('records'):
        settings=list(product(TEMPERATURES,OXYGEN)) if r['family']==GOGOVA else [(r['growth_T_C'],r['scenario_pO2_atm'])]
        for j,(temperature,oxygen) in enumerate(settings):
            x=dict(r);x.update(case_id=r['sample_id']+'::scenario'+str(j),scenario_index=j,
                scenario_count=len(settings),scenario_weight=1./len(settings),growth_T_C=float(temperature),
                scenario_pO2_atm=float(oxygen),physical_condition_id=r['sample_id'],primary_common20=r['sample_id'] in common)
            if r['family']==GOGOVA:
                x['oxygen_input_kind']='diagnostic effective oxygen activity;H2Osource,not measured O2ortrueprobability'
            rows.append(x)
    cases=pd.DataFrame(rows)
    assert len(cases)==49 and cases.case_id.is_unique and cases.sample_id.nunique()==35
    assert np.max(abs(cases.groupby('sample_id').scenario_weight.sum()-1))<1e-14
    assert cases[['incorporated_cm3','growth_T_C','Tobs_K','scenario_pO2_atm']].notna().all().all()
    assert len(cases[cases.mobility_eligible.eq(True)])==25
    e.csvwrite(OUT/'data/conditions35.csv',data.to_dict('records'))
    e.csvwrite(OUT/'data/scenarios49.csv',cases.to_dict('records'))
    e.csvwrite(OUT/'data/episodes35.csv',episodes.to_dict('records'))
    roundtrip=load_conditions()
    pd.testing.assert_frame_equal(roundtrip.iloc[:33][old.columns].reset_index(drop=True),old,check_dtype=False)
    assert roundtrip[roundtrip.family.eq(GOGOVA)].growth_T_C.isna().all()
    paths=[OUT/'data/conditions35.csv',OUT/'data/scenarios49.csv',OUT/'data/episodes35.csv']
    result=dict(passed=True,created_utc=e.now(),reported_conditions=35,scenario_rows=49,
        paired_reported_conditions=11,paired_scenario_rows=25,DOIs=7,related_groups=5,
        original33_values_and99roles_preserved=True,common20_preserved=sorted(common),
        all_query_counts={'0':35,'1':27,'2':20},Hall_anchor_counts={'0':0,'1':7,'2':14},
        primary_data_is_reported_conditions_not_verified_unique_wafers=True,
        new_conditions_have_incomplete_physical_inputs=True,scenario_weights_not_measured_probabilities=True,
        no_independent_sigma_labels=True,Je_not_admitted_to_single_dopant_input=True,
        original_nonmeasurable_C_retained_in_phase28_not_a_zero_label=True,
        no_model_fits_or_property_predictions=True,
        input_hashes=protocol['inputs'],code_hashes=code,
        output_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths},goal_complete=False)
    e.write(target,result)
    print('Built35conditions/49scenarios/11pairedconditions;oldcommon20 unchanged',flush=True)


if __name__=='__main__':main()
