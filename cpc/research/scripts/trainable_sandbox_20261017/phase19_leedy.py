"""Retrospective Leedy2018 source audit and frozen-model transfer.

Four graph readings plus one prose pair, with conflicting prose retained.
No new-source label enters parameter training or model/penalty selection.
"""
from pathlib import Path
import sys,json,argparse,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
from sandbox.property_predictor_v5 import DopedPropertyPredictor
from sandbox.kroger_db import load
from sandbox import transport
import phase2_energy as energy
from transport_chain import median

BASE=energy.BASE;OUT=BASE/'phase19_delivery';SOURCE=OUT/'leedy2018';BUNDLE=OUT/'predictor_bundle_v5'
DOI='10.1063/1.5047214';PRECURSOR='10.1063/1.4991363'


def freeze():
    p=SOURCE/'protocol.json'
    payload=dict(created_utc=energy.now(),version=1,DOI=DOI,
        source_type='original Leedy2018 experimental PLD series;not literature comparison points in Fig8',
        source_pdf_sha256=energy.sha(OUT/'literature/Leedy2018_APL_Materials_101102.pdf'),
        figure_sha256=energy.sha(OUT/'literature/Leedy2018_Fig8_original.png'),
        models_manifest_sha256=energy.sha(BUNDLE/'manifest.json'),code_sha256=energy.sha(Path(__file__)),
        reading='manual marker centers in native992x667 image;axis ticks fit in log coordinates;fixed3pixel reading bounds,not confidence intervals',
        axis_ticks=dict(x_px=[106.5,244.,382.,519.5,657.5,795.],x_log10=[16.,17.,18.,19.,20.,21.],
                        y_px=[566.,192.],y_log10=[1.,2.]),
        pairing='nominal target identifies specimen;actual chemical input is printed SIMS film concentration,never targetwt conversion or Hall-inferred dose',
        primary_labels='four current-work Fig8 points labelled0.01/.025/.05/.1wt%;1wt nH/mu from printed prose;1wt not invented as extra Fig8 marker',
        conflicts='printed0.025wt prose3.25e19/20 versus Fig8 retained as separate label scenario;activation percentages are another rounded consistency check',
        chemical_nonuniformity='0.05wt film SIMS3.08e20 plateau rises to5e20 nearinterface;retain all5 butflag scalar-doseproxy',
        pressure='reported5%O2 in1.3Pa mix ->.065Pa;fitted models use1e-5atm scenario,not measuredpressure;no new-pressure training',
        growth='550C series;590C substrate cleaning precedes film;470C1minN2 contactRTA is postfilm;actual fullcoolinghistory missing',
        anchors=dict(budget0=[],budget1=['0.025'],budget2=['0.025','0.1']),
        queries='all5 and predefineduniform4 excludingnonuniform0.05;alsofixedcommonquerysets;anchor penalties frozen in old-source training',
        transport='report deliveredhead and separate old49pair/10sourceCThead excludingLeedy2017;no targetmu anchors',
        covariance='SIMS andHall measurement covariance unavailable;pixel/rounding ranges are sensitivity,not sampled measurement uncertainty',
        independence='newDOI relative29carriercohort;2017precursoralreadyinconditionalmu52,excludeinextrahead;sampleoverlapnotfullyaudited,no independentlab/blindclaim',
        retrospective=True,new_DFT_MD=0,new_material_measurements=0,goal_complete=False)
    if p.exists():
        old=json.loads(p.read_text())
        for key in ('models_manifest_sha256','code_sha256','source_pdf_sha256','figure_sha256'):assert old[key]==payload[key],key
    else:energy.write(p,payload)


def extract():
    freeze();protocol=json.loads((SOURCE/'protocol.json').read_text());a=protocol['axis_ticks']
    bx=np.polyfit(a['x_px'],a['x_log10'],1);by=np.polyfit(a['y_px'],a['y_log10'],1)
    # Centers read from the inspected original embedded raster. The ranges
    # include center-reading uncertainty only, with an unchanged 3px margin.
    coords={'0.1':(676.6,393.),'0.05':(678.7,382.1),'0.025':(594.1,417.4),'0.01':(590.5,457.7)}
    C={'1':4.15e20,'0.1':3.08e20,'0.05':3.08e20,'0.025':9.25e19,'0.01':9.78e19}
    activation={'1':.42,'0.1':.44,'0.05':.46,'0.025':.37,'0.01':.33}
    rows=[]
    for label in ('0.01','0.025','0.05','0.1','1'):
        if label in coords:
            x,y=coords[label];n=10**np.polyval(bx,x);mu=10**np.polyval(by,y)
            nl,nh=sorted(10**np.polyval(bx,np.array([x-3,x+3])));ml,mh=sorted(10**np.polyval(by,np.array([y-3,y+3])))
            loc='articlep8 Fig8 current-work marker;chemicalp7SIMS'
        else:
            x=y=None;n=1.75e20;mu=27.;nl=1.745e20;nh=1.755e20;ml=26.5;mh=27.5
            loc='articlep3/abstract printed1wt carrier/mobility;chemicalp7SIMS'
        rows.append(dict(sample_id=DOI+'::target-'+label,sample_label=label,element='Si',incorporated_cm3=C[label],
            observed_nH_cm3=n,observed_muH_cm2Vs=mu,nH_reading_low=nl,nH_reading_high=nh,
            mu_reading_low=ml,mu_reading_high=mh,reading_x_px=x,reading_y_px=y,
            activation_proxy=activation[label],activation_derived_nH_cm3=C[label]*activation[label],
            growth_T_C=550.,observation_T_K=300.,postfilm_anneal_C=470.,postfilm_anneal_seconds=60.,
            measured_pO2_atm=.05*1.3/101325.,H_known=False,H_cm3=0.,
            uniform_scalar_input=label!='0.05',chemical_profile_upper_cm3=5e20 if label=='0.05' else C[label],
            source_location=loc,raw_instrument_data=False,not_blind=True))
    energy.csvwrite(SOURCE/'samples.csv',rows)
    energy.write(SOURCE/'digitization.json',dict(x_log_calibration=bx,y_log_calibration=by,
        coordinate_frame='original embedded992x667 raster,pixeloriginupperleft',fixed_reading_margin_px=3.,
        unresolved_prose_conflict=dict(sample='0.025',nH_cm3=3.25e19,muH_cm2Vs=20.),
        SIMS_precision='reportednominal3significantfigures,not measurementCI',
        not_Hall_reverse_dose=True,independent_specimens=5,independent_lab_verified=False))
    print(pd.DataFrame(rows)[['sample_label','incorporated_cm3','observed_nH_cm3','observed_muH_cm2Vs']],flush=True)


def evaluate():
    freeze();d=read(SOURCE/'samples.csv');model=DopedPropertyPredictor(load(),BUNDLE)
    old=read(ROOT/'results/chain_diagnosis_20261015/executed/mobility_bounds_and_inputs.csv')
    old=old[~old.family.isin((PRECURSOR,DOI))]
    weights=1/old.family.map(old.family.value_counts()).to_numpy(float)
    ct=np.array([np.log10(transport.mu_caughey_thomas(r.carrier_cm3,300.)) for r in old.itertuples()])
    offset=float(median(np.log10(old.mu_cm2Vs.to_numpy(float))-ct,weights))
    energy.write(SOURCE/'precursor_excluded_transport.json',dict(offset_dex=offset,records=len(old),
        families=sorted(old.family.unique()),excluded=[PRECURSOR,DOI],target_mobility_labels_used=False))
    rows=[];anchors={0:[],1:['0.025'],2:['0.025','0.1']}
    # Fixed alternatives expose the source's inconsistent prose and the
    # nonuniform chemical profile; they do not select the preferred model.
    scenarios=('primary_graph','prose_0025','nonuniform_upper','reading_lower','reading_upper')
    for scenario in scenarios:
        table=d.copy()
        if scenario=='prose_0025':
            table.loc[table.sample_label.eq('0.025'),['observed_nH_cm3','observed_muH_cm2Vs']]=[3.25e19,20.]
        if scenario=='nonuniform_upper':table.loc[table.sample_label.eq('0.05'),'incorporated_cm3']=5e20
        if scenario in ('reading_lower','reading_upper'):
            side='low' if scenario.endswith('lower') else 'high'
            table['observed_nH_cm3']=table['nH_reading_'+side];table['observed_muH_cm2Vs']=table['mu_reading_'+side]
        for method in model.manifest['methods']:
            for budget in (0,1,2):
                a=[dict(incorporated_cm3=float(table[table.sample_label.eq(label)].incorporated_cm3.iloc[0]),
                        measured_nH_cm3=float(table[table.sample_label.eq(label)].observed_nH_cm3.iloc[0])) for label in anchors[budget]]
                for r in table.itertuples():
                    entry=dict(scenario=scenario,method=method,budget=budget,sample_id=r.sample_id,sample_label=r.sample_label,
                        role='anchor' if r.sample_label in anchors[budget] else 'query',uniform_scalar_input=r.uniform_scalar_input,
                        common_query=r.sample_label not in anchors[2],incorporated_cm3=r.incorporated_cm3,
                        observed_log_nH=np.log10(r.observed_nH_cm3),observed_log_mu=np.log10(r.observed_muH_cm2Vs),
                        observed_log_sigma=np.log10(transport.E*r.observed_nH_cm3*r.observed_muH_cm2Vs))
                    try:
                        p=model.predict('Si',float(r.incorporated_cm3),method=method,anchors=a,growth_T_C=550.)
                        # EmpiricalCT comparison excludes the related precursor.
                        pm=float(np.log10(transport.mu_caughey_thomas(p['nH_cm3'],300.))+offset)
                        entry.update(numeric=True,predicted_log_nH=p['log10_nH_cm3'],predicted_log_mu=p['log10_muH_cm2Vs'],
                            predicted_log_sigma=p['log10_sigma_S_cm'],precursor_excluded_log_mu=pm,
                            precursor_excluded_log_sigma=np.log10(transport.E)+p['log10_nH_cm3']+pm,
                            source_coordinate=p.get('source_coordinate'),outside_training_dose=p['outside_training_chemical_range']
                                if 'outside_training_chemical_range' in p else not p['chemical_training_range_cm3'][0]<=r.incorporated_cm3<=p['chemical_training_range_cm3'][1])
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:entry.update(numeric=False,error=repr(exc))
                    rows.append(entry)
    result=pd.DataFrame(rows);result.to_csv(SOURCE/'predictions.csv',index=False)
    summary=[]
    for (scenario,m,b),g in result[result.role.eq('query')].groupby(['scenario','method','budget']):
        for scope in ('all5','uniform4','common3','common_uniform2'):
            q=g.copy()
            if scope in ('uniform4','common_uniform2'):q=q[q.uniform_scalar_input.eq(True)]
            if scope in ('common3','common_uniform2'):q=q[q.common_query.eq(True)]
            entry=dict(scenario=scenario,method=m,budget=int(b),scope=scope,queries=len(q),numeric=int(q.numeric.sum()))
            for prop in ('nH','mu','sigma'):
                p='predicted_log_'+prop; y='observed_log_'+prop
                entry[prop+'_MAE']=float(np.abs(q[p]-q[y]).mean()) if q.numeric.all() else None
            entry['precursor_excluded_mu_MAE']=float(np.abs(q.precursor_excluded_log_mu-q.observed_log_mu).mean()) if q.numeric.all() else None
            summary.append(entry)
    summary=pd.DataFrame(summary);summary.to_csv(SOURCE/'summary.csv',index=False)
    primary=summary[summary.scenario.eq('primary_graph')&summary.scope.eq('all5')]
    print(primary[['method','budget','queries','numeric','nH_MAE','mu_MAE']].to_string(index=False),flush=True)
    energy.write(SOURCE/'audit.json',dict(new_study_specimens=5,source_DOI=DOI,source_in_29carrier_training=False,
        precursor_in_old52conditional_transport=True,precursor_excluded_extra_head_pairs=len(old),
        primary_input_nonuniform_specimens=1,above_old_Si_dose_range_specimens=3,
        label_scenarios=len(scenarios),rows=len(result),numeric_rows=int(result.numeric.sum()),
        frozen_hyperparameters=True,new_source_parameter_training=False,not_blind=True,
        pressure_mismatch_explicit=True,independent_lab_claim=False,independent_sigma_labels=0,goal_complete=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initialize',action='store_true');ap.add_argument('--extract',action='store_true');ap.add_argument('--evaluate',action='store_true');a=ap.parse_args()
    freeze()
    if a.extract:extract()
    if a.evaluate:evaluate()
