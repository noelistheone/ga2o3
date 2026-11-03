"""Freeze transfer/diagnostic protocols before evaluating the known Leedy source.

Leedy already motivated this design: this is development evidence, not a new
blind source. No query labels enter any fit, source inference or model choice.
"""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
from sandbox import transport
import phase2_energy as energy

BASE=energy.BASE;PARENT=BASE/'phase19_delivery/leedy2018_v2';OUT=BASE/'phase20_source_inventory/external_leedy'
BUNDLE=BASE/'phase20_source_inventory/delivery_v2/predictor_bundle_v7'
SCENARIOS=('primary_graph','prose_0025','nonuniform_upper','reading_lower','reading_upper')
ANCHORS={0:[],1:['0.025'],2:['0.025','0.1']}


def freeze():
    manifest=json.loads((BUNDLE/'manifest.json').read_text());assert manifest['version']==7
    paths=(Path(__file__),Path(__file__).parent/'phase20_runtime.py',ROOT/'src/sandbox/relative_predictor_v2.py')
    inputs=(BUNDLE/'manifest.json',PARENT/'samples.csv',PARENT/'protocol.json',PARENT/'reader_amendment.json',PARENT/'precursor_excluded_transport.json')
    p=dict(created_utc=energy.now(),code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        main_methods=list(manifest['variants'])+manifest['energy_methods'],scenarios=SCENARIOS,anchors=ANCHORS,
        main_pressure_atm=1e-5,pressure_sensitivity_atm=.065/101325.,
        diagnostic_policies=['no_update','unpenalized'],diagnostic_methods=list(manifest['variants']),
        diagnostics='primarygraph only,budgets1/2,same two fixedpolicies forall10models;not queryselected or trainedalgorithms',
        pressure_sensitivity='primarygraph,trainedpolicy,budgets0/1/2,all17models;uncalibratedscenario,not measuredthermalhistory',
        score='fullnumericcoverage required;allquery,uniformquery,fixedcommon3,fixedcommonuniform2',
        source_grade='knownLeedy2018 development source;notblind;2017precursor inold52conditionalmu',
        caveats='actualSIMS;0.05nonuniform;0.025graphproseconflict;3aboveoldSidoserange;300Kcomparisonconvention',
        mobility='deliveredhead andseparatefrozen49pair10sourceCTcalibration;no targetmuanchors',
        new_training_labels=0,new_optimizer_starts=0,new_measurements=0,goal_complete=False)
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==p[k],k
    else:energy.write(path,p)
    return manifest


def table(scenario):
    d=read(PARENT/'samples.csv');d['sample_label']=d.sample_id.str.rsplit('::target-',n=1).str[-1]
    if scenario=='prose_0025':d.loc[d.sample_label.eq('0.025'),['observed_nH_cm3','observed_muH_cm2Vs']]=[3.25e19,20.]
    if scenario=='nonuniform_upper':d.loc[d.sample_label.eq('0.05'),'incorporated_cm3']=5e20
    if scenario.startswith('reading_'):
        side='low' if scenario.endswith('lower') else 'high';d['observed_nH_cm3']=d['nH_reading_'+side];d['observed_muH_cm2Vs']=d['mu_reading_'+side]
    return d


def main():
    manifest=freeze();queries=[];expected={};methods=list(manifest['variants'])+manifest['energy_methods']
    plans=[(scenario,'trained',1e-5,methods,(0,1,2)) for scenario in SCENARIOS]
    plans += [('primary_graph',policy,1e-5,list(manifest['variants']),(1,2)) for policy in ('no_update','unpenalized')]
    plans += [('primary_graph','trained',.065/101325.,methods,(0,1,2))]
    for scenario,policy,pressure,models,budgets in plans:
        d=table(scenario)
        for method in models:
            for budget in budgets:
                anchors=[dict(incorporated_cm3=float(d[d.sample_label.eq(label)].incorporated_cm3.iloc[0]),
                    measured_nH_cm3=float(d[d.sample_label.eq(label)].observed_nH_cm3.iloc[0]),growth_T_C=550.) for label in ANCHORS[budget]]
                for r in d.itertuples():
                    key=f'q{len(queries)}';queries.append(dict(id=key,policy=policy,input=dict(dopant='Si',
                        incorporated_cm3=float(r.incorporated_cm3),method=method,growth_T_C=550.,anchors=anchors,pO2_atm=pressure)))
                    expected[key]=dict(scenario=scenario,policy=policy,pressure_atm=pressure,method=method,budget=budget,
                        sample_id=r.sample_id,sample_label=r.sample_label,role='anchor' if r.sample_label in ANCHORS[budget] else 'query',
                        common_query=r.sample_label not in ANCHORS[2],uniform_scalar_input=bool(r.uniform_scalar_input),
                        incorporated_cm3=float(r.incorporated_cm3),observed_log_nH=float(np.log10(r.observed_nH_cm3)),
                        observed_log_mu=float(np.log10(r.observed_muH_cm2Vs)),
                        observed_log_sigma=float(np.log10(transport.E*r.observed_nH_cm3*r.observed_muH_cm2Vs)))
    energy.write(OUT/'tasks.json',dict(bundle=str(BUNDLE.resolve()),queries=queries));energy.write(OUT/'parent_reference.json',expected)
    output=OUT/'independent_outputs.json'
    run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase20_runtime.py'),'--worker',str(OUT/'tasks.json'),str(output)],cwd=ROOT,capture_output=True,text=True)
    (OUT/'worker.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode();actual=json.loads(output.read_text());assert not actual['blocked_paths']
    offset=json.loads((PARENT/'precursor_excluded_transport.json').read_text())['offset_dex'];rows=[]
    for q in actual['rows']:
        r=dict(id=q['id'],**expected[q['id']],numeric=q['error'] is None,error=q['error'])
        if q['result'] is not None:
            p=q['result'];mu=float(np.log10(transport.mu_caughey_thomas(p['nH_cm3'],300.))+offset)
            r.update(predicted_log_nH=p['log10_nH_cm3'],predicted_log_mu=p['log10_muH_cm2Vs'],predicted_log_sigma=p['log10_sigma_S_cm'],
                precursor_excluded_log_mu=mu,source_effect_dex=p.get('source_effect_dex'),raw_source_coordinate=p.get('raw_source_coordinate'))
            assert abs(p['log10_sigma_S_cm']-np.log10(transport.E)-p['log10_nH_cm3']-p['log10_muH_cm2Vs'])<1e-9
        rows.append(r)
    d=pd.DataFrame(rows);d.to_csv(OUT/'predictions.csv',index=False);summary=[]
    for keys,g in d[d.role.eq('query')].groupby(['scenario','policy','pressure_atm','method','budget']):
        for scope in ('all_query','uniform_query','common3','common_uniform2'):
            q=g.copy()
            if scope in ('uniform_query','common_uniform2'):q=q[q.uniform_scalar_input]
            if scope in ('common3','common_uniform2'):q=q[q.common_query]
            row=dict(zip(['scenario','policy','pressure_atm','method','budget'],keys));row.update(scope=scope,queries=len(q),numeric=int(q.numeric.sum()))
            for prop in ('nH','mu','sigma'):row[prop+'_MAE']=float(abs(q['predicted_log_'+prop]-q['observed_log_'+prop]).mean()) if q.numeric.all() else None
            row['precursor_excluded_mu_MAE']=float(abs(q.precursor_excluded_log_mu-q.observed_log_mu).mean()) if q.numeric.all() else None
            summary.append(row)
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'summary.csv',index=False)
    energy.write(OUT/'audit.json',dict(passed=not actual['blocked_paths'] and len(d)==1730,rows=len(d),numeric_rows=int(d.numeric.sum()),
        main_scenario_rows=int((d.pressure_atm.eq(1e-5)&d.policy.eq('trained')).sum()),
        diagnostic_rows=int(d.policy.ne('trained').sum()),property_tables_read=False,query_labels_in_worker=False,
        source='knownLeedy2018development',unique_published_conditions=5,independent_sigma_labels=0,new_training_labels=0,goal_complete=False))
    main=summary[summary.scenario.eq('primary_graph')&summary.policy.eq('trained')&summary.pressure_atm.eq(1e-5)&summary.scope.eq('all_query')]
    print(main[['method','budget','queries','numeric','nH_MAE','mu_MAE']].to_string(index=False),flush=True)
    assert len(d)==1730,len(d)


if __name__=='__main__':main()
