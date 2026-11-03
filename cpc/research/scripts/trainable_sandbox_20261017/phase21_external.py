"""Known Leedy source transfer under frozen policies and fixed diagnostics."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox import transport
from csv_io_v2 import read_result_csv as read
import phase21_prediction as s
import phase21_export as export
import phase21_affine_controls as affine
import phase20_external as previous
OUT=s.OUT/'external_leedy';SCENARIOS=previous.SCENARIOS;ANCHORS=previous.ANCHORS


def freeze():
    files=(Path(__file__),Path(__file__).parent/'phase21_runtime.py',ROOT/'src/sandbox/functional_predictor.py',ROOT/'src/sandbox/affine_predictor.py')
    inputs=(export.BUNDLE/'manifest.json',s.OUT/'prediction_protocol.json',previous.PARENT/'samples.csv',previous.PARENT/'precursor_excluded_transport.json',affine.OUT/'predictor_bundle/manifest.json')
    p=dict(created_utc=s.energy.now(),code_hashes={str(f.relative_to(ROOT)):s.energy.sha(f) for f in files},
        input_hashes={str(f.relative_to(ROOT)):s.energy.sha(f) for f in inputs},
        source='Leedy2018known development source;not blind or independentlab',scenarios=SCENARIOS,anchors=ANCHORS,
        source_design='allknown5inputconditions,duplicatesdeduplicated;actualSIMS/nonuniformupper alternative;noquerylabels',
        main='trainedpolicy24methods0/1/2budgets;allfive label/inputscenarios,p1e-5atm',
        diagnostic='all17methods including3affinecontrols,no_update/unpenalized,primarygraph,p1e-5atm,budgets1/2;fixedbeforeevaluating',
        pressure='all24methods,trainedpolicy,primarygraph,.065/101325atm,0/1/2;uncalibratedprocesssensitivity',
        caveats='0.025graphproseconflict,0.05nonuniform,3highSidoseextrapolations,300Kcomparisonconvention,2017precursorinoldconditionalmu',
        new_training_labels=0,new_shared_optimizer_starts=0,new_material_measurements=0,goal_complete=False)
    dest=OUT/'protocol.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==p[k]
    else:s.energy.write(dest,p)


def main():
    freeze();manifest=json.loads((export.BUNDLE/'manifest.json').read_text());extra=json.loads((affine.OUT/'predictor_bundle/manifest.json').read_text())
    newmethods=list(manifest['variants'])+list(extra['variants']);methods=newmethods+manifest['energy_methods']
    plans=[(z,'trained',1e-5,methods,(0,1,2)) for z in SCENARIOS]
    plans += [('primary_graph',p,1e-5,newmethods,(1,2)) for p in ('no_update','unpenalized')]
    plans += [('primary_graph','trained',.065/101325.,methods,(0,1,2))]
    queries=[];reference={}
    for scenario,policy,pressure,models,budgets in plans:
        d=previous.table(scenario);lookup={str(r.sample_label):i for i,r in enumerate(d.itertuples())}
        source_inputs=[dict(dopant='Si',incorporated_cm3=float(r.incorporated_cm3),growth_T_C=550.) for r in d.itertuples()]
        for method in models:
            for budget in budgets:
                anchors=[dict(input_index=lookup[label],measured_nH_cm3=float(d[d.sample_label.eq(label)].observed_nH_cm3.iloc[0])) for label in ANCHORS[budget]]
                for r in d.itertuples():
                    key=f'q{len(queries)}';queries.append(dict(id=key,policy=policy,engine='affine' if method in affine.CONFIG else 'functional',
                        bundle=str((affine.OUT/'predictor_bundle' if method in affine.CONFIG else export.BUNDLE).resolve()),input=dict(dopant='Si',incorporated_cm3=float(r.incorporated_cm3),
                        growth_T_C=550.,method=method,pO2_atm=pressure,source_inputs=source_inputs,anchors=anchors)))
                    reference[key]=dict(scenario=scenario,policy=policy,pressure_atm=pressure,method=method,budget=budget,sample_id=r.sample_id,sample_label=r.sample_label,
                        role='anchor' if r.sample_label in ANCHORS[budget] else 'query',common_query=r.sample_label not in ANCHORS[2],
                        uniform_scalar_input=bool(r.uniform_scalar_input),incorporated_cm3=float(r.incorporated_cm3),
                        observed_log_nH=float(np.log10(r.observed_nH_cm3)),observed_log_mu=float(np.log10(r.observed_muH_cm2Vs)),
                        observed_log_sigma=float(np.log10(transport.E*r.observed_nH_cm3*r.observed_muH_cm2Vs)))
    s.energy.write(OUT/'tasks.json',dict(bundle=str(export.BUNDLE.resolve()),queries=queries));s.energy.write(OUT/'parent_reference.json',reference)
    output=OUT/'independent_outputs.json';run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase21_runtime.py'),
        '--worker',str(OUT/'tasks.json'),str(output)],cwd=ROOT,capture_output=True,text=True)
    (OUT/'worker.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode();actual=json.loads(output.read_text());assert not actual['blocked_paths']
    rows=[];offset=json.loads((previous.PARENT/'precursor_excluded_transport.json').read_text())['offset_dex']
    for q in actual['rows']:
        r=dict(id=q['id'],**reference[q['id']],numeric=q['error'] is None,error=q['error']);p=q['result']
        if p is not None:
            r.update(predicted_log_nH=p['log10_nH_cm3'],predicted_log_mu=p['log10_muH_cm2Vs'],predicted_log_sigma=p['log10_sigma_S_cm'],
                raw_source_coordinate=p.get('raw_source_coordinate'),source_design_size=p.get('source_design_size'),
                precursor_excluded_log_mu=float(np.log10(transport.mu_caughey_thomas(p['nH_cm3'],300.))+offset))
            assert abs(p['log10_sigma_S_cm']-np.log10(transport.E)-p['log10_nH_cm3']-p['log10_muH_cm2Vs'])<1e-9
        rows.append(r)
    d=pd.DataFrame(rows);d.to_csv(OUT/'predictions.csv',index=False);summary=[]
    for keys,g in d[d.role.eq('query')].groupby(['scenario','policy','pressure_atm','method','budget']):
        for scope in ('all_query','uniform_query','common3','common_uniform2'):
            q=g.copy()
            if scope in ('uniform_query','common_uniform2'):q=q[q.uniform_scalar_input]
            if scope in ('common3','common_uniform2'):q=q[q.common_query]
            r=dict(zip(['scenario','policy','pressure_atm','method','budget'],keys));r.update(scope=scope,queries=len(q),numeric=int(q.numeric.sum()))
            for prop in ('nH','mu','sigma'):r[prop+'_MAE']=float(abs(q['predicted_log_'+prop]-q['observed_log_'+prop]).mean()) if q.numeric.all() else np.nan
            r['precursor_excluded_mu_MAE']=float(abs(q.precursor_excluded_log_mu-q.observed_log_mu).mean()) if q.numeric.all() else np.nan
            summary.append(r)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    audit=dict(passed=not actual['blocked_paths'] and len(d)==2500,requests=len(d),numeric_requests=int(d.numeric.sum()),
        failures=[r for r in rows if not r['numeric']],main_scenario_requests=int((d.policy.eq('trained')&d.pressure_atm.eq(1e-5)).sum()),
        fixed_diagnostic_requests=int(d.policy.ne('trained').sum()),property_tables_read=False,query_labels_in_worker=False,
        grade='known-sourceinput-setconditionaldevelopment;notblind',new_training_labels=0,independent_sigma_labels=0,goal_complete=False)
    s.energy.write(OUT/'audit.json',audit);print('Leedy requests',len(d),'numeric',int(d.numeric.sum()),flush=True);assert audit['passed']


if __name__=='__main__':main()
