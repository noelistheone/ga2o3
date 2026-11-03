"""Known-source transfer of frozen finite-state policies; all old scenarios retained."""
from pathlib import Path
import json,sys,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import phase22_export as ex
import phase20_external as old
p=ex.p;s=p.s;OUT=p.OUT/'external_leedy'


def main():
    paths=(Path(__file__),Path(__file__).parent/'phase22_runtime.py',ROOT/'src/sandbox/ensemble_predictor.py')
    inputs=(ex.BUNDLE/'manifest.json',old.PARENT/'samples.csv',p.OUT/'all_selections.json')
    protocol=dict(created_utc=s.energy.now(),code_hashes={str(f.relative_to(ROOT)):s.energy.sha(f) for f in paths},
        input_hashes={str(f.relative_to(ROOT)):s.energy.sha(f) for f in inputs},
        grade='known Leedy development source; not blind or independent lab',scenarios=old.SCENARIOS,anchors=old.ANCHORS,
        methods=list(p.VARIANTS),pressure=[1e-5,.065/101325.],design='5scenarios*10methods*3budgets*5conditions + primary pressure sensitivity=900',
        no_query_labels_for_hyperselection=True,new_shared_optimizer_starts=0,new_training_labels=0,goal_complete=False)
    dest=OUT/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text())['code_hashes']==protocol['code_hashes']
    else:s.energy.write(dest,protocol)
    queries=[];expected={};plans=[(z,1e-5) for z in old.SCENARIOS]+[('primary_graph',.065/101325.)]
    for scenario,pressure in plans:
        d=old.table(scenario);lookup={str(r.sample_label):i for i,r in enumerate(d.itertuples())};source=[dict(incorporated_cm3=float(r.incorporated_cm3),growth_T_C=550.) for r in d.itertuples()]
        for method in p.VARIANTS:
            for budget in (0,1,2):
                anchors=[dict(input_index=lookup[label],measured_nH_cm3=float(d[d.sample_label.eq(label)].observed_nH_cm3.iloc[0])) for label in old.ANCHORS[budget]]
                for r in d.itertuples():
                    key=f'q{len(queries)}';queries.append(dict(id=key,input=dict(method=method,dopant='Si',incorporated_cm3=float(r.incorporated_cm3),growth_T_C=550.,
                        pO2_atm=pressure,source_inputs=source,anchors=anchors)))
                    expected[key]=dict(method=method,budget=budget,scenario=scenario,pressure_atm=pressure,sample_id=r.sample_id,sample_label=r.sample_label,
                        role='anchor' if r.sample_label in old.ANCHORS[budget] else 'query',common_query=r.sample_label not in old.ANCHORS[2],uniform_scalar_input=bool(r.uniform_scalar_input),
                        observed_log_nH=float(np.log10(r.observed_nH_cm3)),observed_log_mu=float(np.log10(r.observed_muH_cm2Vs)),
                        observed_log_sigma=float(np.log10(1.602176634e-19*r.observed_nH_cm3*r.observed_muH_cm2Vs)))
    assert len(queries)==900
    s.energy.write(OUT/'tasks.json',dict(bundle=str(ex.BUNDLE.resolve()),queries=queries));s.energy.write(OUT/'parent_reference.json',expected)
    output=OUT/'independent_outputs.json';run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase22_runtime.py'),'--worker',str(OUT/'tasks.json'),str(output)],cwd=ROOT,capture_output=True,text=True)
    (OUT/'worker.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode();actual=json.loads(output.read_text());assert not actual['blocked_paths'];rows=[]
    for q in actual['rows']:
        r=dict(id=q['id'],**expected[q['id']],numeric=q['error'] is None,error=q['error']);v=q['result']
        if v is not None:r.update(predicted_log_nH=v['log10_nH_cm3'],predicted_log_mu=v['log10_muH_cm2Vs'],predicted_log_sigma=v['log10_sigma_S_cm'],
            raw_source_coordinate=v['raw_source_coordinate'],posterior_spike=v['source_calibration'].get('posterior_spike_probability'))
        rows.append(r)
    d=pd.DataFrame(rows);d.to_csv(OUT/'predictions.csv',index=False);summary=[]
    for keys,g in d[d.role.eq('query')].groupby(['method','budget','scenario','pressure_atm']):
        for scope in ('all_query','uniform_query','common3','common_uniform2'):
            q=g if scope=='all_query' else g[g.uniform_scalar_input] if scope=='uniform_query' else g[g.common_query] if scope=='common3' else g[g.common_query&g.uniform_scalar_input]
            r=dict(zip(['method','budget','scenario','pressure_atm'],keys),scope=scope,queries=len(q),numeric=int(q.numeric.sum()))
            for prop in ('nH','mu','sigma'):r[prop+'_MAE']=float(abs(q['predicted_log_'+prop]-q['observed_log_'+prop]).mean()) if q.numeric.all() else None
            summary.append(r)
    s.energy.csvwrite(OUT/'summary.csv',summary);r=dict(passed=len(rows)==900 and not actual['blocked_paths'],requests=len(rows),numeric_requests=int(d.numeric.sum()),all_numeric=bool(d.numeric.all()),
        failures=[r for r in rows if not r['numeric']],point_cost=actual['point_cost'],property_tables_read=False,query_labels_in_worker=False,
        grade='known-source development',new_training_labels=0,goal_complete=False)
    s.energy.write(OUT/'audit.json',r);print('Leedy',len(rows),'numeric',int(d.numeric.sum()),flush=True)


if __name__=='__main__':main()
