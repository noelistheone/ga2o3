"""Full own-temperature property evaluation; retain every posterior-domain refusal."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,sys,copy,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase22_export as ex
import phase21_cohort as old
p=ex.p;s=p.s;OUT=p.OUT/'cohort_properties'


def freeze():
    paths=(Path(__file__),Path(__file__).parent/'phase22_runtime.py',ROOT/'src/sandbox/ensemble_predictor.py',ROOT/'src/sandbox/source_ensemble.py')
    sources=(ex.BUNDLE/'manifest.json',p.OUT/'execution_complete.json',p.OUT/'all_selections.json',s.BASE/'episodes_v1.csv')
    protocol=dict(created_utc=s.energy.now(),code_hashes={str(f.relative_to(ROOT)):s.energy.sha(f) for f in paths},
        input_hashes={str(f.relative_to(ROOT)):s.energy.sha(f) for f in sources},
        design='all10methods,old29/5 andRTanchors;142nT/7Si/2DOI,5RTmu/2DOI,54muT/3Si/1DOI;no newtraininglabels',
        domain='all positive posterior support must satisfy own-T observation domain; refusals retained,not renormalized',
        replay='fresh process,property-table reads blocked;all850 nonguardRT rows tol2e-6dex;fullquery/no dropna',
        workers=2,independent_sigma_labels=0,goal_complete=False)
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['code_hashes']==protocol['code_hashes']
    else:s.energy.write(path,protocol)


def execute_shard(i):
    task=OUT/f'tasks_shard{i}.json';dest=OUT/f'independent_outputs_shard{i}.json'
    run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase22_runtime.py'),'--worker',str(task),str(dest)],cwd=ROOT,capture_output=True,text=True)
    (OUT/f'worker_shard{i}.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode();return json.loads(dest.read_text())


def main():
    freeze();selections=json.loads((p.OUT/'all_selections.json').read_text());targets=s.parent.inputs.TARGETS.copy()
    saved=read(p.OUT/'predictions.csv');rt=targets[targets.endpoint.eq('nH')&targets.kind.eq('RT')].set_index('sample_id')
    queries=[];expected={};replays={};index=[];excluded=[];shards=[[],[]]
    for method in p.VARIANTS:
        for held in s.FAMILIES:
            choices=[c for c in selections if c['variant']==method and c['held']==held];bundle=OUT/'bundles'/s.digest([method,held]);m=ex.build(bundle,choices,grade='sourceexcluded retrospective finite-source research')
            train=s.RT[s.RT.family.ne(held)];m.update(held_source=held,train_sources=sorted(train.family.unique()),
                chemical_support_cm3={e:[float(g.incorporated_cm3.min()),float(g.incorporated_cm3.max())] for e,g in train.groupby('element')})
            m['training_labels'].update(nH_RT_specimens=len(train),sources=int(train.family.nunique()))
            s.energy.write(bundle/'manifest.json',m);index.append(dict(method=method,held=held,bundle=str(bundle.relative_to(ROOT)),fit_key=choices[0]['fit_key']))
            shard=(len(index)-1)%2
            for c in choices:
                b=c['budget'];d=s.table(held,b);roles=d.set_index('sample_id').role;common=set(s.table(held,2).query("role=='query'").sample_id)
                nodes,ai,y,_=s.design_inputs(d);inputs=[old.api_input(r) for r in nodes] if b else [];anchors=[dict(input_index=int(i),measured_nH_cm3=float(10**v)) for i,v in zip(ai,y)]
                reference=saved[saved.variant.eq(method)&saved.budget.eq(b)].set_index('sample_id')
                for t in targets[targets.family.eq(held)].itertuples():
                    if roles[t.sample_id]=='guard':excluded.append(dict(method=method,budget=b,obs_id=t.obs_id,reason='guard not in source design'));continue
                    key=f'q{len(queries)}';q=old.api_input(t);q.update(method=method,source_inputs=inputs,anchors=anchors)
                    task=dict(id=key,bundle=str(bundle.resolve()),input=q);queries.append(task);shards[shard].append(task)
                    expected[key]=dict(method=method,budget=b,family=held,sample_id=t.sample_id,obs_id=t.obs_id,endpoint=t.endpoint,kind=t.kind,T_K=float(t.Tobs_K),
                        observed=float(t.target),role=roles[t.sample_id],common_query=t.sample_id in common)
                    if t.endpoint=='muH' and t.kind=='RT':expected[key]['paired_nH']=float(rt.loc[t.sample_id].target)
                    if t.endpoint=='nH' and t.kind=='RT':replays[key]=float(reference.loc[t.sample_id].predicted)
    assert len(queries)==6880 and len(replays)==850
    s.energy.write(OUT/'tasks.json',dict(queries=queries));s.energy.write(OUT/'parent_reference.json',expected);s.energy.write(OUT/'index.json',index);s.energy.write(OUT/'excluded_guard_requests.json',excluded)
    for i,qs in enumerate(shards):s.energy.write(OUT/f'tasks_shard{i}.json',dict(queries=qs))
    with ThreadPoolExecutor(max_workers=2) as pool:outputs=list(pool.map(execute_shard,range(2)))
    rows=[];derived=[];replay=[]
    for actual in outputs:
        assert not actual['blocked_paths']
        for q in actual['rows']:
            r=dict(id=q['id'],**expected[q['id']],numeric=q['error'] is None,error=q['error']);v=q['result']
            if v is not None:
                r.update(predicted=v['log10_nH_cm3'] if r['endpoint']=='nH' else v['log10_muH_cm2Vs'],
                    predicted_log_nH=v['log10_nH_cm3'],predicted_log_mu=v['log10_muH_cm2Vs'],predicted_log_sigma=v['log10_sigma_S_cm'],
                    raw_source_coordinate=v['raw_source_coordinate'],posterior_spike=v['source_calibration'].get('posterior_spike_probability'))
                assert abs(v['log10_sigma_S_cm']-np.log10(1.602176634e-19)-v['log10_nH_cm3']-v['log10_muH_cm2Vs'])<1e-8
            if q['id'] in replays:replay.append(dict(id=q['id'],error_dex=abs(r['predicted']-replays[q['id']]) if r['numeric'] else None,passed=bool(r['numeric'] and abs(r['predicted']-replays[q['id']])<2e-6)))
            rows.append(r)
            if r['endpoint']=='muH' and r['kind']=='RT':
                z=dict(r);z.update(endpoint='sigma_derived',observed=float(np.log10(1.602176634e-19)+r['paired_nH']+r['observed']))
                if v is not None:z['predicted']=v['log10_sigma_S_cm']
                derived.append(z)
    d=pd.DataFrame(rows+derived);d.to_csv(OUT/'predictions.csv',index=False);by=[]
    for scope in ('all_eligible','query','common_query'):
        z=d if scope=='all_eligible' else d[d.role.eq('query')] if scope=='query' else d[d.common_query]
        for keys,g in z.groupby(['method','budget','endpoint','kind','family','sample_id']):
            by.append(dict(zip(['method','budget','endpoint','kind','family','sample_id'],keys),scope=scope,reads=len(g),numeric=int(g.numeric.sum()),
                MAE=float(abs(g.predicted-g.observed).mean()) if g.numeric.all() else None))
    by=pd.DataFrame(by);by.to_csv(OUT/'by_specimen.csv',index=False);families=[]
    for keys,g in by.groupby(['method','budget','endpoint','kind','scope','family']):
        families.append(dict(zip(['method','budget','endpoint','kind','scope','family'],keys),specimens=len(g),reads=int(g.reads.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else None))
    f=pd.DataFrame(families);f.to_csv(OUT/'by_family.csv',index=False);summaries=[]
    for keys,g in f.groupby(['method','budget','endpoint','kind','scope']):
        summaries.append(dict(zip(['method','budget','endpoint','kind','scope'],keys),families=len(g),reads=int(g.reads.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else None))
    s.energy.csvwrite(OUT/'summary.csv',summaries);s.energy.csvwrite(OUT/'RT_replay.csv',replay)
    r=dict(passed=all(v['passed'] for v in replay) and len(rows)==6880,sourceexcluded_bundles=len(index),inference_requests=len(rows),numeric_requests=sum(x['numeric'] for x in rows),
        all_requests_numeric=all(x['numeric'] for x in rows),RT_replays=len(replay),maximum_RT_replay_error=max(v['error_dex'] for v in replay if v['error_dex'] is not None),
        failed_requests=[x for x in rows if not x['numeric']],point_cost=[a['point_cost'] for a in outputs],guard_requests_explicitly_excluded=len(excluded),
        property_tables_read=False,query_labels_in_worker=False,new_temperature_mu_training_labels=0,independent_sigma_labels=0,goal_complete=False)
    s.energy.write(OUT/'audit.json',r);print('cohort',len(rows),'numeric',r['numeric_requests'],'RTreplay',len(replay),'pass',r['passed'],flush=True);assert r['passed']


if __name__=='__main__':main()
