"""Evaluate chemical-only paired-selected calibrators with every same-budget property control."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from sandbox.chemical_process import SavedChemicalCalibrations
from csv_io_v2 import read_result_csv as read
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';DEST=OUT/'paired_property_prediction';PREV=BASE/'phase23_process_inputs'


def shard(i):
    task=DEST/f'tasks_{i}.json';output=DEST/f'outputs_{i}.json'
    p=subprocess.run([sys.executable,str(Path(__file__).parent/'phase24_pair_runtime.py'),'--worker',str(task),str(output)],capture_output=True,text=True,cwd=ROOT)
    (DEST/f'worker_{i}.log').write_text(p.stdout+'\n'+p.stderr);p.check_returncode();return json.loads(output.read_text())


def main():
    assert json.loads((OUT/'paired_validation/audit.json').read_text())['passed'];chemical=OUT/'delivery/paired_chemical_bundle';cal=SavedChemicalCalibrations(chemical)
    configs=json.loads((OUT/'property_prediction/protocol.json').read_text())['models']
    files=(Path(__file__),Path(__file__).parent/'phase24_pair_runtime.py',Path(__file__).parent/'phase24_property_runtime.py',ROOT/'docs/ML_SANDBOX_PAIRED_CHEMICAL_PROTOCOL_20261018.md')
    inputs=(chemical/'manifest.json',OUT/'paired_validation/audit.json',PREV/'process_audit/published_Hall_table.csv',OUT/'property_prediction/protocol.json')
    p=dict(created_utc=e.now(),code_hashes={str(f.relative_to(ROOT)):e.sha(f) for f in files},input_hashes={str(f.relative_to(ROOT)):e.sha(f) for f in inputs},
        models=configs,chemical_bundle=str(chemical),chemical_models=list(cal.manifest['models']),requests=6588,workers=4,
        Hall_anchor_labels=['d','c'],budgets=[0,1,2],common_query='all12 exceptc/d',selection='chemicalpairedlevelCVonly;noHallqueries',
        old26352predictions_unchanged=True,new_Hall_training=0,goal_complete=False)
    path=DEST/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['code_hashes']==p['code_hashes']
    else:e.write(path,p)
    d=read(PREV/'process_audit/published_Hall_table.csv').reset_index(drop=True);source=[dict(source_T_C=float(r.SiO2_source_C),velocity_um_h=.86 if r.Ga2O_flux_cm2s==6e14 else 1.3) for r in d.itertuples()]
    aids=[int(d[d.sample_label.eq(a)].index[0]) for a in ('d','c')];tasks=[];ref={};shards=[[],[],[],[]]
    for cname in cal.manifest['models']:
        c=cal.get(cname);scenario=cname.split('::')[0];cc=[c.predict(x['source_T_C'],x['velocity_um_h']) for x in source]
        for cfg in configs:
            shard_id=int(hashlib.sha256(json.dumps([cname,cfg['context'],cfg['method']]).encode()).hexdigest()[:8],16)%4
            for budget in (0,1,2):
                anchors=[dict(input_index=i,measured_nH_cm3=float(d.iloc[i].nH_cm3)) for i in aids[:budget]]
                for i,r in enumerate(d.itertuples()):
                    key=f'q{len(tasks)}';q=dict(id=key,engine=cfg['engine'],property_bundle=cfg['bundle'],chemical_bundle=str(chemical),
                        input=dict(source_T_C=source[i]['source_T_C'],velocity_um_h=source[i]['velocity_um_h'],chemical_model=cname,method=cfg['method'],source_inputs=source,anchors=anchors,growth_T_C=525.,observation_T_K=300.,pO2_atm=1e-5))
                    tasks.append(q);shards[shard_id].append(q);ref[key]=dict(context=cfg['context'],method=cfg['method'],chemical_model=cname,selected_chemical_family=c.record['family'],scenario=scenario,budget=budget,
                        sample_label=r.sample_label,sample_id=r.sample_id,role='anchor' if i in aids[:budget] else 'query',common_query=i not in aids,
                        expected_logC=cc[i]['log10_C_cm3'],observed_log_nH=float(np.log10(r.nH_cm3)),observed_log_mu=float(np.log10(r.muH_cm2Vs)),observed_log_sigma=float(np.log10(1.602176634e-19*r.nH_cm3*r.muH_cm2Vs)))
    assert len(tasks)==6588;e.write(DEST/'tasks.json',dict(queries=tasks));e.write(DEST/'parent_reference.json',ref)
    for i,t in enumerate(shards):e.write(DEST/f'tasks_{i}.json',dict(queries=t))
    e.write(DEST/'execution_started.json',dict(created_utc=e.now(),requests=len(tasks),shard_sizes=[len(t) for t in shards]))
    with ThreadPoolExecutor(max_workers=4) as pool:outputs=list(pool.map(shard,range(4)))
    rows=[]
    for w in outputs:
        assert not w['blocked_paths'] and not w['paired_training_blocked_paths']
        for q in w['rows']:
            r=dict(id=q['id'],**ref[q['id']],numeric=q['error'] is None,error=q['error'])
            if q['result'] is not None:
                v=q['result'];r.update(predicted_log_nH=v['log10_nH_cm3'],predicted_log_mu=v['log10_muH_cm2Vs'],predicted_log_sigma=v['log10_sigma_S_cm'],runtime_logC=v['process_calibration']['log10_C_cm3'])
                assert abs(r['runtime_logC']-r['expected_logC'])<1e-10
            rows.append(r)
    df=pd.DataFrame(rows);df.to_csv(DEST/'predictions.csv',index=False);summary=[];keys=['context','method','chemical_model','selected_chemical_family','scenario','budget']
    for k,g in df[df.role.eq('query')].groupby(keys):
        for scope in ('all_query','common_query'):
            q=g if scope=='all_query' else g[g.common_query];r=dict(zip(keys,k),scope=scope,queries=len(q),numeric=int(q.numeric.sum()))
            for prop in ('nH','mu','sigma'):r[prop+'_MAE']=float(np.mean(abs(q['predicted_log_'+prop]-q['observed_log_'+prop]))) if q.numeric.all() else None
            summary.append(r)
    e.csvwrite(DEST/'summary.csv',summary);e.write(DEST/'execution_complete.json',dict(created_utc=e.now(),requests=len(df),numeric=int(df.numeric.sum()),refusals=int((~df.numeric).sum()),summary_rows=len(summary),new_Hall_training=0,all_failures_retained=True,goal_complete=False))
    print('paired-selected process/property',len(df),'numeric',int(df.numeric.sum()),'refusals',int((~df.numeric).sum()),flush=True)


if __name__=='__main__':main()
