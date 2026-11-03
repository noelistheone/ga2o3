"""Run every chemical model/reading scenario through all frozen property methods, same Hall anchors."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from sandbox.chemical_process import SavedChemicalCalibrations
from csv_io_v2 import read_result_csv as read
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';DEST=OUT/'property_prediction';PARENT=BASE/'phase23_process_inputs'


def freeze(configs,chemical):
    files=(Path(__file__),Path(__file__).parent/'phase24_property_runtime.py',ROOT/'src/sandbox/chemical_process.py',ROOT/'src/sandbox/process_property_predictor.py',ROOT/'docs/ML_SANDBOX_CHEMICAL_LEARNING_PROTOCOL_20261018.md')
    inputs=(PARENT/'process_audit/published_Hall_table.csv',PARENT/'source_prediction/protocol.json',chemical/'manifest.json',OUT/'chemical_learning/audit.json',BASE/'artifact_manifest_phase23.json')
    p=dict(created_utc=e.now(),code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in files},input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs},
        models=configs,chemical_models=list(SavedChemicalCalibrations(chemical).manifest['models']),Hall_anchor_labels=['d','c'],budgets=[0,1,2],common_queries='all exceptc/d',
        same_previous12_queries_and_anchors=True,chemistry_budget='12 independent plottedlayerconditions/2stacks perfixedreadingwindow,allmethods receivesametrainedproxy',
        fixed_property_weights_and_sourcehyperparameters=True,new_Hall_training=0,all_model_scenarios_retained=True,no_blind_claim=True,
        workers=4,requests=26352,old_printed_calibration_reference='frozenphase23regime_rates;not rerun or usedfor learning chemicalmodel',goal_complete=False)
    path=DEST/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['code_hashes']==p['code_hashes'] and json.loads(path.read_text())['input_hashes']==p['input_hashes']
    else:e.write(path,p)


def shard(i):
    task=DEST/f'tasks_{i}.json';out=DEST/f'outputs_{i}.json'
    run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase24_property_runtime.py'),'--worker',str(task),str(out)],cwd=ROOT,capture_output=True,text=True)
    (DEST/f'worker_{i}.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode();return json.loads(out.read_text())


def main():
    assert json.loads((OUT/'chemical_learning/audit.json').read_text())['passed']
    configs=json.loads((PARENT/'source_prediction/protocol.json').read_text())['configs'];chemical=OUT/'delivery/chemical_bundle';calibrations=SavedChemicalCalibrations(chemical)
    freeze(configs,chemical);d=read(PARENT/'process_audit/published_Hall_table.csv').reset_index(drop=True);tasks=[];shards=[[],[],[],[]];ref={};inputs=[]
    source=[dict(source_T_C=float(r.SiO2_source_C),velocity_um_h=.86 if r.Ga2O_flux_cm2s==6e14 else 1.3) for r in d.itertuples()];aids=[int(d[d.sample_label.eq(a)].index[0]) for a in ('d','c')]
    for cname in calibrations.manifest['models']:
        cal=calibrations.get(cname);scenario,cmethod=cname.split('::');values=[cal.predict(x['source_T_C'],x['velocity_um_h']) for x in source]
        inputs.extend(dict(chemical_model=cname,scenario=scenario,calibration_method=cmethod,sample_label=r.sample_label,sample_id=r.sample_id,
            source_T_C=source[i]['source_T_C'],velocity_um_h=source[i]['velocity_um_h'],predicted_log10_C=values[i]['log10_C_cm3'],predicted_C_cm3=values[i]['C_cm3'],
            grade='learnedSIMS processproxy,not actualsame-Hall chemicalvalue') for i,r in enumerate(d.itertuples()))
        for c in configs:
            si=int(hashlib.sha256(json.dumps([cname,c['context'],c['method']]).encode()).hexdigest()[:8],16)%4
            for budget in (0,1,2):
                anchors=[dict(input_index=j,measured_nH_cm3=float(d.iloc[j].nH_cm3)) for j in aids[:budget]]
                for j,r in enumerate(d.itertuples()):
                    key=f'q{len(tasks)}';task=dict(id=key,engine=c['engine'],property_bundle=c['bundle'],chemical_bundle=str(chemical),
                        input=dict(source_T_C=source[j]['source_T_C'],velocity_um_h=source[j]['velocity_um_h'],chemical_model=cname,method=c['method'],source_inputs=source,anchors=anchors,
                            growth_T_C=525.,observation_T_K=300.,pO2_atm=1e-5));tasks.append(task);shards[si].append(task)
                    ref[key]=dict(context=c['context'],method=c['method'],engine=c['engine'],chemical_model=cname,calibration_method=cmethod,scenario=scenario,budget=budget,sample_label=r.sample_label,
                        sample_id=r.sample_id,role='anchor' if j in aids[:budget] else 'query',common_query=j not in aids,
                        predicted_log10_C=values[j]['log10_C_cm3'],observed_log_nH=float(np.log10(r.nH_cm3)),observed_log_mu=float(np.log10(r.muH_cm2Vs)),
                        observed_log_sigma=float(np.log10(1.602176634e-19*r.nH_cm3*r.muH_cm2Vs)))
    assert len(tasks)==26352;e.write(DEST/'tasks.json',dict(queries=tasks));e.write(DEST/'parent_reference.json',ref);e.csvwrite(DEST/'source_inputs.csv',inputs)
    for i,t in enumerate(shards):e.write(DEST/f'tasks_{i}.json',dict(queries=t))
    e.write(DEST/'execution_started.json',dict(created_utc=e.now(),requests=len(tasks),shard_sizes=[len(s) for s in shards],new_property_model_fitting=0))
    with ThreadPoolExecutor(max_workers=4) as pool:outputs=list(pool.map(shard,range(4)))
    rows=[]
    for worker in outputs:
        assert not worker['blocked_paths']
        for r in worker['rows']:
            row=dict(id=r['id'],**ref[r['id']],numeric=r['error'] is None,error=r['error'])
            if r['result'] is not None:
                p=r['result'];row.update(predicted_log_nH=p['log10_nH_cm3'],predicted_log_mu=p['log10_muH_cm2Vs'],predicted_log_sigma=p['log10_sigma_S_cm'],
                    runtime_log10_C=p['process_calibration']['log10_C_cm3'],chemical_extrapolation_warning=bool(p['process_calibration']['warnings']))
                assert abs(row['runtime_log10_C']-row['predicted_log10_C'])<1e-10
            rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(DEST/'predictions.csv',index=False);summary=[]
    keys=['context','method','chemical_model','calibration_method','scenario','budget']
    for k,g in df[df.role.eq('query')].groupby(keys):
        for scope in ('all_query','common_query'):
            q=g if scope=='all_query' else g[g.common_query];row=dict(zip(keys,k),scope=scope,queries=len(q),numeric=int(q.numeric.sum()))
            for prop in ('nH','mu','sigma'):row[prop+'_MAE']=float(np.mean(abs(q['predicted_log_'+prop]-q['observed_log_'+prop]))) if q.numeric.all() else None
            summary.append(row)
    e.csvwrite(DEST/'summary.csv',summary);e.write(DEST/'execution_complete.json',dict(created_utc=e.now(),requests=len(df),numeric=int(df.numeric.sum()),refusals=int((~df.numeric).sum()),
        summary_rows=len(summary),all_failures_retained=True,new_shared_property_training=0,new_Hall_training_labels=0,independent_sigma_labels=0,goal_complete=False))
    print('coupled process/property complete',len(df),'numeric',int(df.numeric.sum()),'refusals',int((~df.numeric).sum()),flush=True)


if __name__=='__main__':main()
