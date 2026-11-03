"""New published-source forecasts with explicit independent/calibrated input grades."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase23_process_inputs';DEST=OUT/'source_prediction';HELD='10.1063/5.0101132'
KB=8.617333262145e-5


def methods():
    b21=BASE/'phase21_curve_response';b22=BASE/'phase22_predictive_inference';rows=[]
    for bundle,engine in ((b21/'delivery/predictor_bundle_v8','functional'),(b21/'affine_controls/predictor_bundle','affine'),(b22/'delivery/predictor_bundle_v9','ensemble')):
        m=json.loads((bundle/'manifest.json').read_text())
        for name in list(m['variants'])+m.get('energy_methods',[]):rows.append(dict(context='full5',method=name,bundle=str(bundle),engine=engine))
    for path,default in ((b21/'cohort_properties/index.json','functional'),(b22/'cohort_properties/index.json','ensemble')):
        for r in json.loads(path.read_text()):
            if r['held']==HELD:rows.append(dict(context='exclude_related_McCandless',method=r['method'],bundle=str(ROOT/r['bundle']),engine=r.get('engine',default)))
    assert len(rows)==61
    return rows


def tables():
    raw=read(OUT/'process_audit/published_Hall_table.csv');alltables=[];J=4.6e28*np.exp(-5.3/(KB*(raw.SiO2_source_C+273.15)))
    for scenario in ('regime_rates','nominal_rate','faster40pct'):
        q=raw.copy();v=np.where(q.Ga2O_flux_cm2s.eq(6e14),.86,1.3)
        if scenario=='nominal_rate':v=np.ones(len(q))
        if scenario=='faster40pct':v*=1.4
        q['incorporated_cm3']=J/(v*1e-4/3600.);q['Si_flux_proxy_cm2s']=J;q['velocity_um_h_proxy']=v
        q['tier']='process_calibrated_12';q['scenario']=scenario;q['chemical_input_grade']='independent published calibration proxy;not same-specimen SIMS';alltables.append(q)
    q=alltables[0].sort_values(['incorporated_cm3','sample_label']);first=q.iloc[0];second=q[np.log10(q.incorporated_cm3/first.incorporated_cm3)>=.1].iloc[0]
    anchorsB=[first.sample_label,second.sample_label]
    for scenario,C in (('published_rounded',[2e18,3e19]),('rounding_lower',[1.5e18,2.5e19]),('rounding_upper',[2.5e18,3.5e19])):
        q=raw[raw.sample_label.isin(['e','k'])].sort_values('sample_label').copy();q['incorporated_cm3']=C
        q['tier']='same_specimen_prose_2';q['scenario']=scenario;q['chemical_input_grade']='rounded published same-specimen SIMS scales;precise profile reconstruction unresolved';alltables.append(q)
    return alltables,anchorsB


def freeze(configs,anchorsB):
    files=(Path(__file__),Path(__file__).parent/'phase23_source_runtime.py',ROOT/'docs/ML_SANDBOX_NEW_SOURCE_INPUT_PROTOCOL_20261018.md')
    inputs=(OUT/'process_audit/published_Hall_table.csv',OUT/'process_audit/source_contract.json',BASE/'artifact_manifest_phase22.json')
    manifests={str(Path(c['bundle'])/'manifest.json'):e.sha(Path(c['bundle'])/'manifest.json') for c in configs}
    p=dict(created_utc=e.now(),code_hashes={str(f.relative_to(ROOT)):e.sha(f) for f in files},input_hashes={str(f.relative_to(ROOT)):e.sha(f) for f in inputs},
        model_manifest_hashes=manifests,configs=configs,anchors_by_tier={'same_specimen_prose_2':['e'],'process_calibrated_12':anchorsB},
        anchor_rule='tierB primaryregime calibratedinput:lowest andnextdose>=0.1dex higher;identities fixed across allscenarios,no targetlabel selection',
        budgets={'same_specimen_prose_2':[0,1],'process_calibrated_12':[0,1,2]},queries='allnonanchor;fixedcommon=nonanchoratmaximum viable budget;no fabricated2anchorqueriesfortwo specimens',
        calibration='J=4.6e28exp(-5.3/(kB(TsourceC+273.15))),independentXRR/SIMS line asprinted;vproxy0.86/1.3 or1 or1.4x,not chemicaltruth',
        no_shared_or_sourcehyper_refit=True,oxygen='old effectiveO2p1e-5atm;actualO3cannot substitute',grade='retrospective input robustness,not blind/new high-precision benchmark',workers=2,
        old29_5_unchanged=True,new_training_labels=0,goal_complete=False)
    path=DEST/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['code_hashes']==p['code_hashes'] and old['model_manifest_hashes']==p['model_manifest_hashes']
    else:e.write(path,p)


def shard(i):
    task=DEST/f'tasks_{i}.json';out=DEST/f'outputs_{i}.json'
    run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase23_source_runtime.py'),'--worker',str(task),str(out)],cwd=ROOT,capture_output=True,text=True)
    (DEST/f'worker_{i}.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode();return json.loads(out.read_text())


def main():
    configs=methods();source,anchorsB=tables();freeze(configs,anchorsB);pd.concat(source,ignore_index=True).to_csv(DEST/'source_inputs.csv',index=False)
    queries=[];reference={};shards=[[],[]]
    for d in source:
        tier=d.tier.iloc[0];scenario=d.scenario.iloc[0];maxb=1 if tier=='same_specimen_prose_2' else 2;anchorlabels=['e'] if maxb==1 else anchorsB
        nodes=[dict(incorporated_cm3=float(r.incorporated_cm3),growth_T_C=525.) for r in d.itertuples()];lookup={r.sample_label:i for i,r in enumerate(d.itertuples())}
        for cfg in configs:
            gi=int(hashlib.sha256(json.dumps([cfg['context'],cfg['method'],tier,scenario]).encode()).hexdigest()[:8],16)%2
            for b in range(maxb+1):
                aids=anchorlabels[:b];anchors=[dict(input_index=lookup[a],measured_nH_cm3=float(d[d.sample_label.eq(a)].nH_cm3.iloc[0])) for a in aids]
                for r in d.itertuples():
                    key=f'q{len(queries)}';task=dict(id=key,bundle=cfg['bundle'],engine=cfg['engine'],input=dict(method=cfg['method'],dopant='Si',incorporated_cm3=float(r.incorporated_cm3),growth_T_C=525.,
                        observation_T_K=300.,source_inputs=nodes,anchors=anchors,pO2_atm=1e-5))
                    queries.append(task);shards[gi].append(task)
                    reference[key]=dict(context=cfg['context'],method=cfg['method'],engine=cfg['engine'],tier=tier,scenario=scenario,budget=b,sample_label=r.sample_label,sample_id=r.sample_id,
                        role='anchor' if r.sample_label in aids else 'query',common_query=r.sample_label not in anchorlabels,incorporated_cm3=float(r.incorporated_cm3),
                        observed_log_nH=float(np.log10(r.nH_cm3)),observed_log_mu=float(np.log10(r.muH_cm2Vs)),observed_log_sigma=float(np.log10(1.602176634e-19*r.nH_cm3*r.muH_cm2Vs)),
                        chemical_input_grade=r.chemical_input_grade)
    assert len(queries)==7320
    e.write(DEST/'tasks.json',dict(queries=queries));e.write(DEST/'parent_reference.json',reference)
    for i,q in enumerate(shards):e.write(DEST/f'tasks_{i}.json',dict(queries=q))
    e.write(DEST/'execution_started.json',dict(created_utc=e.now(),requests=len(queries),shards=[len(x) for x in shards],new_fitting=0))
    with ThreadPoolExecutor(max_workers=2) as pool:outputs=list(pool.map(shard,range(2)))
    rows=[]
    for actual in outputs:
        assert not actual['blocked_paths']
        for q in actual['rows']:
            r=dict(id=q['id'],**reference[q['id']],numeric=q['error'] is None,error=q['error']);v=q['result']
            if v is not None:
                r.update(predicted_log_nH=v['log10_nH_cm3'],predicted_log_mu=v['log10_muH_cm2Vs'],predicted_log_sigma=v['log10_sigma_S_cm'])
                assert abs(r['predicted_log_sigma']-np.log10(1.602176634e-19)-r['predicted_log_nH']-r['predicted_log_mu'])<1e-8
            rows.append(r)
    df=pd.DataFrame(rows);df.to_csv(DEST/'predictions.csv',index=False);summary=[]
    for keys,g in df[df.role.eq('query')].groupby(['context','method','tier','scenario','budget']):
        for scope in ('all_query','common_query'):
            q=g if scope=='all_query' else g[g.common_query];r=dict(zip(['context','method','tier','scenario','budget'],keys),scope=scope,queries=len(q),numeric=int(q.numeric.sum()))
            for prop in ('nH','mu','sigma'):r[prop+'_MAE']=float(abs(q['predicted_log_'+prop]-q['observed_log_'+prop]).mean()) if q.numeric.all() else None
            summary.append(r)
    e.csvwrite(DEST/'summary.csv',summary)
    e.write(DEST/'execution_complete.json',dict(created_utc=e.now(),requests=len(rows),numeric=int(df.numeric.sum()),non_numeric=int((~df.numeric).sum()),
        all_failures_retained=True,no_property_tables_in_workers=True,no_query_labels_for_shared_or_hyperlearning=True,new_fitting=0,new_training_labels=0,independent_sigma_labels=0,goal_complete=False))
    print('newsource complete',len(rows),'numeric',int(df.numeric.sum()),'retainedfailures',int((~df.numeric).sum()),flush=True)


if __name__=='__main__':main()
