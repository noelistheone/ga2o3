"""Rebuild new-source input eligibility, label budgets, complete scores and equal-input implications."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase23_process_inputs';DEST=OUT/'source_prediction';AUDIT=DEST/'audit'
KB=8.617333262145e-5;CHARGE=1.602176634e-19;HELD='10.1063/5.0101132'


def main():
    done=json.loads((DEST/'execution_complete.json').read_text());assert done['requests']==7320
    p=json.loads((DEST/'protocol.json').read_text());checks=[]
    def gate(name,good,**extra):checks.append(dict(name=name,passed=bool(good),**extra))
    for rel in ('process_audit/protocol.json','source_prediction/protocol.json'):
        protocol=json.loads((OUT/rel).read_text())
        for key in ('code_hashes','input_hashes'):
            for name,h in protocol[key].items():gate('frozen protocol hash',e.sha(ROOT/name)==h,protocol=rel,path=name)
    cfgs={(c['context'],c['method']):c for c in p['configs']}
    gate('fixed full and source-excluded methods',len(cfgs)==61 and sum(c[0]=='full5' for c in cfgs)==34)
    for path,h in p['model_manifest_hashes'].items():
        manifest=Path(path);gate('frozen bundle manifest',e.sha(manifest)==h,path=path)
        b=json.loads(manifest.read_text())
        for item in b['files']:gate('frozen model weight',e.sha(manifest.parent/item['path'])==item['sha256'],path=item['path'])
    for c in p['configs']:
        b=json.loads((Path(c['bundle'])/'manifest.json').read_text())
        if c['context']=='exclude_related_McCandless':
            gate('related source excluded from saved bundle',b['held_source']==HELD and HELD not in b['train_sources'],method=c['method'])
        for f in (Path(c['bundle'])/'models').glob('*.json'):
            rec=json.loads(f.read_text())
            gate('newsource absent from shared labels','10.1063/5.0139622' not in rec.get('train_families',[]),method=c['method'])
            if c['context']=='exclude_related_McCandless':gate('related source absent from shared weights',HELD not in rec.get('train_families',[]),method=c['method'])
    table=read(OUT/'process_audit/published_Hall_table.csv').set_index('sample_label');inputs=read(DEST/'source_inputs.csv')
    gate('12 distinct Hall specimens',len(table)==12 and set(table.index)==set('abcdefghijkl'))
    for tier,g in inputs.groupby('tier'):
        wanted=12 if tier=='process_calibrated_12' else 2
        gate('exact input scenarios',len(g)==3*wanted and g.scenario.nunique()==3,tier=tier)
        for r in g.itertuples():
            if tier=='process_calibrated_12':
                velocity=1. if r.scenario=='nominal_rate' else (.86 if r.Ga2O_flux_cm2s==6e14 else 1.3)*(1.4 if r.scenario=='faster40pct' else 1.)
                concentration=4.6e28*np.exp(-5.3/(KB*(r.SiO2_source_C+273.15)))/(velocity*1e-4/3600.)
            else:
                pairs={'published_rounded':{'e':2e18,'k':3e19},'rounding_lower':{'e':1.5e18,'k':2.5e19},'rounding_upper':{'e':2.5e18,'k':3.5e19}}
                concentration=pairs[r.scenario][r.sample_label]
            gate('independent input conversion',abs(np.log10(r.incorporated_cm3/concentration))<1e-12,tier=tier,label=r.sample_label,scenario=r.scenario)
    tasks=json.loads((DEST/'tasks.json').read_text())['queries'];refs=json.loads((DEST/'parent_reference.json').read_text());lookup={x['id']:x for x in tasks}
    gate('unique complete requests',len(tasks)==len(refs)==len(lookup)==7320 and set(lookup)==set(refs))
    for q in tasks:
        r=refs[q['id']];x=q['input'];tier=r['tier'];expected=p['anchors_by_tier'][tier][:r['budget']]
        input_table=inputs[inputs.tier.eq(tier)&inputs.scenario.eq(r['scenario'])].reset_index(drop=True)
        gate('only declared source input columns',all(set(row)=={'incorporated_cm3','growth_T_C'} for row in x['source_inputs']),id=q['id'])
        gate('no query labels in payload',set(x)=={'method','dopant','incorporated_cm3','growth_T_C','observation_T_K','source_inputs','anchors','pO2_atm'},id=q['id'])
        gate('fixed budget and known anchors',len(x['anchors'])==r['budget'] and len(expected)==r['budget'],id=q['id'])
        for a,label in zip(x['anchors'],expected):
            i=a['input_index'];gate('allowed anchor only',set(a)=={'input_index','measured_nH_cm3'} and input_table.iloc[i].sample_label==label and
                a['measured_nH_cm3']==table.loc[label,'nH_cm3'] and x['source_inputs'][i]['incorporated_cm3']==input_table.iloc[i].incorporated_cm3,id=q['id'])
        gate('correct role and fixed common membership',r['role']==('anchor' if r['sample_label'] in expected else 'query') and
            r['common_query']==(r['sample_label'] not in p['anchors_by_tier'][tier]),id=q['id'])
        gate('only carrier anchors;fixed observation/oxygen',x['dopant']=='Si' and x['observation_T_K']==300. and x['growth_T_C']==525. and x['pO2_atm']==1e-5,id=q['id'])
    actual=[]
    for shard in (0,1):
        worker=json.loads((DEST/f'outputs_{shard}.json').read_text());selected=json.loads((DEST/f'tasks_{shard}.json').read_text())['queries']
        gate('fresh worker has no blocked reads',not worker['blocked_paths'] and not worker['property_tables_read'],shard=shard)
        gate('shard coverage',{x['id'] for x in worker['rows']}=={x['id'] for x in selected},shard=shard)
        actual+=worker['rows']
    df=read(DEST/'predictions.csv').set_index('id',drop=False);summary=read(DEST/'summary.csv');numeric=0
    gate('all result ids retained',set(df.index)==set(refs) and len(df)==7320)
    for row in actual:
        r=refs[row['id']];saved=df.loc[row['id']];good=row['error'] is None;numeric+=int(good)
        gate('success/refusal retained',bool(saved.numeric)==good and (good or saved.error==row['error']),id=row['id'])
        if good:
            v=row['result'];prediction=np.array([v['log10_nH_cm3'],v['log10_muH_cm2Vs'],v['log10_sigma_S_cm']])
            gate('finite full property vector',np.isfinite(prediction).all(),id=row['id'])
            gate('same prediction retained',max(abs(prediction-np.array([saved.predicted_log_nH,saved.predicted_log_mu,saved.predicted_log_sigma])))<2e-10,id=row['id'])
            gate('conductivity identity',abs(prediction[2]-np.log10(CHARGE)-prediction[0]-prediction[1])<2e-10,id=row['id'])
            if r['method']=='UnitDose' and r['budget']==0:gate('independent unit-dose control',abs(prediction[0]-np.log10(r['incorporated_cm3']))<1e-12,id=row['id'])
    for r in summary.itertuples():
        q=df[df.context.eq(r.context)&df.method.eq(r.method)&df.tier.eq(r.tier)&df.scenario.eq(r.scenario)&df.budget.eq(r.budget)&df.role.eq('query')]
        if r.scope=='common_query':q=q[q.common_query]
        gate('complete fixed query cohort',len(q)==r.queries and int(q.numeric.sum())==r.numeric,method=r.method,tier=r.tier,budget=r.budget,scope=r.scope)
        for prop in ('nH','mu','sigma'):
            score=getattr(r,prop+'_MAE')
            if q.numeric.all():
                val=float(np.mean(np.abs(q['predicted_log_'+prop].to_numpy()-q['observed_log_'+prop].to_numpy())))
                gate('independent complete-query MAE',np.isfinite(score) and abs(val-score)<2e-10,method=r.method,property=prop,budget=r.budget)
            else:gate('incomplete cohort not scored',np.isnan(score),method=r.method,property=prop)
    duplicate=[]
    for keys,g in df[df.tier.eq('process_calibrated_12')].groupby(['context','method','scenario','budget']):
        pair=g[g.sample_label.isin(['a','j'])].set_index('sample_label')
        for prop in ('nH','mu','sigma'):
            if not pair.numeric.all():continue
            difference=float(abs(pair.loc['a','predicted_log_'+prop]-pair.loc['j','predicted_log_'+prop]))
            delta=float(abs(pair.loc['a','observed_log_'+prop]-pair.loc['j','observed_log_'+prop]))
            err=float(abs(pair['predicted_log_'+prop]-pair['observed_log_'+prop]).sum())
            gate('identical input implies identical prediction',difference<2e-10,context=keys[0],method=keys[1],property=prop)
            gate('triangle inequality lower bound',err+2e-10>=delta,context=keys[0],method=keys[1],property=prop)
            duplicate.append(dict(context=keys[0],method=keys[1],scenario=keys[2],budget=keys[3],property=prop,maximum_prediction_difference=difference,
                observed_log_difference=delta,pair_MAE=err/2,pair_MAE_lower_bound=delta/2))
    e.csvwrite(AUDIT/'identical_input_bounds.csv',duplicate)
    comparisons=[]
    for keys,g in summary.groupby(['context','method','tier','budget','scope']):
        for prop in ('nH','mu','sigma'):
            complete=(g.numeric==g.queries).all();values=g[prop+'_MAE'].to_numpy(float)
            comparisons.append(dict(context=keys[0],method=keys[1],tier=keys[2],budget=keys[3],scope=keys[4],property=prop,
                scenarios=len(g),all_scenarios_complete=bool(complete),minimum_scenario_MAE=float(values.min()) if complete else None,
                maximum_scenario_MAE=float(values.max()) if complete else None,scenario_MAE_range=float(np.ptp(values)) if complete else None))
    e.csvwrite(AUDIT/'scenario_sensitivity.csv',comparisons)
    result=dict(passed=all(x['passed'] for x in checks),created_utc=e.now(),checks=len(checks),rows=checks,requests=len(df),numeric_requests=numeric,
        non_numeric_requests=len(df)-numeric,summary_rows=len(summary),duplicate_input_comparisons=len(duplicate),
        maximum_duplicate_input_prediction_difference=max(r['maximum_prediction_difference'] for r in duplicate),
        held_related_source=HELD,newsource_shared_training_labels=0,new_shared_fits=0,strict_actual_dose_validation_of_12=False,
        input_grade='two rounded same-specimen chemistry labels;12 process calibration proxies separate',
        auxiliary_labels='12 paired mu,derivedsigma only;no new temperature curves',independent_blind_validation=False,goal_complete=False,
        code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))})
    e.write(AUDIT/'contract_audit.json',result)
    print('newsource audit',result['passed'],'checks',len(checks),'numeric',numeric,'retainedrefusals',len(df)-numeric,flush=True)
    assert result['passed'],[q for q in checks if not q['passed']][:20]


if __name__=='__main__':main()
