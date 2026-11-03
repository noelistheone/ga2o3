"""Verify complete process-to-property execution, allowed labels, metrics and frozen references."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from csv_io_v2 import read_result_csv as read
from phase24_chemical_audit import manual_predict
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';DEST=OUT/'property_prediction';AUDIT=DEST/'audit';PREVIOUS=BASE/'phase23_process_inputs'


def main():
    assert json.loads((DEST/'execution_complete.json').read_text())['requests']==26352
    protocol=json.loads((DEST/'protocol.json').read_text());checks=[]
    def gate(name,good,**extra):checks.append(dict(name=name,passed=bool(good),**extra))
    for key in ('code_hashes','input_hashes'):
        for p,h in protocol[key].items():gate('property protocol frozen hash',e.sha(ROOT/p)==h,path=p)
    manifests={r['bundle'] for r in protocol['models']}
    for path in manifests:
        b=Path(path);m=json.loads((b/'manifest.json').read_text())
        for f in m['files']:gate('frozen property weight hash',e.sha(b/f['path'])==f['sha256'],path=str(b/f['path']))
    cb=OUT/'delivery/chemical_bundle';cm=json.loads((cb/'manifest.json').read_text());models={name:json.loads((cb/p).read_text()) for name,p in cm['models'].items()}
    for f in cm['files']:gate('frozen chemical model hash',e.sha(cb/f['path'])==f['sha256'],path=f['path'])
    table=read(PREVIOUS/'process_audit/published_Hall_table.csv').reset_index(drop=True);bylabel=table.set_index('sample_label');anchorlabels=['d','c'];refs=json.loads((DEST/'parent_reference.json').read_text())
    tasks=json.loads((DEST/'tasks.json').read_text())['queries'];tmap={x['id']:x for x in tasks};gate('unique complete declared requests',len(tasks)==len(tmap)==len(refs)==26352)
    numerical=0;raw={}
    for shard in range(4):
        w=json.loads((DEST/f'outputs_{shard}.json').read_text());t=json.loads((DEST/f'tasks_{shard}.json').read_text())['queries']
        gate('no forbidden reads in fresh worker',not w['blocked_paths'] and not w['property_tables_read'] and not w['chemical_training_tables_read'],shard=shard)
        gate('complete worker shard ids',{x['id'] for x in t}=={x['id'] for x in w['rows']},shard=shard)
        for q in w['rows']:raw[q['id']]=q
    df=read(DEST/'predictions.csv').set_index('id',drop=False);gate('all requested forecasts retained',len(df)==len(raw)==26352 and set(df.index)==set(refs))
    targetkeys={'source_T_C','velocity_um_h','chemical_model','method','source_inputs','anchors','growth_T_C','observation_T_K','pO2_atm'}
    for key,row in refs.items():
        t=tmap[key]['input'];obs=bylabel.loc[row['sample_label']];expectedlabels=anchorlabels[:row['budget']]
        gate('query contains no labels or chemicaltarget',set(t)==targetkeys and all(set(x)=={'source_T_C','velocity_um_h'} for x in t['source_inputs']),id=key)
        gate('same nominal process inputs',t['source_T_C']==obs.SiO2_source_C and t['velocity_um_h']==(.86 if obs.Ga2O_flux_cm2s==6e14 else 1.3),id=key)
        gate('same fixed Hall roles/commoncohort',len(t['anchors'])==row['budget'] and row['role']==('anchor' if row['sample_label'] in expectedlabels else 'query') and row['common_query']==(row['sample_label'] not in anchorlabels),id=key)
        for anchor,label in zip(t['anchors'],expectedlabels):
            i=anchor['input_index'];gate('only allowed measured Hall anchors',set(anchor)=={'input_index','measured_nH_cm3'} and table.iloc[i].sample_label==label and anchor['measured_nH_cm3']==bylabel.loc[label].nH_cm3,id=key)
        cal=manual_predict(models[row['chemical_model']],t['source_T_C'],t['velocity_um_h']);saved=df.loc[key];actual=raw[key]
        gate('chemical inference independently reproduced',abs(cal-row['predicted_log10_C'])<1e-10,id=key)
        good=actual['error'] is None;numerical+=int(good);gate('success/refusal kept',bool(saved.numeric)==good and (good or saved.error==actual['error']),id=key)
        if good:
            p=actual['result'];values=np.array([p['log10_nH_cm3'],p['log10_muH_cm2Vs'],p['log10_sigma_S_cm']]);stored=np.array([saved.predicted_log_nH,saved.predicted_log_mu,saved.predicted_log_sigma])
            gate('coupled runtime/parent chemical input match',abs(p['process_calibration']['log10_C_cm3']-cal)<1e-10 and abs(saved.runtime_log10_C-cal)<1e-10,id=key)
            gate('full finite same property vector',np.isfinite(values).all() and np.max(abs(values-stored))<2e-10,id=key)
            gate('conductivity algebra',abs(values[2]-values[0]-values[1]-np.log10(1.602176634e-19))<2e-10,id=key)
            if row['budget']==0 and row['method']=='UnitDose':gate('unitdose process control',abs(values[0]-cal)<2e-10,id=key)
    summary=read(DEST/'summary.csv');gate('all expected summaries',len(summary)==4392)
    for row in summary.itertuples():
        q=df[df.context.eq(row.context)&df.method.eq(row.method)&df.chemical_model.eq(row.chemical_model)&df.budget.eq(row.budget)&df.role.eq('query')]
        if row.scope=='common_query':q=q[q.common_query]
        gate('complete unfiltered query set',len(q)==row.queries and int(q.numeric.sum())==row.numeric,chemical=row.chemical_model,method=row.method,budget=row.budget)
        for prop in ('nH','mu','sigma'):
            value=getattr(row,prop+'_MAE')
            if q.numeric.all():gate('independently rebuilt complete query score',abs(np.mean(abs(q['predicted_log_'+prop]-q['observed_log_'+prop]))-value)<2e-10)
            else:gate('no metric for incomplete query coverage',np.isnan(value))
    identical=[]
    for key,q in df.groupby(['context','method','chemical_model','budget']):
        pair=q[q.sample_label.isin(['a','j'])].set_index('sample_label')
        if not pair.numeric.all():continue
        for prop in ('nH','mu','sigma'):
            diff=float(abs(pair.loc['a','predicted_log_'+prop]-pair.loc['j','predicted_log_'+prop]));delta=float(abs(pair.loc['a','observed_log_'+prop]-pair.loc['j','observed_log_'+prop]))
            gate('same process inputs remain same forecast',diff<2e-10,method=key[1],property=prop)
            gate('unresolved identical-input bound retained',float(abs(pair['predicted_log_'+prop]-pair['observed_log_'+prop]).sum())+2e-10>=delta)
            identical.append(dict(context=key[0],method=key[1],chemical_model=key[2],budget=key[3],property=prop,prediction_difference=diff))
    previous=read(PREVIOUS/'source_prediction/summary.csv');previous=previous[previous.tier.eq('process_calibrated_12')&previous.scenario.eq('regime_rates')]
    comparisons=[]
    for row in summary.itertuples():
        old=previous[previous.context.eq(row.context)&previous.method.eq(row.method)&previous.budget.eq(row.budget)&previous.scope.eq(row.scope)];gate('previous printedinput same query count',len(old)==1 and old.queries.iloc[0]==row.queries)
        for prop in ('nH','mu','sigma'):
            a=getattr(row,prop+'_MAE');b=float(old[prop+'_MAE'].iloc[0]);comparisons.append(dict(context=row.context,method=row.method,chemical_model=row.chemical_model,calibration_method=row.calibration_method,scenario=row.scenario,
                budget=row.budget,scope=row.scope,property=prop,new_MAE=a,printed_proxy_MAE=b,MAE_change=a-b if np.isfinite(a) and np.isfinite(b) else None))
    e.csvwrite(AUDIT/'input_change_comparisons.csv',comparisons);e.csvwrite(AUDIT/'identical_input_checks.csv',identical)
    result=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,requests=len(df),numeric=numerical,refusals=len(df)-numerical,
        chemical_models=12,summary_rows=len(summary),identical_input_cases=len(identical),new_property_training=0,new_Hall_training_labels=0,
        chemistry_budget='12 graphlayerconditions/2stacks/1DOI,not newindependentHall labels',sourcechemicalvalues_are_proxy=True,goal_complete=False,
        code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))})
    e.write(AUDIT/'contract_audit.json',result);print('process/property audit',result['passed'],len(checks),'checks;numeric',numerical,'refusals',len(df)-numerical,flush=True)
    assert result['passed'],[c for c in checks if not c['passed']][:20]


if __name__=='__main__':main()
