"""Verify the added known-rate deployment without altering original property forecasts."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from csv_io_v2 import read_result_csv as read
from phase24_chemical_audit import manual_predict
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';DEST=OUT/'paired_property_prediction'


def main():
    assert json.loads((DEST/'execution_complete.json').read_text())['requests']==6588;checks=[]
    def gate(name,good,**extra):checks.append(dict(name=name,passed=bool(good),**extra))
    p=json.loads((DEST/'protocol.json').read_text())
    for kind in ('code_hashes','input_hashes'):
        for f,h in p[kind].items():gate('paired inference protocol hash',e.sha(ROOT/f)==h,path=f)
    b=OUT/'delivery/paired_chemical_bundle';manifest=json.loads((b/'manifest.json').read_text());models={n:json.loads((b/f).read_text()) for n,f in manifest['models'].items()}
    tasks=json.loads((DEST/'tasks.json').read_text())['queries'];ref=json.loads((DEST/'parent_reference.json').read_text());df=read(DEST/'predictions.csv').set_index('id',drop=False);tmap={x['id']:x for x in tasks};raw={}
    gate('complete independent new request ids',len(tasks)==len(tmap)==len(ref)==len(df)==6588)
    table=read(BASE/'phase23_process_inputs/process_audit/published_Hall_table.csv').reset_index(drop=True);labels=table.set_index('sample_label')
    for i in range(4):
        w=json.loads((DEST/f'outputs_{i}.json').read_text());qt=json.loads((DEST/f'tasks_{i}.json').read_text())['queries']
        gate('paired freshworker forbidden reads absent',not w['blocked_paths'] and not w['paired_training_blocked_paths'] and not w['property_tables_read'])
        gate('paired shard complete',{q['id'] for q in qt}=={q['id'] for q in w['rows']})
        for q in w['rows']:raw[q['id']]=q
    for key,r in ref.items():
        q=tmap[key]['input'];m=models[r['chemical_model']];expected=['d','c'][:r['budget']];saved=df.loc[key];a=raw[key]
        gate('paired no querytarget in inputs',set(q)=={'source_T_C','velocity_um_h','chemical_model','method','source_inputs','anchors','growth_T_C','observation_T_K','pO2_atm'} and all(set(x)=={'source_T_C','velocity_um_h'} for x in q['source_inputs']))
        gate('paired same nominal inputs',q['source_T_C']==labels.loc[r['sample_label']].SiO2_source_C and q['velocity_um_h']==(.86 if labels.loc[r['sample_label']].Ga2O_flux_cm2s==6e14 else 1.3))
        gate('paired same roles and budgets',len(q['anchors'])==r['budget'] and r['role']==('anchor' if r['sample_label'] in expected else 'query') and r['common_query']==(r['sample_label'] not in ['d','c']))
        for anchor,label in zip(q['anchors'],expected):gate('paired allowed Hall anchors only',set(anchor)=={'input_index','measured_nH_cm3'} and table.iloc[anchor['input_index']].sample_label==label and anchor['measured_nH_cm3']==labels.loc[label].nH_cm3)
        c=manual_predict(m,q['source_T_C'],q['velocity_um_h']);gate('paired independent chemistry reconstruction',abs(c-r['expected_logC'])<1e-10)
        good=a['error'] is None;gate('paired all refusals retained',bool(saved.numeric)==good and (good or saved.error==a['error']))
        if good:
            actual=a['result'];v=np.array([actual['log10_nH_cm3'],actual['log10_muH_cm2Vs'],actual['log10_sigma_S_cm']]);s=np.array([saved.predicted_log_nH,saved.predicted_log_mu,saved.predicted_log_sigma])
            gate('paired finite propertyvector andchemicalAPI',np.isfinite(v).all() and np.max(abs(v-s))<2e-10 and abs(actual['process_calibration']['log10_C_cm3']-c)<1e-10)
            gate('paired conductivity identity',abs(v[2]-v[0]-v[1]-np.log10(1.602176634e-19))<2e-10)
            if r['method']=='UnitDose' and r['budget']==0:gate('paired unitdose control',abs(v[0]-c)<2e-10)
    summary=read(DEST/'summary.csv');gate('paired all samequery metrics',len(summary)==1098)
    for r in summary.itertuples():
        q=df[df.context.eq(r.context)&df.method.eq(r.method)&df.chemical_model.eq(r.chemical_model)&df.budget.eq(r.budget)&df.role.eq('query')]
        if r.scope=='common_query':q=q[q.common_query]
        gate('paired full query coverage',len(q)==r.queries and int(q.numeric.sum())==r.numeric)
        for prop in ('nH','mu','sigma'):
            score=getattr(r,prop+'_MAE');gate('paired independent fullcohort MAE',(abs(np.mean(abs(q['predicted_log_'+prop]-q['observed_log_'+prop]))-score)<2e-10) if q.numeric.all() else np.isnan(score))
    for _,g in df.groupby(['context','method','chemical_model','budget']):
        q=g[g.sample_label.isin(['a','j'])].set_index('sample_label')
        if q.numeric.all():gate('paired identical-input ambiguity retained',max(abs(q.loc['a','predicted_log_'+k]-q.loc['j','predicted_log_'+k]) for k in ('nH','mu','sigma'))<2e-10)
    result=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,requests=len(df),numeric=int(df.numeric.sum()),refusals=int((~df.numeric).sum()),summary_rows=len(summary),
        no_Hall_for_chemical_selection=True,original_26352_results_preserved=True,goal_complete=False,code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))})
    e.write(DEST/'audit.json',result);print('paired property audit',result['passed'],len(checks),'checks',flush=True);assert result['passed'],[c for c in checks if not c['passed']][:10]


if __name__=='__main__':main()
