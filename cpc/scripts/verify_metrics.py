"""Recompute article metrics from saved per-observation predictions; no fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'source_data'; A=ROOT/'verification'; A.mkdir(exist_ok=True)
checks=[]; claims=[]
def check(name,value,**details):
    checks.append(dict(name=name,passed=bool(value),**details))
def close(name,actual,expected,tol=2e-10,**details):
    check(name,np.isfinite(actual) and abs(actual-expected)<=tol,actual=float(actual),expected=float(expected),tolerance=tol,**details)
def read(n): return pd.read_csv(D/(n+'.csv'))
def macro(d,obs='observed',pred='predicted'):
    e=d.assign(ae=abs(d[pred]-d[obs]))
    return float(e.groupby(['family','sample_id']).ae.mean().groupby('family').mean().mean())
def claim(key,value,unit,source,selection):
    claims.append(dict(id=key,value=float(value),unit=unit,source=source,selection=selection))
def main():
    rt=read('room_temperature_predictions')
    for variant in ['FixedNHReference','LearnedNHInventory']:
        for scope in ['common20','all_query']:
            s=rt[rt.variant.eq(variant)]
            if scope=='common20':s=s[s.primary_common20.eq(True)]
            n=20 if scope=='common20' else 35
            check('room_condition_count',s.sample_id.nunique()==n,variant=variant,scope=scope)
            check('room_prediction_coverage',s.numeric.eq(True).all() and s.valid.eq(True).all(),variant=variant,scope=scope)
            v=macro(s);tab=read('room_temperature_summary')
            expected=tab[tab.cohort.eq('old_cohort_heldout')&tab.variant.eq(variant)&tab.scope.eq(scope)].iloc[0].nH_MAE
            close('room_independent_macro',v,expected,variant=variant,scope=scope)
            claim('room_'+variant+'_'+scope,v,'dex','room_temperature_predictions.csv',f'{variant}; {scope}; scenario->condition->DOI equal')
    src=read('room_temperature_source_scores').pivot(index='family',columns='variant',values='MAE')
    check('room_publications',len(src)==6)
    check('source_gain_count',int((src.LearnedNHInventory<src.FixedNHReference).sum())==3)
    ex=read('external_predictions');tab=read('external_summary')
    for variant in ex.variant.unique():
        s=ex[ex.variant.eq(variant)];check('external_three_conditions',s.sample_id.nunique()==3 and s.family.nunique()==2)
        check('external_valid_coverage',s.numeric.eq(True).all() and s.valid.eq(True).all())
        v=macro(s);expected=tab[tab.variant.eq(variant)].iloc[0].nH_MAE
        close('external_independent_macro',v,expected,variant=variant)
        claim('external_'+variant,v,'dex','external_predictions.csv',variant+'; valid schema-only replay; scenario->condition->DOI equal')
    ep=read('energy_property_predictions');es=read('energy_property_summary')
    for row in es[es.scope.eq('all_query')].itertuples():
        s=ep[ep.method.eq(row.method)&ep.budget.eq(row.budget)&ep.role.eq('query')]
        check('energy_query_count',len(s)==int(row.query_records),method=row.method,budget=int(row.budget))
        v=macro(s,obs='observed_log_nH',pred='predicted_log_nH')
        close('energy_carrier_macro',v,row.MAE,method=row.method,budget=int(row.budget))
        if row.budget==0:claim('energy_'+row.method,v,'dex','energy_property_predictions.csv',row.method+';0 carrier anchors;29 queries;DOI macro')
    tp=read('temperature_predictions');ts=read('temperature_scores');ts=ts[ts.task.eq('temperature')]
    common_sets={}
    for row in ts.itertuples():
        if row.scope not in ('common40_125','all_query'):continue
        s=tp[tp.method.eq(row.method)&tp.budget.eq(row.budget)&tp.role.eq('query')]
        if row.scope=='common40_125':s=s[s.common40_125.eq(True)]
        check('temperature_query_count',len(s)==int(row.reads),method=row.method,budget=int(row.budget),scope=row.scope)
        close('temperature_independent_macro',macro(s),row.MAE,method=row.method,budget=int(row.budget),scope=row.scope)
        if row.scope=='common40_125':
            common_sets[(row.method,row.budget)]=set(s.obs_id)
            claim('temperature_'+row.method+'_k'+str(int(row.budget)),row.MAE,'dex','temperature_predictions.csv',f'{row.method};k={int(row.budget)};80 common queries;read->specimen->DOI equal')
        elif row.scope=='all_query':
            claim('temperature_full_'+row.method+'_k'+str(int(row.budget)),row.MAE,'dex','temperature_predictions.csv',f'{row.method};k={int(row.budget)};{int(row.reads)} full queries;read->specimen->DOI equal')
    ref=next(iter(common_sets.values()))
    for key,value in common_sets.items():check('identical_80_temperature_queries',value==ref,method=key[0],budget=int(key[1]))
    simple=read('temperature_simple_anchor_predictions');s=simple[simple.method.eq('two_anchor_line_p0.0')&simple.budget.eq(2)&simple.role.eq('query')&simple.common40_125.eq(True)]
    check('anchor_only_common_query_ids',set(s.read_id)==ref)
    close('anchor_only_inverse_T_score',macro(s),.2305200770971113)
    claim('temperature_anchor_only_line_k2',macro(s),'dex','temperature_simple_anchor_predictions.csv','two_anchor_line_p0.0;k=2;80 common queries')
    result=dict(checks=len(checks),passed=sum(c['passed'] for c in checks),failed=[c for c in checks if not c['passed']],items=checks)
    (A/'metric_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(claims).to_csv(A/'recomputed_claims.csv',index=False)
    print(json.dumps({k:v for k,v in result.items() if k!='items'},indent=2))
    if result['failed']: raise SystemExit(1)
if __name__=='__main__': main()
