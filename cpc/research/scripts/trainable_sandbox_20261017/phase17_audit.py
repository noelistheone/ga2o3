"""Reconstruct all training/selection/coverage contracts without using fit code."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase17_family_selection as selection
import phase2_energy as energy
final=selection.final;study=final.study;OUT=study.OUT;DEST=OUT/'validation'
ALL_METHODS={**final.ALL_METHODS,**{m:dict(group=members) for m,members in selection.GROUPS.items()}}


def number(v):return np.inf if v is None or v=='infinity' else float(v)


def expected_training(f):
    definition=final.ALL_METHODS[f['method']];d=study.PRED[study.PRED.family.isin(f['train_families'])];expected={}
    for kind,factor in [('RT',1.),('curve',.1 if definition['training']=='joint' else 0.)]:
        if factor==0:continue
        s=d[d.kind.eq(kind)]
        if kind=='curve':s=s[s.Tobs_K.le(275)]
        for family,g in s.groupby('family'):
            for sample,rows in g.groupby('sample_id'):
                for row in rows.itertuples():expected[row.obs_id]=factor/(s.family.nunique()*g.sample_id.nunique()*len(rows))
    return expected


def primary(f,held,budget,lam):
    d=study.core.DATA[study.core.DATA.family.eq(held)]
    roles=study.core.ROLES[study.core.ROLES.family.eq(held)&study.core.ROLES.budget.eq(budget)].set_index('sample_id').role
    look={r['obs_id']:r for r in f['predictions']};raw={r.sample_id:look['RT::'+r.sample_id]['predicted'] for r in d.itertuples()}
    anchors=[r for r in d.itertuples() if roles[r.sample_id]=='anchor'];delta=0.
    if anchors and np.isfinite(lam):
        if any(raw[r.sample_id] is None for r in anchors):return np.inf,{},np.nan
        delta=float(sum(r.observed_log_nH-raw[r.sample_id] for r in anchors)/(len(anchors)+lam))
    pred={s:None if v is None else v+delta for s,v in raw.items()};query=[r for r in d.itertuples() if roles[r.sample_id]=='query']
    score=np.inf if any(pred[r.sample_id] is None for r in query) else float(np.mean([abs(pred[r.sample_id]-r.observed_log_nH) for r in query]))
    return score,pred,delta


def summarize(primary_rows,temps):
    common=set(study.core.ROLES[study.core.ROLES.budget.eq(2)&study.core.ROLES.role.eq('query')].sample_id);assert len(common)==18
    summaries=[];by_sources=[];ts=[]
    for (method,budget),g in primary_rows[primary_rows.role.eq('query')].groupby(['method','budget']):
        for scope in ('all_query','common18'):
            s=g if scope=='all_query' else g[g.sample_id.isin(common)]
            by=s.assign(error=abs(s.predicted_log_nH-s.observed_log_nH)).groupby('family').error.mean()
            summaries.append(dict(method=method,budget=budget,scope=scope,MAE=float(by.mean()) if s.predicted_log_nH.notna().all() else np.inf,
                worst_source=float(by.max()),sources=s.family.nunique(),queries=len(s),finite_queries=int(s.predicted_log_nH.notna().sum())))
            by_sources += [dict(method=method,budget=budget,scope=scope,family=f,MAE=float(v)) for f,v in by.items()]
    for (method,budget),g in temps[temps.role.eq('query')].groupby(['method','budget']):
        for scope in ('all_query','common40_125'):
            s=g if scope=='all_query' else g[g.common40_125.eq(True)]
            by=s.assign(error=abs(s.predicted-s.observed)).groupby(['family','sample_id']).error.mean().groupby('family').mean()
            ts.append(dict(method=method,budget=budget,scope=scope,MAE=float(by.mean()) if s.predicted.notna().all() else np.inf,
                sources=s.family.nunique(),specimens=s.sample_id.nunique(),queries=len(s),finite_queries=int(s.predicted.notna().sum())))
    return pd.DataFrame(summaries),pd.DataFrame(by_sources),pd.DataFrame(ts)


def main():
    study.freeze();final.controls.freeze();selection.freeze();bank=final.composite();assert len(bank)==525
    targets=study.PRED.set_index('obs_id');fitrows=[];objective_errors=[];badstarts=[];best_bad=[];missing_outputs=0;rootfail=0
    for key,f in bank.items():
        expected=expected_training(f);actual=dict(zip(f['training_obs_ids'],f['weight_per_observation']))
        assert set(expected)==set(actual) and max(abs(expected[k]-actual[k]) for k in expected)<1e-13
        look={r['obs_id']:r for r in f['predictions']};assert set(look)==set(study.PRED.obs_id)
        objective=0.
        for obs,w in expected.items():
            assert targets.loc[obs,'family'] in f['train_families'] and look[obs]['predicted'] is not None
            objective+=w*(look[obs]['predicted']-targets.loc[obs,'target'])**2
        trials=[t for t in f['trials'] if t['loss'] is not None];assert trials
        best=min(trials,key=lambda t:t['loss']);error=abs(objective-best['loss']);objective_errors.append(error)
        assert error<2e-5,(key,'objective',error)
        if any(not t['success'] for t in f['trials']):badstarts.append(key)
        if not best['success']:best_bad.append(key)
        allrows=f['predictions']+f['property_predictions'];missing=sum(r['predicted'] is None for r in allrows);missing_outputs+=missing
        rootfail+=len(f['state_failures'])
        masses=[p['diagnostics'].get('foreign_atom_relative_error',0.) for p in allrows if p['predicted'] is not None]
        assert max(masses)<3e-8
        fitrows.append(dict(fit_key=key,method=f['method'],train_families='|'.join(f['train_families']),objective=objective,
            saved_best_objective=best['loss'],replay_error=error,training_rows=len(actual),any_unconverged=key in badstarts,
            best_unconverged=key in best_bad,missing_predictions=missing,state_failures=len(f['state_failures']),
            inherited=bool(f.get('inherited')),max_atom_relative_error=max(masses)))
    prim=[];temp=[];choices_all=[];count=0;selection_errors=[];outer_bad=[]
    for method in ALL_METHODS:
        folder=OUT/'methods'/method;choices=json.loads((folder/'selections.json').read_text());assert len(choices)==15
        rows=read(folder/'primary_predictions.csv')
        for c in choices:
            held=c['held'];train=[f for f in study.FAMILIES if f!=held]
            for candidate in c['candidates']:
                member=candidate.get('member',method);values=[]
                for inner,key in zip(train,candidate['inner_keys']):
                    f=bank[key];assert f['method']==member and set(f['train_families'])==set(train)-{inner}
                    values.append(primary(f,inner,c['budget'],number(candidate['lam']))[0]);count+=1
                err=abs(float(np.mean(values))-number(candidate['score']));selection_errors.append(err);assert err<1e-9
            selected=min(c['candidates'],key=lambda d:(round(number(d['score']),10),-number(d['lam']),d['index']))
            assert selected==c['selected']
            f=bank[c['fit_key']];assert set(f['train_families'])==set(train)
            if c['fit_key'] in best_bad:outer_bad.append(c['fit_key'])
            score,wanted,delta=primary(f,held,c['budget'],number(c['selected']['lam']))
            saved=rows[rows.family.eq(held)&rows.budget.eq(c['budget'])];assert set(saved.sample_id)==set(wanted)
            for r in saved.itertuples():
                assert wanted[r.sample_id] is not None and abs(wanted[r.sample_id]-r.predicted_log_nH)<1e-8
                assert abs(r.offset_dex-delta)<1e-8
            assert abs(score-c['outer_MAE'])<1e-9
        choices_all+=choices;prim+=rows.to_dict('records');temp+=read(folder/'temperature_predictions.csv').to_dict('records')
    assert count==15900
    prim=pd.DataFrame(prim);temp=pd.DataFrame(temp);s,by,t=summarize(prim,temp);DEST.mkdir(exist_ok=True)
    prim.to_csv(DEST/'primary_predictions.csv',index=False);temp.to_csv(DEST/'temperature_predictions.csv',index=False)
    s.to_csv(DEST/'primary_summary.csv',index=False);by.to_csv(DEST/'primary_by_source.csv',index=False);t.to_csv(DEST/'temperature_summary.csv',index=False)
    pd.DataFrame(fitrows).to_csv(DEST/'fit_audit.csv',index=False)
    raw=final.raw_fits();affected=[f['key'] for f in raw.values() if not f.get('inherited') and any(not q['success'] for q in f['trials'])]
    refined=[f for f in bank.values() if f.get('refinement')];assert set(affected)=={f['key'] for f in refined}
    result=dict(passed=True,created_utc=energy.now(),base_structures=17,matched_control_methods=8,selected_families=6,total_methods=31,
        fit_records=525,new_base_nonlinear_fits=225,new_matched_nonlinear_fits=90,new_convex_Ridge_fits=180,inherited_references=30,
        initial_nonlinear_optimizer_starts=630,failed_start_V1_fits=len(affected),refined_fits=len(refined),
        additional_optimizer_starts=sum(f['refinement']['additional_starts'] for f in refined),
        final_any_unconverged=badstarts,final_best_unconverged=best_bad,selected_outer_best_unconverged=outer_bad,
        state_failures=rootfail,missing_saved_predictions=missing_outputs,maximum_objective_replay_error=max(objective_errors),
        inner_scores=count,maximum_inner_score_error=max(selection_errors),RT_saved_rows=len(prim),nT_saved_rows=len(temp),
        missing_primary_queries=int(prim[prim.role.eq('query')].predicted_log_nH.isna().sum()),
        original29_5DOI_anchors_common18_preserved=True,new_mu_training_labels=False,new_sigma_labels=0,
        source_selection='wholeDOI,shape+lambda onlyinner;allmembersretained',goal_complete=False)
    energy.write(DEST/'contract_audit.json',result)
    print(json.dumps(result,indent=2));print(s[s.scope.eq('all_query')].pivot(index='method',columns='budget',values='MAE').to_string())


if __name__=='__main__':main()
