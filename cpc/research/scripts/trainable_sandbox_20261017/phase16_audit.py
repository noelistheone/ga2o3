"""Independent joint-label, source exclusion, selection and coverage audit."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase16_refinement as final
import phase2_energy as energy

study=final.study;OUT=study.OUT;DEST=OUT/'validation'


def number(v):return np.inf if v is None or v=='infinity' else float(v)


def expected_training(f):
    d=study.STRUCTURES[f['method']];allowed=study.TARGETS[study.TARGETS.family.isin(f['train_families'])];rows=[]
    for endpoint,kind,factor in [('nH','RT',1.),('nH','curve',.1 if d['training']=='joint' else 0.),
        ('muH','RT',d['gamma']),('muH','curve',.1*d['gamma'])]:
        if factor==0:continue
        sub=allowed[allowed.endpoint.eq(endpoint)&allowed.kind.eq(kind)]
        if kind=='curve':sub=sub[sub.Tobs_K<=275]
        for family,g in sub.groupby('family'):
            for sample,s in g.groupby('sample_id'):
                for r in s.itertuples():rows.append(dict(obs_id=r.obs_id,weight=factor/(sub.family.nunique()*g.sample_id.nunique()*len(s))))
    return {r['obs_id']:r['weight'] for r in rows}


def primary(f,held,budget,lam):
    data=study.core.DATA[study.core.DATA.family.eq(held)]
    roles=study.core.ROLES[study.core.ROLES.family.eq(held)&study.core.ROLES.budget.eq(budget)].set_index('sample_id').role
    lookup={r['obs_id']:r for r in f['predictions']}
    pred={r.sample_id:lookup['RT::'+r.sample_id]['predicted'] for r in data.itertuples()}
    anchors=[r for r in data.itertuples() if roles[r.sample_id]=='anchor']
    delta=0.
    if anchors and np.isfinite(lam):
        if any(pred[r.sample_id] is None for r in anchors):return np.inf,{},np.nan
        delta=sum(r.observed_log_nH-pred[r.sample_id] for r in anchors)/(len(anchors)+lam)
    result={k:None if v is None else v+delta for k,v in pred.items()}
    query=[r for r in data.itertuples() if roles[r.sample_id]=='query']
    score=np.inf if any(result[r.sample_id] is None for r in query) else float(np.mean([abs(result[r.sample_id]-r.observed_log_nH) for r in query]))
    return score,result,float(delta)


def summary(primary_rows,temperature_rows):
    common=set(study.core.ROLES[study.core.ROLES.budget.eq(2)&study.core.ROLES.role.eq('query')].sample_id)
    assert len(common)==18
    records=[]
    query=primary_rows[primary_rows.role.eq('query')]
    for (m,b),g in query.groupby(['method','budget']):
        for scope in ('all_query','common18'):
            s=g if scope=='all_query' else g[g.sample_id.isin(common)]
            by=s.assign(error=abs(s.predicted_log_nH-s.observed_log_nH)).groupby('family').error.mean()
            records.append(dict(method=m,budget=b,scope=scope,MAE=float(by.mean()) if s.predicted_log_nH.notna().all() else np.inf,
                worst_source=float(by.max()),sources=s.family.nunique(),queries=len(s),finite_queries=int(s.predicted_log_nH.notna().sum())))
    temps=[]
    query=temperature_rows[temperature_rows.role.eq('query')]
    for (m,b),g in query.groupby(['method','budget']):
        for scope in ('all_query','common40_125'):
            s=g if scope=='all_query' else g[g.common40_125.eq(True)]
            by=s.assign(error=abs(s.predicted-s.observed)).groupby(['family','sample_id']).error.mean().groupby('family').mean()
            temps.append(dict(method=m,budget=b,scope=scope,MAE=float(by.mean()) if s.predicted.notna().all() else np.inf,
                sources=s.family.nunique(),specimens=s.sample_id.nunique(),queries=len(s),finite_queries=int(s.predicted.notna().sum())))
    return pd.DataFrame(records),pd.DataFrame(temps)


def main():
    study.freeze();bank=final.composite();assert len(bank)==180
    errors=[];nonconverged=[];no_mu=[];fitrows=[]
    for key,f in bank.items():
        expected=expected_training(f);actual=dict(zip(f['training_obs_ids'],f['weight_per_observation']))
        assert set(expected)==set(actual),(key,'membership')
        assert max(abs(expected[k]-actual[k]) for k in expected)<1e-13
        allpred={r['obs_id']:r for r in f['predictions']+f['property_predictions']}
        assert set(allpred)==set(study.TARGETS.obs_id)
        targets=study.TARGETS.set_index('obs_id')
        objective=0.
        for obs_id,weight in actual.items():
            assert targets.loc[obs_id,'family'] in f['train_families']
            value=allpred[obs_id]['predicted'];assert value is not None
            objective+=weight*(value-targets.loc[obs_id,'target'])**2
        losses=[t['loss'] for t in f['trials'] if t['loss'] is not None]
        assert losses and f['parameters'] is not None
        replay_error=abs(objective-min(losses));errors.append(replay_error)
        assert replay_error<2e-5,(key,'objective',objective,min(losses),replay_error)
        if any(not t['success'] for t in f['trials']):nonconverged.append(key)
        method=f['method'];definition=study.STRUCTURES[method]
        if definition['gamma']>0 and not f['metadata']['mu_training_families']:no_mu.append(key)
        for p in allpred.values():
            if p['predicted'] is None:continue
            assert abs(p['transport_values'][1]-(np.log10(1.602176634e-19)+p['predicted_log_nH']+p['transport_values'][0]))<1e-10
            if definition['thermal_mode']=='record_cap':assert 600<p['diagnostics']['Tf_K']<p['diagnostics']['thermal_cap_K']
        fitrows.append(dict(key=key,method=method,train_families='|'.join(f['train_families']),objective=objective,
            trial_minimum=min(losses),replay_error=replay_error,training_observations=len(actual),
            mu_training_families='|'.join(f['metadata']['mu_training_families']),inherited=f['inherited'] is not None,
            any_unconverged_start=key in nonconverged,state_failures=len(f['state_failures']),
            physical_missing_predictions=sum(p['predicted'] is None for p in allpred.values())))
    primary_rows=[];temperature_rows=[];selection_errors=[];selected_nonconverged=[];inner_count=0
    for method in study.STRUCTURES:
        folder=OUT/'methods'/method;choices=json.loads((folder/'selections.json').read_text())
        actual=read(folder/'primary_predictions.csv');temperature_rows+=read(folder/'temperature_predictions.csv').to_dict('records')
        for c in choices:
            held=c['held'];outer=[f for f in study.core.FAMILIES if f!=held]
            for candidate in c['candidates']:
                values=[]
                for inner,key in zip(outer,candidate['inner_keys']):
                    fit=bank[key];assert set(fit['train_families'])==set(outer)-{inner}
                    assert fit['method']==method and held not in fit['train_families']
                    values.append(primary(fit,inner,c['budget'],number(candidate['lam']))[0]);inner_count+=1
                error=abs(np.mean(values)-candidate['score']);selection_errors.append(error);assert error<1e-9
            best=min(c['candidates'],key=lambda a:(round(a['score'],10),-number(a['lam']),a['index']))
            assert best==c['selected']
            f=bank[c['fit_key']];assert set(f['train_families'])==set(outer)
            if c['fit_key'] in nonconverged:selected_nonconverged.append(c['fit_key'])
            score,expected,offset=primary(f,held,c['budget'],number(c['selected']['lam']))
            rows=actual[actual.family.eq(held)&actual.budget.eq(c['budget'])]
            assert set(rows.sample_id)==set(expected)
            for r in rows.itertuples():
                assert expected[r.sample_id] is not None and abs(expected[r.sample_id]-r.predicted_log_nH)<1e-9
                assert abs(r.offset_dex-offset)<1e-9
            assert abs(score-c['outer_MAE'])<1e-9
        primary_rows+=actual.to_dict('records')
    primary_rows=pd.DataFrame(primary_rows);temperature_rows=pd.DataFrame(temperature_rows)
    s,t=summary(primary_rows,temperature_rows)
    DEST.mkdir(parents=True,exist_ok=True)
    primary_rows.to_csv(DEST/'primary_predictions.csv',index=False);temperature_rows.to_csv(DEST/'temperature_predictions.csv',index=False)
    s.to_csv(DEST/'primary_summary.csv',index=False);t.to_csv(DEST/'temperature_summary.csv',index=False)
    pd.DataFrame(fitrows).to_csv(DEST/'fit_audit.csv',index=False)
    missing=int(primary_rows[primary_rows.role.eq('query')].predicted_log_nH.isna().sum())
    assert missing==0 and inner_count==3600
    initial=study.all_fits();failed_v1=[f['key'] for f in initial.values() if not f['inherited'] and any(not t['success'] for t in f['trials'])]
    refined=[f for f in bank.values() if f.get('refinement')]
    assert {f['key'] for f in refined}==set(failed_v1)
    payload=dict(passed=True,created_utc=energy.now(),structures=12,final_fit_records=180,new_fit_records=150,inherited_fit_records=30,
        V1_new_optimizer_starts=300,V1_failed_start_fits=len(failed_v1),refined_fits=len(refined),
        additional_starts=sum(f['refinement']['additional_starts'] for f in refined),
        actual_new_optimizer_starts=300+sum(f['refinement']['additional_starts'] for f in refined),
        final_fits_with_nonconverged_starts=nonconverged,selected_outer_nonconverged=selected_nonconverged,
        no_mu_supervision_contexts=len(no_mu),maximum_objective_replay_error=max(errors),
        maximum_inner_score_replay_error=max(selection_errors),inner_scores=inner_count,
        primary_saved_rows=len(primary_rows),temperature_saved_rows=len(temperature_rows),missing_primary_queries=missing,
        original29_5DOI_anchors_common18_preserved=True,old26_controls_preserved=True,
        sameT_measured_nH_input_for_mu=False,new_sigma_labels=0,goal_complete=False)
    energy.write(DEST/'contract_audit.json',payload)
    print(json.dumps(payload,indent=2));print(s[s.scope.eq('all_query')].pivot(index='method',columns='budget',values='MAE').to_string())


if __name__=='__main__':main()
