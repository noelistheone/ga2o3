"""Independent training, anchor-protocol and source-adaptation reconstruction."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase18_source_adaptation_v2 as source
import phase2_energy as energy
final=source.final;study=final.study;OUT=study.OUT


def number(v):return np.inf if v is None else float(v)


def primary(f,held,budget,lam):
    d=study.PRED[study.PRED.family.eq(held)&study.PRED.kind.eq('RT')]
    roles=study.core.ROLES[study.core.ROLES.family.eq(held)&study.core.ROLES.budget.eq(budget)].set_index('sample_id').role
    look={p['obs_id']:p['predicted'] for p in f['predictions']};a=d[d.sample_id.map(roles).eq('anchor')]
    delta=float(sum(r.target-look[r.obs_id] for r in a.itertuples())/(len(a)+lam)) if len(a) and np.isfinite(lam) else 0.
    wanted={r.sample_id:look[r.obs_id]+delta for r in d.itertuples()}
    q=d[d.sample_id.map(roles).eq('query')]
    return float(np.mean([abs(wanted[r.sample_id]-r.target) for r in q.itertuples()])),wanted,delta


def summaries(rows,temps=None):
    common=set(study.core.ROLES[study.core.ROLES.budget.eq(2)&study.core.ROLES.role.eq('query')].sample_id);assert len(common)==18
    result=[];by=[];ts=[]
    for (m,b),g in rows[rows.role.eq('query')].groupby(['method','budget']):
        for scope in ('all_query','common18'):
            s=g if scope=='all_query' else g[g.sample_id.isin(common)]
            v=s.assign(error=abs(s.predicted_log_nH-s.observed_log_nH)).groupby('family').error.mean()
            result.append(dict(method=m,budget=b,scope=scope,MAE=float(v.mean()),worst_source=float(v.max()),queries=len(s),sources=s.family.nunique(),finite_queries=int(s.predicted_log_nH.notna().sum())))
            by += [dict(method=m,budget=b,scope=scope,family=f,MAE=float(e)) for f,e in v.items()]
    if temps is not None:
        for (m,b),g in temps[temps.role.eq('query')].groupby(['method','budget']):
            for scope in ('all_query','common40_125'):
                s=g if scope=='all_query' else g[g.common40_125.eq(True)]
                v=s.assign(error=abs(s.predicted-s.observed)).groupby(['family','sample_id']).error.mean().groupby('family').mean()
                ts.append(dict(method=m,budget=b,scope=scope,MAE=float(v.mean()),sources=s.family.nunique(),specimens=s.sample_id.nunique(),queries=len(s)))
    return pd.DataFrame(result),pd.DataFrame(by),pd.DataFrame(ts)


def training_weights(f):
    d=study.PRED[study.PRED.family.isin(f['train_families'])];result={}
    for kind,factor in [('RT',1.),('curve',.1 if f['definition']['training']=='joint' else 0.)]:
        if factor==0:continue
        s=d[d.kind.eq(kind)]
        if kind=='curve':s=s[s.Tobs_K.le(275)]
        for family,g in s.groupby('family'):
            for sample,rows in g.groupby('sample_id'):
                for r in rows.itertuples():result[r.obs_id]=factor/(s.family.nunique()*g.sample_id.nunique()*len(rows))
    return result


def verify_legacy_temperature(f,held,budget,lam,rows):
    lookup={r['obs_id']:r['predicted'] for r in f['predictions']};wanted=[]
    samples=study.core.prior.prior.SPECS[study.core.prior.prior.SPECS.family.eq(held)].index
    for sample in samples:
        d=study.PRED[study.PRED.sample_id.eq(sample)];anchors=[d[d.kind.eq('RT')].iloc[0]] if budget else []
        if budget==2:
            candidates=d[d.kind.eq('curve')&d.Tobs_K.between(125,175)]
            idx=abs(candidates.Tobs_K-150).sort_values(kind='stable').index[0];anchors.append(d.loc[idx])
        delta=float(sum(r.target-lookup[r.obs_id] for r in anchors)/(len(anchors)+lam)) if anchors and np.isfinite(lam) else 0.
        ids={r.obs_id for r in anchors}
        for r in d.itertuples():
            role='unavailable_reference' if r.kind=='RT' else 'query'
            if r.obs_id in ids:role='anchor'
            elif budget and abs(r.Tobs_K-300)<=25:role='guard'
            if budget==2 and abs(r.Tobs_K-anchors[1].Tobs_K)<=10 and r.obs_id not in ids:role='guard'
            actual=rows[rows.obs_id.eq(r.obs_id)&rows.budget.eq(budget)];assert len(actual)==1
            a=actual.iloc[0];assert a.role==role and abs(a.predicted-lookup[r.obs_id]-delta)<1e-8
            assert bool(a.common40_125)==bool(40<=r.Tobs_K<=125)
            wanted.append(r.obs_id)
    assert set(rows[rows.family.eq(held)&rows.budget.eq(budget)].obs_id)==set(wanted)


def base_audit():
    study.freeze();bank=final.composite();assert len(bank)==120
    fits=[];errors=[];unconverged=[];best_bad=[];state_failures=0;missing=0;identities=[]
    targets=study.PRED.set_index('obs_id')
    for key,f in bank.items():
        expect=training_weights(f);actual=dict(zip(f['training_obs_ids'],f['weight_per_observation']))
        assert set(expect)==set(actual) and max(abs(expect[k]-actual[k]) for k in expect)<1e-13
        look={r['obs_id']:r for r in f['predictions']};assert set(look)==set(study.PRED.obs_id)
        loss=sum(w*(look[o]['predicted']-targets.loc[o,'target'])**2 for o,w in expect.items())
        finite=[t for t in f['trials'] if t['loss'] is not None];best=min(finite,key=lambda t:t['loss'])
        error=abs(loss-best['loss']);errors.append(error);assert error<2e-5
        if any(not t['success'] for t in f['trials']):unconverged.append(key)
        if not best['success']:best_bad.append(key)
        allrows=f['predictions']+f['property_predictions'];missing+=sum(r['predicted'] is None for r in allrows);state_failures+=len(f['state_failures'])
        mass=[]
        for r in allrows:
            assert r['predicted'] is not None
            mass.append(r['diagnostics'].get('foreign_atom_relative_error',0.))
            identity=abs(r['transport_values'][1]-np.log10(study.transport.E)-r['predicted_log_nH']-r['transport_values'][0]);identities.append(identity);assert identity<1e-10
        assert max(mass)<3e-8
        fits.append(dict(fit_key=key,method=f['method'],train_families='|'.join(f['train_families']),objective=loss,
            objective_replay_error=error,any_unconverged=key in unconverged,best_unconverged=key in best_bad,
            inherited=bool(f.get('inherited')),max_atom_error=max(mass)))
    prim=[];temp=[];count=0;outer_bad=[]
    for method in study.STRUCTURES:
        folder=OUT/'methods'/method;choices=json.loads((folder/'selections.json').read_text());rows=read(folder/'primary_predictions.csv');ts=read(folder/'temperature_predictions.csv')
        assert len(choices)==15
        for c in choices:
            train=[f for f in study.FAMILIES if f!=c['held']]
            for candidate in c['candidates']:
                values=[]
                for inner,key in zip(train,candidate['inner_keys']):
                    f=bank[key];assert set(f['train_families'])==set(train)-{inner} and f['method']==method
                    values.append(primary(f,inner,c['budget'],number(candidate['lam']))[0]);count+=1
                assert abs(float(np.mean(values))-number(candidate['score']))<1e-9
            selected=min(c['candidates'],key=lambda p:(round(number(p['score']),10),-number(p['lam']),p['index']));assert selected==c['selected']
            f=bank[c['fit_key']];assert set(f['train_families'])==set(train)
            if f['key'] in best_bad:outer_bad.append(f['key'])
            score,wanted,delta=primary(f,c['held'],c['budget'],number(selected['lam']))
            actual=rows[rows.family.eq(c['held'])&rows.budget.eq(c['budget'])];assert set(actual.sample_id)==set(wanted)
            assert max(abs(r.predicted_log_nH-wanted[r.sample_id]) for r in actual.itertuples())<1e-8
            assert max(abs(actual.offset_dex-delta))<1e-8 and abs(score-c['outer_MAE'])<1e-9
            verify_legacy_temperature(f,c['held'],c['budget'],number(selected['lam']),ts)
        prim+=rows.to_dict('records');temp+=ts.to_dict('records')
    assert count==2400
    prim=pd.DataFrame(prim);temp=pd.DataFrame(temp);s,by,t=summaries(prim,temp);dest=OUT/'validation';dest.mkdir(exist_ok=True)
    for name,frame in [('primary_predictions',prim),('temperature_predictions',temp),('primary_summary',s),('primary_by_source',by),('temperature_summary',t),('fit_audit',pd.DataFrame(fits))]:frame.to_csv(dest/(name+'.csv'),index=False)
    raw=study.all_fits();affected=[f['key'] for f in raw.values() if not f.get('inherited') and any(not t['success'] for t in f['trials'])]
    refined=[f for f in bank.values() if f.get('refinement')];assert set(affected)=={f['key'] for f in refined}
    result=dict(passed=True,created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),methods=8,records=120,
        new_fits=105,initial_new_starts=210,inherited_references=15,initial_failed_start_fits=len(affected),refined_fits=len(refined),
        additional_starts=sum(f['refinement']['additional_starts'] for f in refined),final_any_unconverged=unconverged,
        final_best_unconverged=best_bad,selected_outer_nonconverged=outer_bad,state_failures=state_failures,missing_outputs=missing,
        maximum_objective_replay_error=max(errors),maximum_sigma_identity_error=max(identities),inner_scores=count,
        RT_saved_rows=len(prim),legacy_temperature_saved_rows=len(temp),missing_primary_queries=int(prim[prim.role.eq('query')].predicted_log_nH.isna().sum()),
        primary_protocol='RTspecimenanchors',legacy_temperature_protocol='percurveRT/150Ktemperatureanchors,notprimaryspecimenbudget',
        original29_5DOI_anchors_common18_preserved=True,new_mu_training_labels=False,new_sigma_labels=0,goal_complete=False)
    energy.write(dest/'contract_audit.json',result);print(json.dumps(result,indent=2));print(s[s.scope.eq('all_query')].pivot(index='method',columns='budget',values='MAE').to_string())


def source_audit():
    source.freeze();bank=final.composite();episodes={p.stem:json.loads(p.read_text()) for p in (source.OUT/'episodes').glob('*.json')};assert len(episodes)==300
    failures=0;query_failures=0;search_failures=0;objectives=[];boundary=0;callbacks=0
    for key,ep in episodes.items():
        f=bank[ep['fit_key']];assert ep['held'] not in f['train_families']
        assert energy.sha(ROOT/ep['fit_origin']['path'])==ep['fit_origin']['sha256']
        roles=study.core.ROLES[study.core.ROLES.family.eq(ep['held'])&study.core.ROLES.budget.eq(ep['budget'])].set_index('sample_id').role
        d=study.PRED[study.PRED.family.eq(ep['held'])&study.PRED.kind.eq('RT')];anchors=d[d.sample_id.map(roles).eq('anchor')]
        assert ep['anchor_obs_ids']==anchors.obs_id.tolist() and len(anchors)==ep['budget']
        assert np.array_equal(ep['anchor_observed'],anchors.target.to_numpy(float)) and not ep['query_labels_used_in_source_fit']
        callbacks+=ep['prediction_callbacks'];failures+=len(ep['failed_probes']);query_failures+=len(ep['query_state_failures'])
        for result in ep['results']:
            z=result['z'];lam=number(result['lam']);p=np.array(result['anchor_predictions']);y=np.array(ep['anchor_observed'])
            loss=float((p-y)@(p-y)+(0 if np.isinf(lam) else lam*z*z));error=abs(loss-result['objective']);objectives.append(error);assert error<1e-10 and abs(z)<=6+1e-12
            if np.isinf(lam):assert z==0
            boundary+=int(abs(z)>5.999);search_failures+=sum(not r['success'] for r in result['local_searches'])
            rows=pd.DataFrame(result['records']);assert set(rows.sample_id)==set(d.sample_id)
            q=rows[rows.role.eq('query')];score=float(abs(q.predicted_log_nH-q.observed_log_nH).mean())
            assert result['MAE'] is not None and abs(score-result['MAE'])<1e-10
    rows_all=[];temps_all=[];count=0;outer_bound=0;property_rows=[];max_selected_error=0.
    for variant,method in source.VARIANTS.items():
        folder=source.OUT/'methods'/variant;choices=json.loads((folder/'selections.json').read_text());rows=read(folder/'primary_predictions.csv');temps=read(folder/'temperature_predictions.csv')
        props=json.loads((folder/'property_predictions.json').read_text());assert len(choices)==15
        for c in choices:
            train=[s for s in study.FAMILIES if s!=c['held']]
            for candidate in c['candidates']:
                values=[]
                for inner,key in zip(train,candidate['inner_keys']):
                    f=bank[key];assert set(f['train_families'])==set(train)-{inner} and f['method']==method
                    if c['budget']==0:value=primary(f,inner,0,np.inf)[0]
                    else:
                        ep=next(e for e in episodes.values() if e['fit_key']==key and e['held']==inner and e['budget']==c['budget'])
                        value=source.choose(ep,number(candidate['lam']))['MAE']
                    values.append(value);count+=1
                assert abs(float(np.mean(values))-number(candidate['score']))<1e-9
            selected=min(c['candidates'],key=lambda p:(round(number(p['score']),10),-number(p['lam'])));assert selected==c['selected']
            f=bank[c['fit_key']];actual=rows[rows.family.eq(c['held'])&rows.budget.eq(c['budget'])]
            if c['budget']==0:
                score,wanted,_=primary(f,c['held'],0,np.inf);assert c['source_z']==0
            else:
                ep=episodes[c['episode_key']];r=source.choose(ep,number(selected['lam']));wanted={p['sample_id']:p['predicted_log_nH'] for p in r['records']};score=r['MAE']
                assert c['source_z']==r['z'];outer_bound+=int(abs(r['z'])>5.999)
            error=max(abs(r.predicted_log_nH-wanted[r.sample_id]) for r in actual.itertuples());max_selected_error=max(error,max_selected_error);assert error<2e-6
            assert abs(score-c['outer_MAE'])<1e-9
        rows_all+=rows.to_dict('records');temps_all+=temps.to_dict('records');property_rows+=props
    assert count==1800
    rows=pd.DataFrame(rows_all);temps=pd.DataFrame(temps_all);s,by,t=summaries(rows,temps);dest=source.OUT/'validation';dest.mkdir(exist_ok=True)
    for name,frame in [('primary_predictions',rows),('RT_specimen_budget_temperature_predictions',temps),('primary_summary',s),('primary_by_source',by),('RT_specimen_budget_temperature_summary',t)]:frame.to_csv(dest/(name+'.csv'),index=False)
    energy.write(dest/'property_predictions.json',property_rows)
    result=dict(passed=True,created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),variants=6,episodes=300,inner_scores=count,
        scalar_source_fits=2100,source_prediction_callbacks=callbacks,failed_source_probes=failures,query_state_failures=query_failures,
        local_scalar_searches_not_stopped=search_failures,all_source_coordinate_boundary_hits=boundary,outer_coordinate_boundary_hits=outer_bound,
        maximum_source_objective_replay_error=max(objectives),maximum_selected_RT_replay_error=max_selected_error,
        RT_saved_rows=len(rows),RT_specimen_budget_curve_rows=len(temps),ownT_source_output_rows=len(property_rows),
        primary='unchangedRTspecimenanchors29/5/common18',temperature_scope='sameRTspecimenbudget;separatefromlegacypercurveRT/150Kanchors',
        query_labels_in_source_fit=False,new_mu_training_labels=False,new_sigma_labels=0,goal_complete=False)
    energy.write(dest/'contract_audit.json',result);print(json.dumps(result,indent=2));print(s[s.scope.eq('all_query')].pivot(index='method',columns='budget',values='MAE').to_string())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-only',action='store_true');args=ap.parse_args()
    if args.source_only:source_audit()
    else:base_audit()


if __name__=='__main__':main()
