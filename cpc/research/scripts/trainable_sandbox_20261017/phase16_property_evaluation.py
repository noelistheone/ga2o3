"""Full chemical-input mu/derived-sigma evaluation including matched controls."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase16_refinement as final
import phase2_energy as energy

study=final.study;OUT=study.OUT/'property_evaluation'


def main():
    runtime=study.OUT/'research_runtime'
    assert json.loads((runtime/'verification.json').read_text())['passed']
    assert json.loads((study.OUT/'mu_controls/contract_audit.json').read_text())['passed']
    protocol=dict(code_sha256=energy.sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (runtime/'verification.json',runtime/'independent_outputs.json',
            runtime/'index.json',study.OUT/'mu_controls/selections.json',study.OUT/'prediction_protocol.json',
            energy.BASE/'dataset_v1.csv',energy.BASE/'episodes_v1.csv')},
        metrics='queryspecimens;RT DOIequal;muT withincurve thenDOIequal;fixedcommonRT2/commoncurve1 acrossbudgets',
        observed_sigma='derived only from RT nH/mu pairs;no independent conductivity or invented sameT nH for muT',
        retrospective=True,goal_complete=False)
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==energy.clean(protocol)
    else:energy.write(pp,protocol)
    mu=study.TARGETS[study.TARGETS.endpoint.eq('muH')].set_index('obs_id');roles=study.core.ROLES
    rt=study.core.PRED[study.core.PRED.kind.eq('RT')].set_index('sample_id')
    index={r['task_id']:r for r in json.loads((runtime/'index.json').read_text())};rows=[]
    def record(o,method,budget,pred,nH=None,sigma=None,same=None):
        role=roles[roles.sample_id.eq(o.sample_id)&roles.budget.eq(budget)].role.iloc[0]
        return dict(method=method,budget=budget,family=o.family,sample_id=o.sample_id,obs_id=o.name,
            kind=o.kind,Tobs_K=float(o.Tobs_K),role=role,observed_log_mu=float(o.target),predicted_log_mu=float(pred),
            predicted_log_nH=nH,predicted_log_sigma=sigma,
            observed_log_sigma=float(np.log10(1.602176634e-19)+rt.loc[o.sample_id,'target']+o.target) if o.kind=='RT' else None,
            sigma_is_derived=True,same_physical_state_for_output=same,observed_nH_feature_used=False)
    for r in json.loads((runtime/'independent_outputs.json').read_text())['rows']:
        if r['obs_id'] not in mu.index:continue
        assert r['error'] is None
        plan=index[r['task_id']];o=mu.loc[r['obs_id']]
        rows.append(record(o,plan['method'],plan['budget'],r['transport_values'][0],r['values'][0],r['transport_values'][1],r['same_physical_state_for_output']))
    model_count=len(rows);assert model_count==2124
    controls=json.loads((study.OUT/'mu_controls/selections.json').read_text())
    for c in controls:
        pred={r['obs_id']:r['predicted'] for r in c['fit']['predictions']}
        assert c['held'] not in c['fit']['train_families']
        for obs_id,o in mu[mu.family.eq(c['held'])].iterrows():
            for budget in (0,1,2):rows.append(record(o,'mu_control_'+c['kind'],budget,pred[obs_id]))
    data=pd.DataFrame(rows);assert len(data)==2832
    common=set(roles[roles.budget.eq(2)&roles.role.eq('query')].sample_id)
    common_rt=common&set(mu[mu.kind.eq('RT')].sample_id);common_T=common&set(mu[mu.kind.eq('curve')].sample_id)
    assert len(common_rt)==2 and len(common_T)==1
    summary=[];by_source=[]
    for (m,b,kind),g in data[data.role.eq('query')].groupby(['method','budget','kind']):
        for scope in ('all_query','common_RT2' if kind=='RT' else 'common_curve'):
            s=g if scope=='all_query' else g[g.sample_id.isin(common_rt if kind=='RT' else common_T)]
            by=s.assign(mu_error=abs(s.predicted_log_mu-s.observed_log_mu)).groupby(['family','sample_id']).mu_error.mean().groupby('family').mean()
            summary.append(dict(method=m,budget=b,kind=kind,scope=scope,MAE=float(by.mean()),worst_source=float(by.max()),
                specimens=s.sample_id.nunique(),sources=s.family.nunique(),queries=len(s),finite_queries=int(s.predicted_log_mu.notna().sum())))
            for f,e in by.items():by_source.append(dict(method=m,budget=b,kind=kind,scope=scope,family=f,MAE=float(e)))
    summary=pd.DataFrame(summary)
    consistency=[]
    for (m,obs_id),g in data.groupby(['method','obs_id']):
        spread=float(np.ptp(g.predicted_log_mu.to_numpy()))
        assert spread<1e-6,(m,obs_id,spread)
        consistency.append(spread)
    full=data[data.kind.eq('RT')&data.role.eq('query')&data.method.isin(study.STRUCTURES)].copy()
    assert len(full)==132
    full['observed_log_nH']=full.sample_id.map(rt.target)
    fs=[]
    for (m,b,f),g in full.groupby(['method','budget','family']):
        row=dict(method=m,budget=b,family=f,queries=len(g))
        for prop in ('nH','mu','sigma'):row[prop+'_MAE']=float(abs(g['predicted_log_'+prop]-g['observed_log_'+prop]).mean())
        fs.append(row)
    fs=pd.DataFrame(fs)
    data.to_csv(OUT/'predictions.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(by_source).to_csv(OUT/'by_source.csv',index=False);full.to_csv(OUT/'full_chain_predictions.csv',index=False)
    fs.to_csv(OUT/'full_chain_by_source.csv',index=False)
    fs.groupby(['method','budget'])[['nH_MAE','mu_MAE','sigma_MAE']].mean().reset_index().to_csv(OUT/'full_chain_summary.csv',index=False)
    audit=dict(passed=True,ownT_model_rows=model_count,total_model_and_control_rows=len(data),
        full_chain_query_rows=len(full),RT_mu_pairs=5,RT_mu_DOIs=2,temperature_mu_reads=54,
        temperature_mu_specimens=3,temperature_mu_DOIs=1,common_RT_specimens=2,common_temperature_specimens=1,
        maximum_mu_prediction_change_across_anchor_budgets=max(consistency),
        mu_anchor_used=False,sameT_observed_nH_feature_used=False,new_sigma_labels=0,
        crossbudget_mu_score_changes_are_query_composition=True,goal_complete=False)
    energy.write(OUT/'contract_audit.json',audit)
    print(json.dumps(audit,indent=2));print(summary[summary.scope.eq('all_query')&summary.budget.eq(0)][['method','kind','MAE']].to_string(index=False))


if __name__=='__main__':main()
