"""Own-temperature full-chain scoring under the primary specimen budget."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
from phase16_transport_scope import moments
import phase18_audit as audit
import phase2_energy as energy
study=audit.study;OUT=study.OUT/'property_evaluation'


def main():
    runtime=study.OUT/'research_runtime_v2';rt=json.loads((runtime/'verification.json').read_text());assert rt['passed']
    protocol=dict(code_sha256=energy.sha(Path(__file__)),input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in
        (runtime/'verification.json',runtime/'independent_outputs.json',runtime/'index.json',study.OUT/'temperature_anchor_protocol_audit.json')},
        primary_budget='RTspecimenspersource,no extraper-curveTanchors;ownTnH/mu/sigma',
        scope='mu5RTpairs/2DOI,54oldTreads/3/1;sigmaRTderived;14methods',transport_scope='4basephysical atbudget0+2internalsourcephysical atbudget1/2;exclude duplicate sourcezero',goal_complete=False)
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==energy.clean(protocol)
    else:energy.write(p,protocol)
    index={r['task_id']:r for r in json.loads((runtime/'index.json').read_text())};targets=study.TARGETS.set_index('obs_id')
    mu=study.TARGETS[study.TARGETS.endpoint.eq('muH')];nrt=study.PRED[study.PRED.kind.eq('RT')].set_index('sample_id')
    roles=study.core.ROLES;common=set(roles[roles.budget.eq(2)&roles.role.eq('query')].sample_id)
    common_muRT=common&set(mu[mu.kind.eq('RT')].sample_id);common_muT=common&set(mu[mu.kind.eq('curve')].sample_id)
    assert len(common_muRT)==2 and len(common_muT)==1
    rows=[];nr=[];scopes=[];errors=[]
    for r in json.loads((runtime/'independent_outputs.json').read_text())['rows']:
        assert r['error'] is None
        o=targets.loc[r['obs_id']];plan=index[r['task_id']]
        role=roles[roles.sample_id.eq(o.sample_id)&roles.budget.eq(plan['budget'])].role.iloc[0]
        meta=dict(method=plan['method'],base_method=plan['base_method'],budget=plan['budget'],family=o.family,sample_id=o.sample_id,
            obs_id=r['obs_id'],kind=o.kind,T_K=float(o.Tobs_K),role=role,source_kind=plan['source_kind'])
        if o.endpoint=='nH' and o.kind=='curve':
            nr.append(dict(**meta,observed=float(o.target),predicted=float(r['values'][0]),common=bool(o.sample_id in common and 40<=o.Tobs_K<=125)))
        if o.endpoint!='muH':continue
        rows.append(dict(**meta,observed_log_mu=float(o.target),predicted_log_mu=float(r['transport_values'][0]),
            observed_log_nH=float(nrt.loc[o.sample_id,'target']) if o.kind=='RT' else None,predicted_log_nH=float(r['values'][0]),
            observed_log_sigma=float(np.log10(study.transport.E)+nrt.loc[o.sample_id,'target']+o.target) if o.kind=='RT' else None,
            predicted_log_sigma=float(r['transport_values'][1]),sigma_is_derived=True,observed_query_nH_input=False,
            same_physical_state_for_output=r['diagnostics']['same_physical_state_for_output']))
        physical='empirical' not in study.STRUCTURES[plan['base_method']]
        inspect_state=physical and ((plan['method'] in study.STRUCTURES and plan['budget']==0) or (plan['method'] in audit.source.VARIANTS and plan['budget'] in (1,2)))
        if inspect_state:
            d=r['diagnostics'];v=moments(d['free_n_cm3'],10**r['values'][2],float(o.Tobs_K))
            e=abs(v['reconstructed_log_muH']-r['transport_values'][0]);errors.append(e);assert e<1e-6
            scopes.append(dict(**meta,predicted_log_mu=r['transport_values'][0],native_fraction=d['max_native_parent_fraction'],
                bound_equilibrium_response_over_free_screen=d['bound_equilibrium_response_over_free_screen'],foreign_atom_relative_error=d['foreign_atom_relative_error'],**v))
    data=pd.DataFrame(rows);nd=pd.DataFrame(nr);sd=pd.DataFrame(scopes)
    assert len(data)==2478 and len(nd)==5964 and len(sd)==472
    summaries=[];by=[]
    for (m,b,k),g in data[data.role.eq('query')].groupby(['method','budget','kind']):
        for scope in ('all_query','common_RT2' if k=='RT' else 'common_curve24'):
            s=g if scope=='all_query' else g[g.sample_id.isin(common_muRT if k=='RT' else common_muT)]
            v=s.assign(error=abs(s.predicted_log_mu-s.observed_log_mu)).groupby(['family','sample_id']).error.mean().groupby('family').mean()
            summaries.append(dict(method=m,budget=b,kind=k,scope=scope,MAE=float(v.mean()),queries=len(s),specimens=s.sample_id.nunique(),sources=s.family.nunique()))
            by += [dict(method=m,budget=b,kind=k,scope=scope,family=f,MAE=float(e)) for f,e in v.items()]
    nt=[]
    for (m,b),g in nd[nd.role.eq('query')].groupby(['method','budget']):
        for scope in ('all_query','common_specimens_T40_125'):
            s=g if scope=='all_query' else g[g.common.eq(True)]
            v=s.assign(error=abs(s.predicted-s.observed)).groupby(['family','sample_id']).error.mean().groupby('family').mean()
            nt.append(dict(method=m,budget=b,scope=scope,MAE=float(v.mean()),queries=len(s),specimens=s.sample_id.nunique(),sources=s.family.nunique(),anchor_protocol='RTspecimens'))
    full=data[data.role.eq('query')&data.kind.eq('RT')];assert len(full)==154
    fb=[]
    for (m,b,f),g in full.groupby(['method','budget','family']):
        v=dict(method=m,budget=b,family=f,queries=len(g))
        for prop in ('nH','mu','sigma'):v[prop+'_MAE']=float(abs(g['predicted_log_'+prop]-g['observed_log_'+prop]).mean())
        fb.append(v)
    fb=pd.DataFrame(fb)
    for name,frame in [('predictions',data),('summary',pd.DataFrame(summaries)),('by_source',pd.DataFrame(by)),('RT_specimen_budget_nT_predictions',nd),
        ('RT_specimen_budget_nT_summary',pd.DataFrame(nt)),('full_chain_predictions',full),('full_chain_by_source',fb),
        ('full_chain_summary',fb.groupby(['method','budget'])[['nH_MAE','mu_MAE','sigma_MAE']].mean().reset_index()),('transport_scope',sd)]:frame.to_csv(OUT/(name+'.csv'),index=False)
    result=dict(passed=True,ownT_mu_rows=len(data),same_RT_specimen_budget_nT_rows=len(nd),RT_fullchain_query_rows=len(full),methods=14,
        muRT_pairs=5,muRT_DOIs=2,muT_reads=54,muT_specimens=3,muT_DOIs=1,new_sigma_labels=0,new_mu_supervision=False,
        common_muRT_pairs=2,common_muT_reads=24,common_nT_reads=int(pd.DataFrame(nt).query("scope=='common_specimens_T40_125'").queries.iloc[0]),
        transport_scope_states=len(sd),majority_sigma_kl_lt1=int(sd.fraction_sigma_kell_lt1.gt(.5).sum()),
        maximum_moment_replay_error=max(errors),maximum_native_parent_fraction=float(sd.native_fraction.max()),
        maximum_foreign_atom_relative_error=float(sd.foreign_atom_relative_error.max()),maximum_bound_response_over_free_screen=float(sd.bound_equilibrium_response_over_free_screen.max()),
        removed_rows=0,material_Born_validity_established=False,goal_complete=False)
    energy.write(OUT/'contract_audit.json',result);print(json.dumps(result,indent=2));print(pd.DataFrame(summaries).query("budget==0 and scope=='all_query'")[['method','kind','MAE']].to_string(index=False))


if __name__=='__main__':main()
