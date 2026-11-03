"""Own-temperature full-chain properties and scoped transport diagnostics."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase17_audit as audit
import phase2_energy as energy
from phase16_transport_scope import moments

study=audit.study;OUT=study.OUT/'property_evaluation'


def main():
    runtime=study.OUT/'research_runtime_v2';assert json.loads((runtime/'verification.json').read_text())['passed']
    protocol=dict(code_sha256=energy.sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (runtime/'verification.json',runtime/'independent_outputs.json',runtime/'index.json',
            study.OUT/'prediction_protocol.json',study.OUT/'family_selection_protocol.json')},
        metrics='ownT,RTqueryspecimens,withinsamplethenDOIequal;commonRT2/commoncurve24 controlsquerycomposition',
        mu_control='directCT uses calibrated predictednH;no query observednH feature;physical bandmu remains from solvedstocks',
        scope='muRT5/2DOI,muT54reads/3sameoldspecimens/1DOI;sigma derivedRT only;no newvalidation specimens',
        transport_diagnostic='samephase16 energyresolved kl fractions,basephysicalmodels atbudget0,not universal bound',goal_complete=False)
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==energy.clean(protocol)
    else:energy.write(p,protocol)
    index={r['task_id']:r for r in json.loads((runtime/'index.json').read_text())}
    mu=study.TARGETS[study.TARGETS.endpoint.eq('muH')].set_index('obs_id');nrt=study.PRED[study.PRED.kind.eq('RT')].set_index('sample_id')
    roles=study.core.ROLES;common=set(roles[roles.budget.eq(2)&roles.role.eq('query')].sample_id);common_rt=common&set(mu[mu.kind.eq('RT')].sample_id)
    common_T=common&set(mu[mu.kind.eq('curve')].sample_id);assert len(common_rt)==2 and len(common_T)==1
    rows=[];scope_rows=[];moment_errors=[];physical_base=set(study.STRUCTURES)-{'activation_sigmoid_RT'}
    for r in json.loads((runtime/'independent_outputs.json').read_text())['rows']:
        if r['obs_id'] not in mu.index:continue
        assert r['error'] is None
        plan=index[r['task_id']];o=mu.loc[r['obs_id']];role=roles[roles.sample_id.eq(o.sample_id)&roles.budget.eq(plan['budget'])].role.iloc[0]
        row=dict(method=plan['method'],selected_member=plan['selected_member'],budget=plan['budget'],family=o.family,sample_id=o.sample_id,
            obs_id=r['obs_id'],kind=o.kind,T_K=float(o.Tobs_K),role=role,observed_log_mu=float(o.target),
            predicted_log_mu=float(r['transport_values'][0]),predicted_log_nH=float(r['values'][0]),predicted_log_sigma=float(r['transport_values'][1]),
            observed_log_nH=float(nrt.loc[o.sample_id,'target']) if o.kind=='RT' else None,
            observed_log_sigma=float(np.log10(1.602176634e-19)+nrt.loc[o.sample_id,'target']+o.target) if o.kind=='RT' else None,
            sigma_is_derived=True,observed_query_nH_input=False,new_mu_supervision=False,
            same_physical_state_for_output=r['diagnostics']['same_physical_state_for_output'])
        rows.append(row)
        if plan['budget']==0 and plan['method'] in physical_base:
            v=moments(r['diagnostics']['free_n_cm3'],10**r['values'][2],float(o.Tobs_K))
            error=abs(v['reconstructed_log_muH']-r['transport_values'][0]);moment_errors.append(error);assert error<1e-6
            scope_rows.append(dict(method=plan['method'],sample_id=o.sample_id,family=o.family,obs_id=r['obs_id'],kind=o.kind,T_K=float(o.Tobs_K),
                predicted_log_mu=r['transport_values'][0],native_fraction=r['diagnostics']['max_native_parent_fraction'],
                bound_equilibrium_response_over_free_screen=r['diagnostics']['bound_equilibrium_response_over_free_screen'],
                foreign_atom_relative_error=r['diagnostics']['foreign_atom_relative_error'],**v))
    data=pd.DataFrame(rows);assert len(data)==5487
    summaries=[];by=[]
    for (method,budget,kind),g in data[data.role.eq('query')].groupby(['method','budget','kind']):
        for scope in ('all_query','common_RT2' if kind=='RT' else 'common_curve'):
            s=g if scope=='all_query' else g[g.sample_id.isin(common_rt if kind=='RT' else common_T)]
            errors=s.assign(error=abs(s.predicted_log_mu-s.observed_log_mu)).groupby(['family','sample_id']).error.mean().groupby('family').mean()
            summaries.append(dict(method=method,budget=budget,kind=kind,scope=scope,MAE=float(errors.mean()),worst_source=float(errors.max()),
                specimens=s.sample_id.nunique(),sources=s.family.nunique(),queries=len(s),finite_queries=int(s.predicted_log_mu.notna().sum())))
            by += [dict(method=method,budget=budget,kind=kind,scope=scope,family=f,MAE=float(e)) for f,e in errors.items()]
    for method in physical_base:
        d=data[data.method.eq(method)]
        for obs,g in d.groupby('obs_id'):assert np.ptp(g.predicted_log_mu.to_numpy())<2e-6,(method,obs)
    full=data[data.kind.eq('RT')&data.role.eq('query')];assert len(full)==341
    full_by=[]
    for (method,budget,family),g in full.groupby(['method','budget','family']):
        row=dict(method=method,budget=budget,family=family,queries=len(g))
        for prop in ('nH','mu','sigma'):row[prop+'_MAE']=float(abs(g['predicted_log_'+prop]-g['observed_log_'+prop]).mean())
        full_by.append(row)
    full_by=pd.DataFrame(full_by);summaries=pd.DataFrame(summaries);scopes=pd.DataFrame(scope_rows);assert len(scopes)==944
    data.to_csv(OUT/'predictions.csv',index=False);summaries.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(by).to_csv(OUT/'by_source.csv',index=False)
    full.to_csv(OUT/'full_chain_predictions.csv',index=False);full_by.to_csv(OUT/'full_chain_by_source.csv',index=False)
    full_by.groupby(['method','budget'])[['nH_MAE','mu_MAE','sigma_MAE']].mean().reset_index().to_csv(OUT/'full_chain_summary.csv',index=False)
    scopes.to_csv(OUT/'transport_scope.csv',index=False)
    scopes.groupby(['method','kind']).agg(states=('obs_id','count'),sigma_fraction_lt1_mean=('fraction_sigma_kell_lt1','mean'),
        majority_sigma_kl_lt1=('fraction_sigma_kell_lt1',lambda s:int((s>.5).sum())),max_bound_response_over_free=('bound_equilibrium_response_over_free_screen','max')).reset_index().to_csv(OUT/'transport_scope_summary.csv',index=False)
    payload=dict(passed=True,ownT_mu_predictions=len(data),fullchain_RT_query_rows=len(full),methods=31,RT_mu_pairs=5,RT_mu_DOIs=2,
        muT_reads=54,muT_specimens=3,muT_DOIs=1,common_RT2=True,common_curve24=True,new_mu_training_labels=False,new_sigma_labels=0,
        transport_scope_states=944,majority_sigma_kl_lt1_states=int(scopes.fraction_sigma_kell_lt1.gt(.5).sum()),
        maximum_moment_replay_error=max(moment_errors),maximum_native_parent_fraction=float(scopes.native_fraction.max()),
        maximum_bound_equilibrium_response_over_free_screen=float(scopes.bound_equilibrium_response_over_free_screen.max()),
        maximum_foreign_atom_relative_error=float(scopes.foreign_atom_relative_error.max()),removed_rows=0,
        material_Born_validity_established=False,
        crossbudget_note='basephysicalmu invariant;familyselection andempiricalCTcalibratedn canchange predwithbudget,alsoquerycomposition',goal_complete=False)
    energy.write(OUT/'contract_audit.json',payload)
    print(json.dumps(payload,indent=2));print(summaries[summaries.scope.eq('all_query')&summaries.budget.eq(0)][['method','kind','MAE']].to_string(index=False))


if __name__=='__main__':main()
