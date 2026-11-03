"""Post-result fairness check: same n-trained state plus one mu output offset."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import phase16_refinement as final
import phase2_energy as energy

study=final.study;OUT=study.OUT/'shared_state_head_control'


def main():
    bank=final.composite();references=[m for m,d in study.STRUCTURES.items() if d['gamma']==0]
    protocol=dict(code_sha256=energy.sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (study.OUT/'validation/contract_audit.json',
            study.OUT/'research_runtime/verification.json',study.OUT/'research_runtime/independent_outputs.json')},
        reference_methods=references,
        rationale='mu-onlycontrols share mu-label budget but do not share alln labels;isolate jointinternallearning from a one-scalar mu outputcalibration using identical n-trainedstate',
        timing='added after primaryTierA n/mu results;exploratory fairness diagnostic,not preregistered or blind',
        labels='same sourceexcluded nfit,muRT+.1muT<=275;no observedqueryn input;wholeDOI excluded',
        output='mu+=constant,sigma+=constant,n unchanged;mu is empiricaloutput,not oneupdatedphysicalstate',
        no_new_Sandbox_fits=True,goal_complete=False)
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==energy.clean(protocol)
    else:energy.write(pp,protocol)
    offsets={};fits=[]
    for key,f in bank.items():
        if f['method'] not in references:continue
        positive=next(m for m,d in study.STRUCTURES.items() if d['gamma']==1 and d['reference_method']==study.STRUCTURES[f['method']]['reference_method'] and d['thermal_mode']==study.STRUCTURES[f['method']]['thermal_mode'])
        tr=study.training_data(f['train_families'],positive);tr=tr[tr.endpoint.eq('muH')]
        lookup={p['obs_id']:p['predicted'] for p in f['property_predictions']}
        if len(tr):
            residual=np.array([r.target-lookup[r.obs_id] for r in tr.itertuples()]);w=tr.loss_weight.to_numpy(float)
            shift=float(w@residual/w.sum());assert abs(w@(shift-residual))<1e-10
        else:shift=0.
        offsets[key]=shift;fits.append(dict(method='mu_head_'+f['method'],base_method=f['method'],fit_key=key,
            shared_n_state_origin=f['composite_origin'],train_families=f['train_families'],
            mu_training_obs_ids=tr.obs_id.tolist(),mu_training_weights=tr.loss_weight.to_list(),mu_offset=shift,
            no_mu_labels=not len(tr),new_shared_parameters=0,new_output_parameters=1))
    assert len(fits)==60
    runtime=study.OUT/'research_runtime';index={p['task_id']:p for p in json.loads((runtime/'index.json').read_text())}
    choices={}
    for method in references:
        for c in json.loads((study.OUT/'methods'/method/'selections.json').read_text()):choices[method,c['held'],c['budget']]=c
    targets=study.TARGETS[study.TARGETS.endpoint.eq('muH')].set_index('obs_id');roles=study.core.ROLES
    records=[]
    for p in json.loads((runtime/'independent_outputs.json').read_text())['rows']:
        plan=index[p['task_id']];method=plan['method']
        if method not in references or p['obs_id'] not in targets.index:continue
        c=choices[method,plan['held'],plan['budget']];offset=offsets[c['fit_key']];o=targets.loc[p['obs_id']]
        assert o.family not in bank[c['fit_key']]['train_families']
        role=roles[roles.sample_id.eq(o.sample_id)&roles.budget.eq(plan['budget'])].role.iloc[0]
        records.append(dict(method='mu_head_'+method,base_method=method,budget=plan['budget'],family=o.family,sample_id=o.sample_id,
            obs_id=p['obs_id'],kind=o.kind,Tobs_K=o.Tobs_K,role=role,observed_log_mu=o.target,
            predicted_log_mu=p['transport_values'][0]+offset,predicted_log_nH=p['values'][0],
            predicted_log_sigma=p['transport_values'][1]+offset,mu_offset=offset,
            same_physical_state_for_mu=False if offset!=0 else p['same_physical_state_for_output']))
    data=pd.DataFrame(records);assert len(data)==708
    common=set(roles[roles.budget.eq(2)&roles.role.eq('query')].sample_id);summ=[]
    for (m,b,kind),g in data[data.role.eq('query')].groupby(['method','budget','kind']):
        for scope in ('all_query','common_RT2' if kind=='RT' else 'common_curve'):
            s=g if scope=='all_query' else g[g.sample_id.isin(common)]
            by=s.assign(error=abs(s.predicted_log_mu-s.observed_log_mu)).groupby(['family','sample_id']).error.mean().groupby('family').mean()
            summ.append(dict(method=m,budget=b,kind=kind,scope=scope,MAE=float(by.mean()),queries=len(s),specimens=s.sample_id.nunique(),sources=s.family.nunique()))
    OUT.mkdir(parents=True,exist_ok=True);data.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(summ).to_csv(OUT/'summary.csv',index=False)
    energy.write(OUT/'fits.json',fits);energy.write(OUT/'contract_audit.json',dict(passed=True,reference_methods=4,
        closed_form_offsets=60,predictions=708,new_shared_Sandbox_fits=0,new_output_parameters_per_fit=1,
        same_all_n_and_mu_label_budget=True,post_primary_exploratory=True,new_sigma_labels=0,goal_complete=False))
    print(pd.DataFrame(summ).query('scope=="all_query" and budget==0')[['method','kind','MAE']].to_string(index=False))


if __name__=='__main__':main()
