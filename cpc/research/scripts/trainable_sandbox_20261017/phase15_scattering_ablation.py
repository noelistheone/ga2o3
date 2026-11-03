"""Same-state scattering-density ablation in the explicit rH=1 controls.

Only these controls permit changing NI without changing the trained nH
operator. No Hall-moment model is silently assigned a new carrier score.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
from sandbox.coherent_fermi import CoherentMaterialProfile
from sandbox.coherent_transport import fermi_transport
import phase15_prediction as base
import phase2_energy as energy

OUT=base.OUT/'scattering_ablation'


def main():
    runtime=base.OUT/'research_runtime';temp=base.OUT/'temperature_transport'
    assert json.loads((runtime/'verification.json').read_text())['passed'] and json.loads((temp/'contract_audit.json').read_text())['passed']
    inputs=(runtime/'tasks.json',runtime/'independent_outputs.json',temp/'tasks.json',temp/'independent_outputs.json',
        base.core.prior.prior.TEMPDIR/'temperature_observations.csv',energy.BASE/'dataset_v1.csv')
    sources=(Path(__file__),ROOT/'src/sandbox/coherent_transport.py')
    protocol=dict(created_utc=energy.now(),code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in sources},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        models=['FD_original_RT','FD_binding_joint'],readout='rH1 only;changingNI doesnotchange nH training',
        proxies=['Nq2','Nabs','free_n','chemical_plus_H_plus_background','constant205'],
        information='same savedstate/screen/profile/R0=205;no newfit,mutargets only for scoring',
        limits='frozen-state diagnostic in unitHall controls;not unique microscopicattribution or retrained physicalHall ablation',goal_complete=False)
    pp=OUT/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text())
        for key in ('code_hashes','input_hashes'):assert old[key]==protocol[key],key
    else:energy.write(pp,protocol)
    mu=read(base.core.prior.prior.TEMPDIR/'temperature_observations.csv');mu=mu[mu.endpoint.eq('muH_cm2Vs')].set_index('read_id')
    paired=base.core.DATA[base.core.DATA.mobility_eligible.fillna(False)&base.core.DATA.mu_cm2Vs.gt(0)].set_index('sample_id')
    rows=[];errors=[]
    for endpoint,folder in (('RT',runtime),('curve',temp)):
        tasks={t['task_id']:t for t in json.loads((folder/'tasks.json').read_text())}
        outputs=json.loads((folder/'independent_outputs.json').read_text())['rows']
        for record in outputs:
            task=tasks[record['task_id']];bundle=json.loads(Path(task['bundle']).read_text());method=bundle.get('base_method',bundle['method'])
            if method not in ('FD_original_RT','FD_binding_joint') or bundle['source_kind']!='offset':continue
            obs_id=record['obs_id'];point=next(p['input'] for p in task['points'] if p['obs_id']==obs_id)
            if endpoint=='RT':
                sample=obs_id.removeprefix('RT::')
                if sample not in paired.index:continue
                observed=float(np.log10(paired.loc[sample,'mu_cm2Vs']));family=paired.loc[sample,'family']
            else:
                observed=float(np.log10(mu.loc[obs_id,'observed']));sample=mu.loc[obs_id,'sample_id'];family=mu.loc[obs_id,'family']
            budget=bundle['budget'];role=base.core.ROLES[base.core.ROLES.family.eq(family)&base.core.ROLES.budget.eq(budget)].set_index('sample_id').loc[sample,'role']
            if role!='query':continue
            assert record['error'] is None
            profile=CoherentMaterialProfile.from_dict(bundle['material_profile']);logn=record['physical_values'][0]
            proxies=dict(Nq2=record['values'][2],Nabs=record['values'][1],free_n=logn,
                chemical_plus_H_plus_background=float(np.log10(point['incorporated_cm3']+point['H_cm3']+10**bundle['parameters'][1])))
            for kind,logNI in proxies.items():
                p=fermi_transport(logn,logNI,point['Tobs_K'],Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,
                    mass_ratio=profile.electron_mass_ratio,extra_mobility_cm2Vs=205.,hall=False)
                prediction=float(p.values[1])
                if kind=='Nq2':errors.append(abs(prediction-record['transport_values'][0]))
                rows.append(dict(endpoint=endpoint,model=method,proxy=kind,budget=budget,family=family,sample_id=sample,obs_id=obs_id,
                    T_K=point['Tobs_K'],log_n=logn,log_NI=logNI,observed_log_mu=observed,predicted_log_mu=prediction))
            rows.append(dict(endpoint=endpoint,model=method,proxy='constant205',budget=budget,family=family,sample_id=sample,obs_id=obs_id,
                T_K=point['Tobs_K'],log_n=logn,log_NI=np.nan,observed_log_mu=observed,predicted_log_mu=float(np.log10(205.))))
    d=pd.DataFrame(rows);by=[]
    for keys,g in d.groupby(['endpoint','model','proxy','budget','family','sample_id']):
        by.append(dict(zip(['endpoint','model','proxy','budget','family','sample_id'],keys),MAE=float(abs(g.predicted_log_mu-g.observed_log_mu).mean()),queries=len(g)))
    by=pd.DataFrame(by);families=by.groupby(['endpoint','model','proxy','budget','family']).MAE.mean().reset_index()
    summary=families.groupby(['endpoint','model','proxy','budget']).MAE.mean().reset_index()
    d.to_csv(OUT/'predictions.csv',index=False);by.to_csv(OUT/'by_specimen.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False)
    assert errors and max(errors)<1e-9
    energy.write(OUT/'contract_audit.json',dict(passed=True,created_utc=energy.now(),rows=len(d),models=2,proxies=5,
        maximum_original_readout_replay_error=max(errors),new_fit_jobs=0,new_mu_labels_used=False,
        core_nH_operator_unchanged=True,only_unitHall_models=True,goal_complete=False))
    print('Scattering-density ablation',len(d),'rows;exactreadouterror',max(errors))
    print(summary[summary.budget.eq(0)].pivot(index=['endpoint','model'],columns='proxy',values='MAE').to_string())


if __name__=='__main__':main()
