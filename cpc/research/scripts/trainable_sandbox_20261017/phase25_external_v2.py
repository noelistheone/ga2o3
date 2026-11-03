"""Frozen new-method transfer to the previously audited Azizie conditions.

No held-source hyperparameter selection is allowed. Approximate same-Hall
chemistry and learned process proxies are reported as separate input tasks.
"""
from pathlib import Path
import sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from sandbox.chemical_process import SavedChemicalCalibrations
import phase2_energy as e

BASE=e.BASE;OUT=BASE/'phase25_shared_process/external_Azizie_v2';BUNDLE=BASE/'phase25_shared_process/delivery/predictor_bundle'
CHEM=BASE/'phase24_chemical_process/delivery/paired_chemical_bundle'
PAPER=BASE/'phase23_process_inputs/process_audit/published_Hall_table.csv'
CONTEXTS=('DOI::full','cluster::full','cluster::without:related_AFRL_Cornell_Wright')


def source_tables():
    d=read(PAPER)
    actual=d[d.sample_label.isin(['e','k'])].sort_values('sample_label').copy()
    actual['C_input_cm3']=[2e18,3e19];actual['input_tier']='rounded_same_Hall_2'
    cal=SavedChemicalCalibrations(CHEM).get('central50::PairSelected');proxy=d.copy();values=[]
    for r in proxy.itertuples():
        v=.86 if r.Ga2O_flux_cm2s==6e14 else 1.3
        p=cal.predict(r.SiO2_source_C,v);values.append(p['C_cm3'])
    proxy['C_input_cm3']=values;proxy['input_tier']='learned_process_proxy_12'
    return [actual.reset_index(drop=True),proxy.reset_index(drop=True)]


def freeze_inputs():
    code=[Path(__file__),ROOT/'scripts/trainable_sandbox_20261017/phase25_api_runtime_v2.py',ROOT/'src/sandbox/context_property_predictor.py',ROOT/'src/sandbox/context_property_predictor_v2.py']
    refs=[PAPER,CHEM/'manifest.json',BASE/'phase23_process_inputs/process_audit/source_contract.json',BASE/'artifact_manifest_phase24.json']
    p=dict(created_utc=e.now(),purpose='frozen trained context models transferred to an already exposed DOI;no new shared/sourcehypertraining',
        contexts=CONTEXTS,chemical_preset='central50::PairSelected',
        same_Hall_chemistry='e~2e18,k~3e19 printed approximate values,not instrument CI',
        proxy='learned separateSIMS stack calibration,not same-Hall chemicaltruth;no duplicate flux division',
        atoms_history='actualO3 not equilibriumO2;effective1e-5atm scenario,Hunknown0,525Cgrowth,300Kcomparison',
        anchors={'rounded_same_Hall_2':['e'],'learned_process_proxy_12':['d','c']},
        budgets={'rounded_same_Hall_2':[0,1],'learned_process_proxy_12':[0,1,2]},
        expected_group_requests=285,expected_point_rows=2280,expected_summary_rows=570,
        actual_Hall_conditions=12,same_Hall_approximate_chemistry_pairs=2,independent_labs_verified=False,
        e_k_overlap_proxy=True,independent_sigma_labels=0,retrospective=True,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in refs})
    dest=OUT/'protocol_before_transfer.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        for k in ('code_hashes','input_hashes','contexts'):assert old[k]==e.clean(p[k]),k
    else:e.write(dest,p)
    return p


def worker(i):
    run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase25_api_runtime_v2.py'),str(OUT/f'tasks_{i}.json'),str(OUT/f'outputs_{i}.json')],
        cwd=ROOT,capture_output=True,text=True)
    (OUT/f'worker_{i}.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode()
    return json.loads((OUT/f'outputs_{i}.json').read_text())


def main():
    p=freeze_inputs();assert json.loads((BASE/'phase25_shared_process/delivery/cohort_replay_v2/audit.json').read_text())['passed']
    manifest=json.loads((BUNDLE/'manifest.json').read_text());e.write(OUT/'model_lineage.json',dict(manifest_sha256=e.sha(BUNDLE/'manifest.json'),
        shared_labels_or_source_hyperparameters_fitted=False,only_allowed_Hall_anchors_for_adaptation=True))
    tasks=[];refs={}
    for table in source_tables():
        tier=table.input_tier.iloc[0];labels=p['anchors'][tier]
        nodes=[dict(element='Si',incorporated_cm3=float(r.C_input_cm3),Tobs_K=300.,growth_T_C=525.,H_known=False,H_cm3=0.,
            carbon_known=False,carbon_cm3=0.,scenario_pO2_atm=1e-5,
            oxygen_input_kind='effective unknownO2 scenario;actualreportedO3 is not substituted') for r in table.itertuples()]
        for context in CONTEXTS:
            presets=[name for name in manifest['presets'] if name.startswith(context+'::')]
            assert len(presets)==19
            for preset in presets:
                variant=manifest['presets'][preset]['variant']
                for budget in p['budgets'][tier]:
                    aids=[int(table[table.sample_label.eq(label)].index[0]) for label in labels[:budget]]
                    anchors=[dict(input_index=j,measured_nH_cm3=float(table.iloc[j].nH_cm3)) for j in aids]
                    key='q'+str(len(tasks));tasks.append(dict(id=key,input=dict(preset=preset,inputs=nodes,anchors=anchors)))
                    refs[key]=dict(context=context,variant=variant,preset=preset,input_tier=tier,budget=budget,
                        nodes=[dict(sample_label=r.sample_label,sample_id=r.sample_id,C_input_cm3=float(r.C_input_cm3),
                            observed=float(np.log10(r.nH_cm3)),observed_log_mu=float(np.log10(r.muH_cm2Vs)),
                            role='anchor' if i in aids else 'query',common_query=r.sample_label not in labels) for i,r in enumerate(table.itertuples())])
    assert len(tasks)==285
    e.write(OUT/'parent_reference.json',refs)
    for i in range(4):e.write(OUT/f'tasks_{i}.json',dict(bundle=str(BUNDLE.resolve()),queries=tasks[i::4]))
    with ThreadPoolExecutor(max_workers=4) as pool:outputs=list(pool.map(worker,range(4)))
    rows=[]
    for packet in outputs:
        assert not packet['blocked_paths']
        for response in packet['rows']:
            ref=refs[response['id']];result=response['result']
            for i,node in enumerate(ref['nodes']):
                point=result['predictions'][i] if result is not None else dict(numeric=False,valid=False)
                entry=dict(id=response['id'],**{k:v for k,v in ref.items() if k!='nodes'},**node,
                    numeric=point['numeric'],valid=point['valid'],error=response['error'])
                if point['valid']:
                    entry.update(predicted=point['log10_nH_cm3'],predicted_log_mu=point['log10_muH_cm2Vs'],
                        predicted_log_sigma=point['log10_sigma_S_cm'],predicted_log_mu_CT=point['log10_mu_CT'],
                        predicted_log_sigma_CT=point['log10_sigma_CT'])
                rows.append(entry)
    assert len(rows)==2280;df=pd.DataFrame(rows);e.csvwrite(OUT/'predictions.csv',rows)
    summary=[];keys=['context','variant','input_tier','budget']
    for key,g in df[df.role.eq('query')].groupby(keys):
        for scope in ('all_query','common_query'):
            q=g if scope=='all_query' else g[g.common_query.eq(True)]
            entry=dict(zip(keys,key),scope=scope,queries=len(q),valid=int(q.valid.sum()),numeric=int(q.numeric.sum()))
            for label,column,observed in [('nH','predicted','observed'),('native_mu','predicted_log_mu','observed_log_mu'),
                ('CT_mu','predicted_log_mu_CT','observed_log_mu')]:
                entry[label+'_MAE']=float(np.abs(q[column]-q[observed]).mean()) if q.valid.all() else None
            observed_sigma=np.log10(1.602176634e-19)+q.observed+q.observed_log_mu
            entry['native_derived_sigma_MAE']=float(np.abs(q.predicted_log_sigma-observed_sigma).mean()) if q.valid.all() else None
            summary.append(entry)
    assert len(summary)==570;e.csvwrite(OUT/'summary.csv',summary)
    ambiguity=[]
    for key,g in df[df.input_tier.eq('learned_process_proxy_12')].groupby(['context','variant','budget']):
        a=g[g.sample_label.eq('a')].iloc[0];j=g[g.sample_label.eq('j')].iloc[0]
        if a.valid and j.valid:
            delta=float(abs(a.observed-j.observed));difference=float(abs(a.predicted-j.predicted))
            assert difference<1e-12
            ambiguity.append(dict(context=key[0],variant=key[1],budget=int(key[2]),same_input_prediction_difference=difference,
                pair_observed_log_gap=delta,all12_lower_bound=delta/12.,common10_lower_bound=delta/10.))
    e.csvwrite(OUT/'identical_input_bound.csv',ambiguity)
    e.write(OUT/'execution_complete.json',dict(created_utc=e.now(),group_requests=len(tasks),rows=len(df),numeric=int(df.numeric.sum()),valid=int(df.valid.sum()),
        summary_rows=len(summary),Hall_conditions=12,approximate_same_Hall_chemistry_pairs=2,new_shared_training=0,
        source_hyperparameter_selection_on_Azizie=False,independent_blind_or_lab_validation=False,goal_complete=False))
    print('New shared-model Azizie transfer complete',len(df),'valid',int(df.valid.sum()),flush=True)


if __name__=='__main__':main()
