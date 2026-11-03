"""Source-excluded v7 replay and own-temperature property prediction.

No temperature/mobility label trains the new models. The same explicit RT
specimen anchors condition all temperatures and all property endpoints.
"""
from pathlib import Path
import sys,json,copy,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase20_prediction as s
import phase2_energy as energy

OUT=s.OUT/'cohort_properties';SOURCE=s.OUT/'source_adaptation_v2';CONTROL=s.OUT/'reference_budget_control'
BUNDLE=s.OUT/'delivery_v2/predictor_bundle_v7'


def freeze():
    paths=(Path(__file__),Path(__file__).parent/'phase20_runtime.py',ROOT/'src/sandbox/relative_predictor_v2.py')
    inputs=(BUNDLE/'manifest.json',SOURCE/'execution_complete.json',s.OUT/'prediction_protocol.json',
        s.BASE/'episodes_v1.csv',CONTROL/'validation/selections.json')
    p=dict(created_utc=energy.now(),code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        source_fit='all models exclude entirequeryDOI;frozen old29/5RT trainingonly;no newfit',
        nH_RT='29specimens;roles0/1/2;guardexcluded;common18',
        nH_temperature='142oldreads7Si specimens;ownT;explicitRTanchors,never percurve150K labels',
        mobility='5RTpairs2DOI plus54oldTreads3Si/1DOI;no mu anchors;all/query/common subsets',
        conductivity='RTderivedmatchednH/muonly;not independent labels;no fabricated sigmaT pairing',
        replay='savedselectedsourceRT predictions versusfreshprocessAPI;tol2e-6dex',
        score='fullvalidcoverage only;within specimen thenwithinDOI thenequalDOI;report counts separately',
        uncertainty='none calibrated;Born/RTA and205empiricalextra not universally validated',
        new_training_labels=0,new_measurements=0,goal_complete=False)
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==p[k]
    else:energy.write(path,p)


def main():
    freeze();full=json.loads((BUNDLE/'manifest.json').read_text());targets=s.inputs.TARGETS.copy()
    rt=targets[targets.endpoint.eq('nH')&targets.kind.eq('RT')].set_index('sample_id')
    assert len(rt)==29 and len(targets)==230
    plans={m:(SOURCE/'variants'/m) for m in full['variants'] if m!='OriginalBudget'};plans['OriginalBudget']=CONTROL/'validation'
    queries=[];expected={};rt_references={};bundles=[]
    for method,folder in plans.items():
        selections=json.loads((folder/'selections.json').read_text());saved=read(folder/'predictions.csv')
        for held in s.FAMILIES:
            choices=[c for c in selections if c['held']==held];assert len(choices)==3
            keys={c['fit_key'] for c in choices};assert len(keys)==1;key=next(iter(keys))
            path=(CONTROL if method=='OriginalBudget' else s.OUT)/'fits'/f'{key}.json';f=json.loads(path.read_text());assert held not in f['train_families']
            minimal={k:f[k] for k in ('key','method','parameters','coefficients','intercept','center','scale','offset_dex','reg','train_families',
                'exact_reference_fallback','reference_search_control','training_valid') if k in f}
            bundle=OUT/'bundles'/s.digest([method,held]);mp=bundle/'models'/f'{key}.json';energy.write(mp,minimal)
            manifest=copy.deepcopy(full);manifest['files']=[dict(path=str(mp.relative_to(bundle)),sha256=energy.sha(mp))]
            spec=copy.deepcopy(full['variants'][method]);spec.update(model_key=key,global_reg=choices[0]['global_reg'],
                source_selection={str(c['budget']):dict(prior=c['selected_source']['prior'],lam=c['selected_source']['lam']) for c in choices})
            manifest.update(variants={method:spec},energy_methods=[],held_source=held,train_sources=f['train_families'],validation_grade='sourceexcludedretrospectiveresearchbundle')
            energy.write(bundle/'manifest.json',manifest);bundles.append(dict(method=method,held=held,bundle=str(bundle.relative_to(ROOT)),fit_key=key))
            sampletargets=targets[targets.family.eq(held)]
            for c in choices:
                budget=c['budget'];roles=s.ROLES[s.ROLES.family.eq(held)&s.ROLES.budget.eq(budget)].set_index('sample_id').role
                common=set(s.ROLES[s.ROLES.family.eq(held)&s.ROLES.budget.eq(2)&s.ROLES.role.eq('query')].sample_id)
                ar=rt[rt.family.eq(held) & rt.index.isin(roles[roles.eq('anchor')].index)]
                anchors=[dict(incorporated_cm3=float(a.incorporated_cm3),measured_nH_cm3=float(10**a.target),
                    H_known=bool(a.H_known),H_cm3=float(a.H_cm3),carbon_known=bool(a.carbon_known),carbon_cm3=float(a.carbon_cm3),growth_T_C=float(a.growth_T_C)) for a in ar.itertuples()]
                assert len(anchors)==budget
                ref=saved[saved.budget.eq(budget)].set_index('sample_id')
                for t in sampletargets.itertuples():
                    tid=f'q{len(queries)}';queries.append(dict(id=tid,bundle=str(bundle.resolve()),input=dict(dopant=t.element,
                        incorporated_cm3=float(t.incorporated_cm3),method=method,observation_T_K=float(t.Tobs_K),
                        H_known=bool(t.H_known),H_cm3=float(t.H_cm3),carbon_known=bool(t.carbon_known),carbon_cm3=float(t.carbon_cm3),
                        growth_T_C=float(t.growth_T_C),anchors=anchors)))
                    expected[tid]=dict(method=method,budget=budget,family=held,sample_id=t.sample_id,obs_id=t.obs_id,
                        endpoint=t.endpoint,kind=t.kind,T_K=float(t.Tobs_K),role=roles[t.sample_id],common_query=t.sample_id in common,observed=float(t.target))
                    if t.endpoint=='muH' and t.kind=='RT':expected[tid]['paired_nH']=float(rt.loc[t.sample_id].target)
                    if t.endpoint=='nH' and t.kind=='RT':rt_references[tid]=dict(predicted=float(ref.loc[t.sample_id].predicted),valid=bool(ref.loc[t.sample_id].valid))
    energy.write(OUT/'index.json',bundles);energy.write(OUT/'tasks.json',dict(queries=queries));energy.write(OUT/'parent_reference.json',expected)
    output=OUT/'independent_outputs.json';run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase20_runtime.py'),
        '--worker',str(OUT/'tasks.json'),str(output)],cwd=ROOT,capture_output=True,text=True)
    (OUT/'worker.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode();actual=json.loads(output.read_text());assert not actual['blocked_paths']
    rows=[];replay=[];scope_rows=[]
    for q in actual['rows']:
        r=dict(id=q['id'],**expected[q['id']],numeric=q['error'] is None,error=q['error'])
        p=q['result']
        if p is not None:
            r.update(predicted=p['log10_nH_cm3'] if r['endpoint']=='nH' else p['log10_muH_cm2Vs'],
                predicted_log_nH=p['log10_nH_cm3'],predicted_log_mu=p['log10_muH_cm2Vs'],predicted_log_sigma=p['log10_sigma_S_cm'])
            dg=p['raw_Sandbox_diagnostics']
            for k in ('same_physical_state_for_output','majority_weight_kl_below_one','hall_factor'):
                if k in dg:r[k]=dg[k]
            assert abs(p['log10_sigma_S_cm']-np.log10(1.602176634e-19)-p['log10_nH_cm3']-p['log10_muH_cm2Vs'])<1e-9
        if q['id'] in rt_references:
            e=rt_references[q['id']];replay.append(dict(id=q['id'],method=r['method'],budget=r['budget'],sample_id=r['sample_id'],
                expected_valid=e['valid'],actual_valid=r['numeric'],error_dex=abs(r['predicted']-e['predicted']) if r['numeric'] else None))
        rows.append(r)
        if r['endpoint']=='muH' and r['kind']=='RT':
            sigma=dict(r);sigma.update(endpoint='sigma_derived',observed=float(np.log10(1.602176634e-19)+r['paired_nH']+r['observed']))
            if p is not None:sigma['predicted']=p['log10_sigma_S_cm']
            scope_rows.append(sigma)
    d=pd.DataFrame(rows+scope_rows);d.to_csv(OUT/'predictions.csv',index=False);by=[]
    for scope in ('all_specimens','query','common_query'):
        z=d if scope=='all_specimens' else d[d.role.eq('query')] if scope=='query' else d[d.common_query]
        for keys,g in z.groupby(['method','budget','endpoint','kind','family','sample_id']):
            entry=dict(zip(['method','budget','endpoint','kind','family','sample_id'],keys));entry.update(scope=scope,reads=len(g),numeric=int(g.numeric.sum()),
                MAE=float(abs(g.predicted-g.observed).mean()) if g.numeric.all() else np.nan)
            by.append(entry)
    by=pd.DataFrame(by);by.to_csv(OUT/'by_specimen.csv',index=False);family=[]
    for keys,g in by.groupby(['method','budget','endpoint','kind','scope','family']):
        r=dict(zip(['method','budget','endpoint','kind','scope','family'],keys));r.update(specimens=len(g),reads=int(g.reads.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else np.nan);family.append(r)
    family=pd.DataFrame(family);family.to_csv(OUT/'by_family.csv',index=False);summary=[]
    for keys,g in family.groupby(['method','budget','endpoint','kind','scope']):
        r=dict(zip(['method','budget','endpoint','kind','scope'],keys));r.update(families=len(g),specimens=int(g.specimens.sum()),reads=int(g.reads.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else np.nan);summary.append(r)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);energy.csvwrite(OUT/'rt_replay.csv',replay)
    maximum=max(r['error_dex'] for r in replay if r['error_dex'] is not None)
    audit=dict(passed=maximum<2e-6 and all(r['expected_valid']==r['actual_valid'] for r in replay) and not actual['blocked_paths'],
        sourceexcluded_bundles=len(bundles),inference_requests=len(rows),RT_replays=len(replay),maximum_RT_replay_error=maximum,
        numeric_requests=sum(r['numeric'] for r in rows),failures=[r for r in rows if not r['numeric']],
        nH_RT_specimens=29,nH_temperature_reads=142,nH_temperature_Si_specimens=7,
        mu_RT_pairs=5,mu_RT_sources=2,mu_temperature_reads=54,mu_temperature_Si_specimens=3,mu_temperature_sources=1,
        new_mu_supervision=False,new_nH_temperature_supervision=False,independent_sigma_labels=0,
        property_table_open_guard_active=True,property_tables_read=False,query_labels_in_worker=False,goal_complete=False)
    energy.write(OUT/'audit.json',audit);print('cohort completed',len(rows),'replays',len(replay),'max',maximum,'failures',len(audit['failures']),flush=True)
    assert audit['passed'],audit


if __name__=='__main__':main()
