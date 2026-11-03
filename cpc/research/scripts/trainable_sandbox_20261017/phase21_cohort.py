"""Own-temperature source-excluded predictions with explicit RT source designs."""
from pathlib import Path
import sys,json,copy,shutil,subprocess,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import phase21_prediction as s
import phase21_export as export
import phase21_affine_controls as affine
from csv_io_v2 import read_result_csv as read
OUT=s.OUT/'cohort_properties'


def api_input(r):
    p=s.inputs(r);p['dopant']=p.pop('element');p['observation_T_K']=p.pop('Tobs_K');return p


def freeze():
    assert json.loads((s.OUT/'execution_complete.json').read_text())['variants']==14
    files=(Path(__file__),Path(__file__).parent/'phase21_runtime.py',ROOT/'src/sandbox/functional_predictor.py',ROOT/'src/sandbox/affine_predictor.py')
    inputs=(export.BUNDLE/'manifest.json',s.OUT/'prediction_protocol.json',s.OUT/'execution_complete.json',s.BASE/'episodes_v1.csv',
        s.PARENT/'cohort_properties/index.json',affine.OUT/'execution_complete.json',affine.OUT/'predictor_bundle/manifest.json')
    p=dict(created_utc=s.energy.now(),code_hashes={str(f.relative_to(ROOT)):s.energy.sha(f) for f in files},
        input_hashes={str(f.relative_to(ROOT)):s.energy.sha(f) for f in inputs},
        source='allweights sourceexcluded andunchanged;indexedRTanchorsonly;no newfit',
        design='nonguardactualRTsourceinputs;guardrequestsnotin declaredsupport areexplicitlyomitted,notcountedasqueryfailures',
        scope='sameprimaryquery/commonsets;ownT142nH/7Si/2DOI+5RTmu/2DOI+54Tmu/3Si/1DOI;sigmaderived',
        replay='RTnewAPI versus selectedcachedsourcepredictions,fullcoverageandtol2e-5dex',new_training_labels=0,goal_complete=False)
    dest=OUT/'protocol.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==p[k]
    else:s.energy.write(dest,p)


def main():
    freeze();full=json.loads((export.BUNDLE/'manifest.json').read_text());extra=json.loads((affine.OUT/'predictor_bundle/manifest.json').read_text());targets=s.parent.inputs.TARGETS.copy()
    rt=targets[targets.endpoint.eq('nH')&targets.kind.eq('RT')].set_index('sample_id')
    oldindex=json.loads((s.PARENT/'cohort_properties/index.json').read_text());queries=[];expected={};replay_ref={};index=[];excluded=[]
    for method in list(full['variants'])+list(extra['variants']):
        is_affine=method in affine.CONFIG
        folder=(affine.OUT if is_affine else s.OUT)/'variants'/method;selection=json.loads((folder/'selections.json').read_text());saved=read(folder/'predictions.csv')
        for held in s.FAMILIES:
            choices=[r for r in selection if r['held']==held];keys={r['fit_key'] for r in choices};assert len(keys)==1;key=next(iter(keys))
            parentmethod=affine.CONFIG[method][0] if is_affine else 'RelativeInventory' if method in s.FIELD_MODES else method
            old=next(i for i in oldindex if i['method']==parentmethod and i['held']==held);assert old['fit_key']==key
            src=ROOT/old['bundle']/'models'/f'{key}.json';bundle=OUT/'bundles'/s.digest([method,held]);dest=bundle/'models'/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert s.energy.sha(src)==s.energy.sha(dest)
            else:shutil.copyfile(src,dest)
            template=extra if is_affine else full;m=copy.deepcopy(template);spec=copy.deepcopy(template['variants'][method]);spec.update(model_key=key,global_reg=choices[0]['reg'],
                source_selection={str(c['budget']):dict(prior=c['selected_source']['prior'],lam=c['selected_source']['lam']) for c in choices})
            train=s.RT[s.RT.family.ne(held)];m.update(variants={method:spec},files=[dict(path=str(dest.relative_to(bundle)),sha256=s.energy.sha(dest))],energy_methods=[],
                parent_bundle_relative=os.path.relpath(export.PARENT,bundle),held_source=held,train_sources=sorted(train.family.unique()),
                chemical_support_cm3={e:[float(g.incorporated_cm3.min()),float(g.incorporated_cm3.max())] for e,g in train.groupby('element')},
                validation_grade='sourceexcludedretrospective;inputdesignconditional')
            engine='affine' if is_affine else 'functional'
            s.energy.write(bundle/'manifest.json',m);index.append(dict(method=method,held=held,bundle=str(bundle.relative_to(ROOT)),fit_key=key,engine=engine))
            for c in choices:
                budget=c['budget'];d=s.table(held,budget);roles=d.set_index('sample_id').role;common=set(s.table(held,2).query("role == 'query'").sample_id)
                nodes,ai,y,aids=s.design_inputs(d);support=[api_input(r) for r in nodes] if budget else []
                anchors=[dict(input_index=int(i),measured_nH_cm3=float(10**v)) for i,v in zip(ai,y)]
                references=saved[saved.budget.eq(budget)].set_index('sample_id')
                for t in targets[targets.family.eq(held)].itertuples():
                    if roles[t.sample_id]=='guard':excluded.append(dict(method=method,budget=budget,obs_id=t.obs_id,reason='guardinputexcludedfromdeclareddesign'));continue
                    keyq=f'q{len(queries)}';q=api_input(t);q.update(method=method,source_inputs=support,anchors=anchors)
                    queries.append(dict(id=keyq,bundle=str(bundle.resolve()),engine=engine,input=q));expected[keyq]=dict(method=method,budget=budget,family=held,
                        sample_id=t.sample_id,obs_id=t.obs_id,endpoint=t.endpoint,kind=t.kind,T_K=float(t.Tobs_K),observed=float(t.target),
                        role=roles[t.sample_id],common_query=t.sample_id in common)
                    if t.endpoint=='muH' and t.kind=='RT':expected[keyq]['paired_nH']=float(rt.loc[t.sample_id].target)
                    if t.endpoint=='nH' and t.kind=='RT':replay_ref[keyq]=dict(predicted=float(references.loc[t.sample_id].predicted) if references.loc[t.sample_id].numeric else None,
                        valid=bool(references.loc[t.sample_id].valid))
    s.energy.write(OUT/'tasks.json',dict(queries=queries));s.energy.write(OUT/'parent_reference.json',expected);s.energy.write(OUT/'index.json',index)
    s.energy.write(OUT/'excluded_guard_requests.json',excluded);output=OUT/'independent_outputs.json'
    run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase21_runtime.py'),'--worker',str(OUT/'tasks.json'),str(output)],cwd=ROOT,capture_output=True,text=True)
    (OUT/'worker.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode();actual=json.loads(output.read_text());assert not actual['blocked_paths']
    rows=[];derived=[];replay=[]
    for q in actual['rows']:
        r=dict(id=q['id'],**expected[q['id']],numeric=q['error'] is None,error=q['error']);p=q['result']
        if p is not None:
            r.update(predicted=p['log10_nH_cm3'] if r['endpoint']=='nH' else p['log10_muH_cm2Vs'],predicted_log_nH=p['log10_nH_cm3'],
                predicted_log_mu=p['log10_muH_cm2Vs'],predicted_log_sigma=p['log10_sigma_S_cm'],raw_source_coordinate=p['raw_source_coordinate'])
            assert abs(p['log10_sigma_S_cm']-np.log10(1.602176634e-19)-p['log10_nH_cm3']-p['log10_muH_cm2Vs'])<1e-9
        if q['id'] in replay_ref:
            e=replay_ref[q['id']];replay.append(dict(id=q['id'],method=r['method'],budget=r['budget'],sample_id=r['sample_id'],expected_valid=e['valid'],
                actual_valid=r['numeric'],error_dex=abs(r['predicted']-e['predicted']) if r['numeric'] and e['predicted'] is not None else None))
        rows.append(r)
        if r['endpoint']=='muH' and r['kind']=='RT':
            extra=dict(r);extra.update(endpoint='sigma_derived',observed=float(np.log10(1.602176634e-19)+r['paired_nH']+r['observed']))
            if p is not None:extra['predicted']=p['log10_sigma_S_cm']
            derived.append(extra)
    d=pd.DataFrame(rows+derived);d.to_csv(OUT/'predictions.csv',index=False);by=[]
    for scope in ('all_eligible','query','common_query'):
        z=d if scope=='all_eligible' else d[d.role.eq('query')] if scope=='query' else d[d.common_query]
        for keys,g in z.groupby(['method','budget','endpoint','kind','family','sample_id']):
            r=dict(zip(['method','budget','endpoint','kind','family','sample_id'],keys));r.update(scope=scope,reads=len(g),numeric=int(g.numeric.sum()),
                MAE=float(abs(g.predicted-g.observed).mean()) if g.numeric.all() else np.nan);by.append(r)
    by=pd.DataFrame(by);by.to_csv(OUT/'by_specimen.csv',index=False);families=[];summary=[]
    for keys,g in by.groupby(['method','budget','endpoint','kind','scope','family']):
        r=dict(zip(['method','budget','endpoint','kind','scope','family'],keys));r.update(specimens=len(g),reads=int(g.reads.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else np.nan);families.append(r)
    families=pd.DataFrame(families);families.to_csv(OUT/'by_family.csv',index=False)
    for keys,g in families.groupby(['method','budget','endpoint','kind','scope']):
        r=dict(zip(['method','budget','endpoint','kind','scope'],keys));r.update(families=len(g),specimens=int(g.specimens.sum()),reads=int(g.reads.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else np.nan);summary.append(r)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);s.energy.csvwrite(OUT/'RT_replay.csv',replay)
    maximum=max(r['error_dex'] for r in replay if r['error_dex'] is not None);mismatch=[r for r in replay if r['actual_valid']!=r['expected_valid']]
    audit=dict(passed=maximum<2e-5 and not mismatch and not actual['blocked_paths'],sourceexcluded_bundles=len(index),inference_requests=len(rows),
        numeric_requests=sum(r['numeric'] for r in rows),RT_replays=len(replay),maximum_RT_replay_error=maximum,
        guard_requests_explicitly_excluded=len(excluded),RT_validity_mismatches=mismatch,failures=[r for r in rows if not r['numeric']],
        property_tables_read=False,query_labels_in_worker=False,same_RT_specimen_anchors=True,new_temperature_mu_supervision=False,independent_sigma_labels=0,goal_complete=False)
    s.energy.write(OUT/'audit.json',audit);print('cohort',len(rows),'RTreplay',len(replay),'max',maximum,'mismatch',len(mismatch),flush=True);assert audit['passed']


if __name__=='__main__':main()
