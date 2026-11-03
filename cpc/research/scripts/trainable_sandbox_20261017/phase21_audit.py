"""Complete source/coverage/compute audit, including extra affine controls."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase21_prediction as s
import phase21_affine_controls as affine
import phase21_export as export
OUT=s.OUT/'audit'


def main():
    checks=[];frames=[];selections=[];candidate_rows=[]
    protected={r['path']:r['sha256'] for r in json.loads((s.BASE/'artifact_manifest_phase20.json').read_text())['files']}
    def check(name,ok,**data):checks.append(dict(test=name,passed=bool(ok),**data))
    common=set(s.ROLES[s.ROLES.budget.eq(2)&s.ROLES.role.eq('query')].sample_id);check('fixed_common18',len(common)==18)
    for method in s.VARIANTS+list(affine.CONFIG):
        aux=method in affine.CONFIG;folder=(affine.OUT if aux else s.OUT)/'variants'/method
        d=read(folder/'predictions.csv');d['method']=method;frames.append(d);cc=json.loads((folder/'selections.json').read_text());selections+=cc
        check('outerchoices',len(cc)==15,method=method)
        for c in cc:
            held,budget=c['held'],c['budget'];expected=s.ROLES[s.ROLES.family.eq(held)&s.ROLES.budget.eq(budget)].set_index('sample_id').role
            actual=d[d.family.eq(held)&d.budget.eq(budget)].set_index('sample_id')
            check('full_role_table',set(actual.index)==set(expected.index) and actual.role.eq(expected.reindex(actual.index)).all(),method=method,held=held,budget=budget)
            epdir=affine.OUT if aux else s.OUT;e=json.loads((epdir/'episodes'/f"{c['outer_episode_key']}.json").read_text())
            check('outerexclusion',held not in e['train_families'] and set(e['train_families'])==set(s.FAMILIES)-{held},method=method,held=held,budget=budget)
            anchor_key='anchor_ids';check('only_role_anchors',set(e[anchor_key])==set(expected[expected.eq('anchor')].index),method=method,held=held,budget=budget)
            check('querylabels_not_in_fit',not e.get('query_labels_used_in_fit',e.get('query_labels_in_inference',True)),method=method,held=held,budget=budget)
            check('immutable_frozen_fit',e['fit_path'] in protected and s.energy.sha(ROOT/e['fit_path'])==protected[e['fit_path']],method=method,held=held,budget=budget)
            for cand in c['source_candidates']:
                for key in cand['inner_keys']:
                    inner=json.loads((epdir/'episodes'/f'{key}.json').read_text())
                    check('innerexclusion',held not in inner['train_families'] and inner['held']!=held and inner['held'] not in inner['train_families'],method=method,budget=budget)
            mode=method if method in s.FIELD_MODES else 'OutputSafe';results=e['results'] if aux else e['results'][mode]
            for r in results:
                q=[x for x in r['records'] if x['role']=='query'];complete=all(x['numeric'] and x['valid'] for x in q)
                candidate_rows.append(dict(method=method,family=held,budget=budget,prior=r['prior'],lam=r['lam'],queries=len(q),numeric=sum(x['numeric'] and x['valid'] for x in q),
                    MAE=float(np.mean([abs(x['predicted']-x['observed']) for x in q])) if complete else np.nan,
                    selected=r['prior']==c['selected_source']['prior'] and r['lam']==c['selected_source']['lam']))
        # Frozen shared weights mean that all zero-anchor values must replay.
        parentname=affine.CONFIG[method][0] if aux else 'RelativeInventory' if method in s.FIELD_MODES else method
        parentfolder=s.PARENT/'reference_budget_control/validation' if parentname=='OriginalBudget' else s.PARENT/'source_adaptation_v2/variants'/parentname
        old=read(parentfolder/'predictions.csv');old=old[old.budget.eq(0)].set_index('sample_id');new=d[d.budget.eq(0)].set_index('sample_id')
        delta=float(np.max(abs(new.predicted-old.predicted.reindex(new.index))))
        check('zeroanchor_replays_parent',delta<2e-6,method=method,maximum_dex=delta)
    d=pd.concat(frames,ignore_index=True);by=[];summary=[]
    for scope in ('all_query','common18'):
        q=d[d.role.eq('query')] if scope=='all_query' else d[d.sample_id.isin(common)]
        for keys,g in q.groupby(['method','budget','family']):
            r=dict(zip(['method','budget','family'],keys));ok=g.numeric&g.valid
            r.update(scope=scope,queries=len(g),numeric=int(ok.sum()),MAE=float(abs(g.predicted-g.observed).mean()) if ok.all() else np.nan);by.append(r)
    by=pd.DataFrame(by);OUT.mkdir(parents=True,exist_ok=True);by.to_csv(OUT/'by_family.csv',index=False)
    for keys,g in by.groupby(['method','budget','scope']):
        r=dict(zip(['method','budget','scope'],keys));r.update(families=len(g),queries=int(g.queries.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else np.nan,worst_family_MAE=float(g.MAE.max()) if g.MAE.notna().all() else np.nan);summary.append(r)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);s.energy.write(OUT/'all_selections.json',selections)
    cand=pd.DataFrame(candidate_rows);cand.to_csv(OUT/'candidate_by_family.csv',index=False);cand['lambda_key']=cand.lam.map(lambda x:'inf' if pd.isna(x) else str(float(x)))
    cs=[]
    for keys,g in cand.groupby(['method','budget','prior','lambda_key']):
        r=dict(zip(['method','budget','prior','lambda_key'],keys));r.update(families=len(g),queries=int(g.queries.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else np.nan);cs.append(r)
    pd.DataFrame(cs).to_csv(OUT/'candidate_summary.csv',index=False)
    episodes=[json.loads(p.read_text()) for p in (s.OUT/'episodes').glob('*.json')];check('all518groups',len(episodes)==518)
    costs=[];source_failures=[]
    for e in episodes:
        for mode,m in e['mode_metadata'].items():
            check('declared_point_budget',m['nominal_point_calls']<=32768,episode=e['key'],mode=mode,calls=m['nominal_point_calls'])
            costs.append(dict(episode_key=e['key'],mode=mode,held=e['held'],budget=e['budget'],kind=e['kind'],
                nominal_points=m['nominal_point_calls'],nominal_vectors=m['nominal_vector_calls'],failures=len(m['failures'])))
            source_failures += [dict(episode=e['key'],mode=mode,**f) for f in m['failures']]
    s.energy.csvwrite(OUT/'inference_compute.csv',costs);s.energy.write(OUT/'source_failures.json',source_failures)
    for name in ('numerical_gates.json','api_gates/verification.json','affine_controls/numerical_gates_v2.json','affine_controls/api_gates.json',
            'cohort_properties/audit.json','external_leedy/audit.json','chemical_transport_controls/gates.json','chemical_transport_controls/audit.json'):
        check('module_audit',json.loads((s.OUT/name).read_text())['passed'],module=name)
    new=json.loads((export.BUNDLE/'manifest.json').read_text());parent=json.loads((export.PARENT/'manifest.json').read_text())
    for r in new['files']:
        check('model_weights_bytepreserved',s.energy.sha(export.BUNDLE/r['path'])==s.energy.sha(export.PARENT/r['path'])==r['sha256'],path=r['path'])
    for rel in ('prediction_protocol.json','delivery/export_protocol.json','cohort_properties/protocol.json','external_leedy/protocol.json',
            'affine_controls/protocol.json','chemical_transport_controls/protocol.json'):
        value=json.loads((s.OUT/rel).read_text())
        for kind in ('code_hashes','input_hashes'):
            for p,h in value.get(kind,{}).items():check('frozen_protocol_hash',s.energy.sha(ROOT/p)==h,protocol=rel,path=p)
    cp=read(s.OUT/'cohort_properties/predictions.csv');tc=read(s.OUT/'chemical_transport_controls/predictions.csv');ext=read(s.OUT/'external_leedy/predictions.csv')
    check('transport_count',len(tc)==2*(int(cp.endpoint.eq('muH').sum())+len(ext)))
    valid=tc[tc.numeric];check('transport_Hall_inversion',valid.inversion_error.max()<2e-7,maximum_error=float(valid.inversion_error.max()))
    check('transport_sigma_identity',float(np.max(abs(valid.predicted_log_sigma-np.log10(1.602176634e-19)-valid.predicted_log_nH-valid.predicted_log_mu)))<2e-7)
    # Pinpoint the bad adaptation without hiding it behind the aggregate.
    risk=next(c for c in selections if c['variant']=='FieldCurveSafe' and c['held']=='10.1002/aelm.202300679' and c['budget']==1)
    s.energy.write(OUT/'TEOS_adaptation_case.json',dict(selection=risk,scope='negativecase retained;statevaliddoesnotmeanaccurate'))
    record=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,primary_variants=14,additional_affine_controls=3,
        primary_groups=518,RT_prediction_rows=len(d),all_primary_query_rows=int(d.role.eq('query').sum()),source_failures=len(source_failures),
        actual_forward_points=sum(e['actual_forward_points'] for e in episodes),actual_source_vectors=sum(e['actual_source_vectors'] for e in episodes),
        nominal_forward_points=sum(r['nominal_points'] for r in costs),group_seconds=sum(e['seconds'] for e in episodes),
        maximum_nominal_group_points=max(r['nominal_points'] for r in costs),new_shared_optimizer_starts=0,new_material_measurements=0,
        theory_correction='finiteinputstatechecks are not predictiveaccuracyguarantees;fixedlambda transfer can fail onTEOS',
        energy_budget='same873energiesforphysicalfamilies;directmodelsconsumezeroenergeticlabels',goal_complete=False)
    s.energy.write(OUT/'contract_audit.json',record);print('audit',record['passed'],'checks',len(checks),'points',record['actual_forward_points'],flush=True);assert record['passed'],[r for r in checks if not r['passed']]


if __name__=='__main__':main()
