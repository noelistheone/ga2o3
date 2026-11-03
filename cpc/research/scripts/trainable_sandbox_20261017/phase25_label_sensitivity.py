"""Frozen source-excluded prediction sensitivity to the retained Leedy prose label.

This changes allowed anchors/scoring only. It does not replace primary labels
or perform the much larger alternative-label shared-training experiment.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'src'))
from sandbox.context_saved_model import INPUT_NAMES
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE=e.BASE/'phase25_shared_process';OUT=BASE/'Leedy_label_sensitivity';BUNDLE=BASE/'delivery/predictor_bundle'
LEEDY='10.1063/1.5047214'


def protocol():
    p=OUT/'protocol_before_sensitivity.json'
    value=dict(created_utc=e.now(),alternative=dict(target='0.025wt',nH_cm3=3.25e19,muH_cm2Vs=20.),
        scope='source-excluded shared weights/sourcehyperparameters fixed;allowed-anchor and scoring sensitivity only',
        contexts=['DOI::without:'+LEEDY,'cluster::without:related_AFRL_Cornell_Wright'],
        primary_data_replaced=False,shared_training_label_sensitivity_performed=False,
        expected_requests=114,expected_rows=456,expected_summary_rows=228,
        heldout_result_used_for_model_selection=False,retrospective=True,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in [Path(__file__),ROOT/'scripts/trainable_sandbox_20261017/phase25_api_runtime.py']},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in [BASE/'data/conditions33.csv',BASE/'data/episodes33.csv',BASE/'data/qualification.json']})
    if p.exists():
        old=json.loads(p.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==value[k]
    else:e.write(p,value)
    return value


def worker(i):
    r=subprocess.run([sys.executable,str(Path(__file__).parent/'phase25_api_runtime.py'),str(OUT/f'tasks_{i}.json'),str(OUT/f'outputs_{i}.json')],
        cwd=ROOT,capture_output=True,text=True)
    (OUT/f'worker_{i}.log').write_text(r.stdout+'\n'+r.stderr);r.check_returncode()
    return json.loads((OUT/f'outputs_{i}.json').read_text())


def main():
    p=protocol();assert json.loads((BASE/'delivery/cohort_replay/audit.json').read_text())['passed']
    manifest=json.loads((BUNDLE/'manifest.json').read_text());d=read(BASE/'data/conditions33.csv')
    d=d[d.family.eq(LEEDY)].sort_values(['incorporated_cm3','sample_id']).reset_index(drop=True)
    roles=read(BASE/'data/episodes33.csv');primary=read(BASE/'prediction/predictions.csv')
    nodes=[{k:e.clean(getattr(r,k)) for k in INPUT_NAMES} for r in d.itertuples()];tasks=[];refs={}
    for context in p['contexts']:
        scheme=context.split('::')[0]
        for preset,spec in manifest['presets'].items():
            if spec['context']!=context:continue
            for budget in (0,1,2):
                rr=roles[roles.family.eq(LEEDY)&roles.budget.eq(budget)].set_index('sample_id').role
                anchors=[];points=[]
                for i,r in enumerate(d.itertuples()):
                    alternative=r.sample_id.endswith('::target-0.025')
                    n=3.25e19 if alternative else float(10**r.target);mu=20. if alternative else float(r.mu_cm2Vs)
                    if rr[r.sample_id]=='anchor':anchors.append(dict(input_index=i,measured_nH_cm3=n))
                    points.append(dict(sample_id=r.sample_id,role=rr[r.sample_id],common_query=r.sample_id.endswith(('::target-0.01','::target-1')),
                        observed=float(np.log10(n)),observed_log_mu=float(np.log10(mu))))
                key='q'+str(len(tasks));tasks.append(dict(id=key,input=dict(preset=preset,inputs=nodes,anchors=anchors)))
                refs[key]=dict(scheme=scheme,variant=spec['variant'],budget=budget,context=context,points=points)
    assert len(tasks)==114
    e.write(OUT/'parent_reference.json',refs);e.write(OUT/'model_lineage.json',dict(bundle_sha256=e.sha(BUNDLE/'manifest.json'),new_shared_or_hyper_fits=0))
    for i in range(4):e.write(OUT/f'tasks_{i}.json',dict(bundle=str(BUNDLE.resolve()),queries=tasks[i::4]))
    with ThreadPoolExecutor(max_workers=4) as pool:packets=list(pool.map(worker,range(4)))
    rows=[];checks=[]
    def check(n,x,tol=0.):checks.append(dict(name=n,error=float(x),tolerance=tol,passed=bool(x<=tol)))
    for packet in packets:
        check('no_label_files_read',len(packet['blocked_paths']))
        for response in packet['rows']:
            ref=refs[response['id']];result=response['result']
            for i,point in enumerate(ref['points']):
                x=result['predictions'][i] if result is not None else dict(valid=False,numeric=False)
                r=dict(id=response['id'],**{k:v for k,v in ref.items() if k!='points'},**point,valid=x['valid'],numeric=x['numeric'],error=response['error'])
                if x['valid']:r.update(predicted=x['log10_nH_cm3'],predicted_log_mu=x['log10_muH_cm2Vs'])
                if ref['budget']==0:
                    old=primary[primary.scheme.eq(ref['scheme'])&primary.variant.eq(ref['variant'])&primary.budget.eq(0)&primary.sample_id.eq(point['sample_id'])].iloc[0]
                    check(response['id']+'/'+point['sample_id']+'/frozen_zero_domain',float(old.valid!=x['valid']))
                    if x['valid']:check(response['id']+'/'+point['sample_id']+'/frozen_zero_value',abs(old.predicted-r['predicted']),2e-6)
                rows.append(r)
    assert len(rows)==456;e.csvwrite(OUT/'predictions.csv',rows);df=pd.DataFrame(rows);summary=[]
    for (scheme,v,b),g in df.groupby(['scheme','variant','budget']):
        parent=primary[primary.scheme.eq(scheme)&primary.variant.eq(v)&primary.budget.eq(b)&primary.family.eq(LEEDY)]
        for scope in ('all_query','common_query'):
            q=g[g.role.eq('query')];old=parent[parent.role.eq('query')]
            if scope=='common_query':q=q[q.common_query.eq(True)];old=old[old.common_query.eq(True)]
            check(f'{scheme}/{v}/{b}/{scope}/count',abs(len(q)-len(old)))
            complete=bool(q.numeric.all() and q.valid.all());original=bool(old.numeric.all() and old.valid.all())
            summary.append(dict(scheme=scheme,variant=v,budget=int(b),scope=scope,queries=len(q),valid=int(q.valid.sum()),
                alternative_nH_MAE=float(abs(q.predicted-q.observed).mean()) if complete else None,
                primary_nH_MAE=float(abs(old.predicted-old.observed).mean()) if original else None,
                alternative_mu_MAE=float(abs(q.predicted_log_mu-q.observed_log_mu).mean()) if complete else None,
                primary_mu_MAE=float(abs(old.predicted_log_mu-old.observed_log_mu).mean()) if original else None))
    assert len(summary)==228;e.csvwrite(OUT/'summary.csv',summary)
    audit=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,requests=len(tasks),point_rows=len(df),
        numeric=int(df.numeric.sum()),valid=int(df.valid.sum()),new_shared_fits=0,new_sourcehyperselection=False,
        primary_labels_unchanged=True,shared_training_label_sensitivity_performed=False,goal_complete=False)
    e.write(OUT/'audit.json',audit);print('Leedy retained prose sensitivity',audit['passed'],len(df),flush=True)
    if not audit['passed']:raise RuntimeError('Keep failed label sensitivity before repair')


if __name__=='__main__':
    if '--protocol-only' in sys.argv:protocol();print('Prose sensitivity protocol frozen')
    else:main()
