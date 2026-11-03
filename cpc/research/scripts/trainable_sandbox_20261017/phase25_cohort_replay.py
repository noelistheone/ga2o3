"""Replay every selected outer prediction through the exported label-free API."""
from pathlib import Path
import sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from sandbox.context_saved_model import INPUT_NAMES
import phase25_training as t
import phase2_energy as e

OUT=t.OUT/'delivery/cohort_replay';BUNDLE=t.OUT/'delivery/predictor_bundle'


def worker(i):
    run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase25_api_runtime.py'),str(OUT/f'tasks_{i}.json'),str(OUT/f'outputs_{i}.json')],cwd=ROOT,capture_output=True,text=True)
    (OUT/f'worker_{i}.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode()
    return json.loads((OUT/f'outputs_{i}.json').read_text())


def main():
    assert json.loads((t.OUT/'prediction/audit.json').read_text())['passed']
    manifest=json.loads((BUNDLE/'manifest.json').read_text());reference=read(t.OUT/'prediction/predictions.csv')
    choices=json.loads((t.OUT/'prediction/selections.json').read_text());tasks=[];refs={}
    for choice in choices:
        held=choice['held_unit']
        if held is None:continue
        scheme=choice['scheme'];col=t.unit_column(scheme);b=choice['budget']
        preset=scheme+'::without:'+held+'::'+choice['variant'];assert preset in manifest['presets']
        for family in sorted(t.TABLE.loc[t.TABLE[col].eq(held),'family'].unique()):
            d=t.TABLE[t.TABLE.family.eq(family)].sort_values(['incorporated_cm3','sample_id'])
            rows=list(d.itertuples());nodes=[{k:e.clean(getattr(r,k)) for k in INPUT_NAMES} for r in rows]
            roles=t.ROLES[t.ROLES.family.eq(family)&t.ROLES.budget.eq(b)].set_index('sample_id').role
            anchors=[dict(input_index=i,measured_nH_cm3=float(10**r.target)) for i,r in enumerate(rows) if roles[r.sample_id]=='anchor']
            key='q'+str(len(tasks));tasks.append(dict(id=key,input=dict(preset=preset,inputs=nodes,anchors=anchors)))
            refs[key]=dict(scheme=scheme,variant=choice['variant'],budget=b,family=family,sample_ids=[r.sample_id for r in rows],preset=preset)
    assert len(tasks)==684
    hashes=[Path(__file__),ROOT/'scripts/trainable_sandbox_20261017/phase25_api_runtime.py',ROOT/'src/sandbox/context_property_predictor.py']
    e.write(OUT/'protocol.json',dict(created_utc=e.now(),requests=len(tasks),expected_point_rows=3762,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in hashes},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (BUNDLE/'manifest.json',t.OUT/'prediction/predictions.csv')},
        grade='fresh API replay of retrospective held-out models,not newmaterial validation',query_targets_in_packets=False,goal_complete=False))
    e.write(OUT/'parent_reference.json',refs)
    for i in range(4):e.write(OUT/f'tasks_{i}.json',dict(bundle=str(BUNDLE.resolve()),queries=tasks[i::4]))
    with ThreadPoolExecutor(max_workers=4) as pool:outputs=list(pool.map(worker,range(4)))
    indexed=reference.set_index(['scheme','variant','budget','family','sample_id']);saved=[];checks=[];maximum=0.
    def check(name,error,tol=0.):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
    for packet in outputs:
        check('no_label_reads',len(packet['blocked_paths']))
        for response in packet['rows']:
            ref=refs[response['id']];result=response['result']
            for i,sid in enumerate(ref['sample_ids']):
                r=indexed.loc[ref['scheme'],ref['variant'],ref['budget'],ref['family'],sid]
                actual=result['predictions'][i] if result is not None else dict(valid=False,numeric=False)
                label=response['id']+'/'+sid
                check(label+'/domain',float(bool(actual['valid'])!=bool(r.valid)))
                row=dict(id=response['id'],sample_id=sid,scheme=ref['scheme'],variant=ref['variant'],budget=ref['budget'],
                    reference_valid=bool(r.valid),API_valid=bool(actual['valid']),error=response['error'])
                if actual['valid'] and r.valid:
                    values=np.array([actual['log10_nH_cm3'],actual['log10_muH_cm2Vs'],actual['log10_sigma_S_cm']])
                    expected=np.array([r.predicted,r.predicted_log_mu,r.predicted_log_sigma])
                    difference=float(np.max(np.abs(values-expected)));maximum=max(maximum,difference)
                    check(label+'/full_chain_replay',difference,2e-6);row['maximum_log_error']=difference
                saved.append(row)
    check('point_rows',abs(len(saved)-3762))
    e.csvwrite(OUT/'replayed_points.csv',saved)
    result=dict(passed=all(r['passed'] for r in checks),checks=len(checks),rows=checks,requests=len(tasks),
        point_rows=len(saved),maximum_log_property_difference=maximum,no_experimental_tables_read=True,goal_complete=False)
    e.write(OUT/'audit.json',result)
    print('context API cohort replay',result['passed'],len(tasks),len(saved),'max',maximum,flush=True)
    if not result['passed']:raise RuntimeError('Preserve failed independent API replay before repair')


if __name__=='__main__':main()
