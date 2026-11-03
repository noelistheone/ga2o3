"""Replay saved training predictions through the deployed API at each own T.

Complements V1 integration replay with an external saved-training reference.
All measurements stay in the parent; a fresh worker receives input/anchors only.
"""
from pathlib import Path
import json,sys,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import phase19_delivery as s


def main():
    out=s.OUT/'training_api_replay';out.mkdir(exist_ok=True)
    targets=s.TARGETS.set_index('obs_id');queries=[];expected={}
    for method in s.SPECS:
        refs=json.loads((s.OUT/'training_targets'/f'{method}.json').read_text())
        for r in refs:
            t=targets.loc[r['obs_id']];key=f'q{len(queries)}'
            q=dict(dopant=t.element,incorporated_cm3=float(t.incorporated_cm3),method=method,
                observation_T_K=float(t.Tobs_K),H_known=bool(t.H_known),H_cm3=float(t.H_cm3),growth_T_C=float(t.growth_T_C))
            queries.append(dict(id=key,input=q));expected[key]=dict(method=method,obs_id=r['obs_id'],endpoint=t.endpoint,predicted=r['prediction'])
    # Explicit distinct-dose anchors check calibration, not prediction accuracy.
    rt=s.inputs.PRED[s.inputs.PRED.kind.eq('RT')]
    for method in ('original_RT','stock_const_noext_RT','stock_const_internal','HallII_original_RT','unit_dose','median_activation'):
        for held,g in rt.groupby('family'):
            g=g.sort_values('incorporated_cm3').drop_duplicates('incorporated_cm3')
            for b in (1,2):
                if len(g)<b:continue
                row=g.iloc[-1];same=g[g.H_known.eq(row.H_known)&g.H_cm3.eq(row.H_cm3)]
                if len(same)<b:continue
                a=[dict(incorporated_cm3=float(t.incorporated_cm3),measured_nH_cm3=float(10**t.target)) for t in same.iloc[:b].itertuples()]
                key=f'q{len(queries)}';queries.append(dict(id=key,input=dict(dopant=row.element,incorporated_cm3=float(row.incorporated_cm3),method=method,
                    H_known=bool(row.H_known),H_cm3=float(row.H_cm3),anchors=a)))
                expected[key]=dict(method=method,obs_id='distinctdoseanchor',endpoint=None,predicted=None)
    s.write(out/'tasks.json',dict(bundle=str(s.BUNDLE.resolve()),queries=queries))
    run=subprocess.run([sys.executable,str(Path(__file__).parent/'phase19_verify.py'),'--worker',str(out/'tasks.json'),str(out/'outputs.json')],cwd=ROOT,capture_output=True,text=True)
    (out/'worker.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode();actual=json.loads((out/'outputs.json').read_text());rows=[]
    for q in actual['rows']:
        r=expected[q['id']];e=None
        if q['error'] is None and r['endpoint'] is not None:
            field='log10_nH_cm3' if r['endpoint']=='nH' else 'log10_muH_cm2Vs'
            e=abs(q['result'][field]-r['predicted'])
        rows.append(dict(id=q['id'],**r,replay_error=e,error=q['error']))
    maximum=max(r['replay_error'] for r in rows if r['replay_error'] is not None)
    audit=dict(passed=maximum<2e-6 and not any(r['error'] for r in rows) and not actual['blocked'],
        queries=len(rows),saved_training_prediction_replays=sum(r['endpoint'] is not None for r in rows),
        distinct_dose_anchor_tasks=sum(r['endpoint'] is None for r in rows),maximum_error=maximum,
        property_tables_read=False,query_labels_in_worker=False,blocked_paths=actual['blocked'],
        scope='training/deployment numerical consistency and calibration;no new material validation')
    s.energy.csvwrite(out/'replay.csv',rows);s.write(out/'verification.json',audit);print(json.dumps(audit),flush=True);assert audit['passed']


if __name__=='__main__':main()
