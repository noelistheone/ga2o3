"""Choose operational predictor policies using training-inner scores only."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE=e.BASE/'phase25_shared_process';OUT=BASE/'method_policy'
INTERNAL=('InventoryConstant_internal','InventoryDose_internal','InventoryProcess_internal')


def complexity(choice):
    method=choice['method']
    if method=='UnitDose':n=0
    elif method=='MedianActivation':n=1
    elif method in ('DoseLine','FixedPhysical'):n=2
    elif method=='DirectRidge':n=11
    else:n=2+dict(Constant=1,Dose=2,Process=4)[next(s for s in ('Constant','Dose','Process') if method.endswith(s))]
    if choice['selected_source']['lam'] is not None:n+=2 if choice['kind']=='affine' else 1
    return n


def preregister():
    p=OUT/'protocol_before_policy.json'
    value=dict(created_utc=e.now(),purpose='operationalbest predictor delivery;no new sharedfit or test-label selection',
        policies={'BestOverall':'all19variants','BestInternal':INTERNAL},
        zero_anchor_score='shared gamma-selected innerzeroanchor score,not outerMAE',
        nonzero_anchor_score='selected lambda training-inner query score,not outerMAE',
        ties='roundscore10digits,then declared parametercount,then variantname',
        budget_dependence='selected model may change with anchorbudget;do not call its errorchange pure sourceadaptationgain',
        inheritance='all original19variant results remain primary;this additional policy analysis is separately disclosed',
        sha256=e.sha(Path(__file__)),goal_complete=False)
    if p.exists():assert json.loads(p.read_text())['sha256']==value['sha256']
    else:e.write(p,value)


def number(x):return np.inf if x is None else float(x)


def main():
    preregister();assert json.loads((BASE/'prediction/audit.json').read_text())['passed']
    choices=json.loads((BASE/'prediction/selections.json').read_text());shared=json.loads((BASE/'prediction/shared_choices.json').read_text())
    shared={(r['scheme'],r['held'],r['method']):r for r in shared}
    groups={}
    for c in choices:groups.setdefault((c['scheme'],c['held_unit'],c['budget']),[]).append(c)
    aliases={};selected=[];checks=[]
    for (scheme,held,budget),candidates in groups.items():
        for policy in ('BestOverall','BestInternal'):
            subset=candidates if policy=='BestOverall' else [c for c in candidates if c['variant'] in INTERNAL]
            assert len(subset)==(19 if policy=='BestOverall' else 3)
            scored=[]
            for c in subset:
                if budget:
                    score=number(c['selected_source']['score'])
                else:
                    source=shared[scheme,held,c['method']]
                    match=next(r for r in source['candidates'] if r['reg']==source['reg'])
                    score=number(match['score'])
                scored.append(dict(variant=c['variant'],inner_score=score,complexity=complexity(c)))
            best=min(scored,key=lambda r:(round(r['inner_score'],10),r['complexity'],r['variant']))
            context=scheme+('::full' if held is None else '::without:'+held)
            name=context+'::'+policy
            aliases.setdefault(name,{})[str(budget)]=context+'::'+best['variant']
            selected.append(dict(scheme=scheme,held_unit=held,budget=budget,policy=policy,selected_variant=best['variant'],
                inner_score=best['inner_score'],all_candidates=scored,query_labels_used=False))
    assert len(selected)==78 and len(aliases)==26
    e.write(OUT/'selections.json',selected)
    e.write(OUT/'predictor_policies.json',dict(version=1,aliases=aliases,selection='training-inner scores only',
        source_choices_sha256=e.sha(BASE/'prediction/selections.json'),query_targets_in_policy=False,goal_complete=False))
    df=read(BASE/'prediction/predictions.csv');rows=[]
    for c in selected:
        if c['held_unit'] is None:continue
        g=df[df.scheme.eq(c['scheme'])&df.held_unit.eq(c['held_unit'])&df.budget.eq(c['budget'])&df.variant.eq(c['selected_variant'])]
        assert len(g)>0
        for r in g.to_dict('records'):rows.append(dict(r,selected_variant=r['variant'],variant=c['policy']))
    e.csvwrite(OUT/'predictions.csv',rows);d=pd.DataFrame(rows);summary=[]
    for (scheme,policy,budget),g in d.groupby(['scheme','variant','budget']):
        for scope in ('all_query','common20'):
            q=g[g.role.eq('query')]
            if scope=='common20':q=q[q.common_query.eq(True)]
            full=bool(q.numeric.all() and q.valid.all())
            summary.append(dict(scheme=scheme,policy=policy,budget=int(budget),scope=scope,queries=len(q),valid=int(q.valid.sum()),
                nH_MAE=float(np.abs(q.predicted-q.observed).groupby(q.family).mean().mean()) if full else None,
                prediction_method_changes_with_budget=True))
    e.csvwrite(OUT/'summary.csv',summary)
    e.write(OUT/'audit.json',dict(passed=True,selected_choices=len(selected),aliases=len(aliases),outer_prediction_rows=len(rows),
        no_new_fits=True,selection_inputs_are_inner_scores_only=True,outer_labels_used_for_scoring_only=True,
        source_file_sha256=e.sha(Path(__file__)),goal_complete=False))
    print('Training-inner predictor policies selected',len(selected),'aliases',len(aliases),flush=True)


if __name__=='__main__':main()
