"""Exploratory read-only diagnosis of all pre-existing source penalty candidates.

Outer labels are used descriptively here. No candidate found with these labels
becomes a validated model or replaces the frozen nested-selection results.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE=e.BASE/'phase25_shared_process';P=BASE/'prediction';OUT=BASE/'analysis/candidate_diagnosis'


def main():
    assert json.loads((P/'audit.json').read_text())['passed']
    plans=json.loads((P/'plans.json').read_text());refs=json.loads((P/'parent_reference.json').read_text())
    table=read(BASE/'data/conditions33.csv');roles=read(BASE/'data/episodes33.csv')
    results={r['id']:r for i in range(4) for r in json.loads((P/f'outputs_{i}.json').read_text())['records']}
    rows=[]
    for plan in plans:
        if plan['held_unit'] is None or plan['kind']!='field':continue
        for request in plan['outer']:
            tid=request['task_id'];ref=refs[tid];d=table[table.family.eq(ref['family'])]
            for ep in results[tid]['episodes']:
                b=ep['budget']
                if not b:continue
                rr=roles[roles.family.eq(ref['family'])&roles.budget.eq(b)].set_index('sample_id').role
                common=set(roles[roles.family.eq(ref['family'])&roles.budget.eq(2)&roles.role.eq('query')].sample_id)
                for solution in ep['solutions']:
                    anchor=[];query=[];ja=[];jq=[]
                    for r in d.itertuples():
                        p=solution['records'][ref['sample_node'][r.sample_id]]
                        if not (p['numeric'] and p['valid']):continue
                        ae=abs(p['predicted']-r.target);derivative=abs(p['source_derivative'])
                        if rr[r.sample_id]=='anchor':anchor.append(ae);ja.append(derivative)
                        elif r.sample_id in common:query.append(ae);jq.append(derivative)
                    complete=len(query)==len(common) and len(anchor)==b
                    amin=float(min(ja)) if ja else 0.;qmax=float(max(jq)) if jq else 0.
                    rows.append(dict(scheme=plan['scheme'],variant=plan['variant'],family=ref['family'],
                        related_cluster=str(d.related_cluster.iloc[0]),budget=b,lambda_text='inf' if solution['lam'] is None else str(solution['lam']),
                        raw_shift=solution['raw_shift'],common_queries=len(query),complete=complete,
                        common_MAE=float(np.mean(query)) if complete else None,anchor_MAE=float(np.mean(anchor)) if complete else None,
                        minimum_anchor_abs_derivative=amin,maximum_query_abs_derivative=qmax,
                        local_query_to_anchor_ratio=qmax/amin if amin>0 else None,
                        source_bound_active=abs(solution['raw_shift'])>5.999 if solution['raw_shift'] is not None else False))
    e.csvwrite(OUT/'per_family_candidates.csv',rows);d=pd.DataFrame(rows);summary=[]
    for key,g in d.groupby(['scheme','variant','budget','lambda_text']):
        full=bool(g.complete.all() and len(g)==6)
        summary.append(dict(scheme=key[0],variant=key[1],budget=int(key[2]),lambda_text=key[3],full_coverage=full,
            DOI_macro_MAE=float(g.common_MAE.mean()) if full else None,
            group_macro_MAE=float(g.groupby('related_cluster').common_MAE.mean().mean()) if full else None,
            worst_DOI_MAE=float(g.common_MAE.max()) if full else None,source_bounds=int(g.source_bound_active.sum())))
    e.csvwrite(OUT/'summary.csv',summary)
    e.write(OUT/'audit.json',dict(passed=True,family_candidate_rows=len(rows),summaries=len(summary),
        original_predictions_and_selection_unchanged=True,new_fits=0,new_point_evaluations=0,
        uses_outer_labels_for_exploratory_diagnosis=True,validated_oracle_selection=False,goal_complete=False,
        code_sha256=e.sha(Path(__file__))))
    print('Pre-existing source-candidate diagnosis',len(rows),len(summary),'no new fitting',flush=True)


if __name__=='__main__':main()
