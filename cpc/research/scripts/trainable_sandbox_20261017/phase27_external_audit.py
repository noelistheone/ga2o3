"""Audit transfer inputs, calibration budgets, uncertainty grades and warnings."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase27_external as x
import phase2_energy as e


def main():
    out=x.OUT;p=json.loads((out/'protocol_before_transfer.json').read_text())
    for k in ('code_hashes','input_hashes'):
        for name,h in p[k].items():assert e.sha(ROOT/name)==h,name
    refs=json.loads((out/'parent_reference.json').read_text());df=read(out/'predictions.csv');summary=read(out/'summary.csv')
    manifest=json.loads((x.BUNDLE/'manifest.json').read_text());tables={d.input_tier.iloc[0]:d for d in x.source_tables()}
    checks=[];scope=[]
    def check(name,error,tol=0.):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
    check('output_rows',abs(len(df)-5760));check('summary_rows',abs(len(summary)-1440));check('requests',abs(len(refs)-720))
    for i in range(4):
        task=json.loads((out/f'tasks_{i}.json').read_text());results=json.loads((out/f'outputs_{i}.json').read_text())
        check(f'worker{i}/no_label_file_reads',len(results['blocked_paths']))
        rq={r['id']:r for r in task['queries']}
        for response in results['rows']:
            key=response['id'];ref=refs[key];request=rq[key]['input'];table=tables[ref['input_tier']]
            check(key+'/no_query_targets',sum('observed' in n or 'target' in n for n in request['inputs']))
            check(key+'/source_budget',abs(len(request['anchors'])-ref['budget']))
            aids=[int(table[table.sample_label.eq(label)].index[0]) for label in p['anchors'][ref['input_tier']][:ref['budget']]]
            check(key+'/anchor_identity',float([a['input_index'] for a in request['anchors']]!=aids))
            for a in request['anchors']:
                check(key+'/allowed_anchor_value/'+str(a['input_index']),abs(a['measured_nH_cm3']/table.iloc[a['input_index']].nH_cm3-1.),1e-12)
            for j,row in enumerate(table.itertuples()):
                node=request['inputs'][j]
                check(key+f'/{j}/chemical_input',abs(node['incorporated_cm3']/row.C_input_cm3-1.),1e-12)
                check(key+f'/{j}/explicit_oxygen_scenario',abs(node['scenario_pO2_atm']-1e-5))
                check(key+f'/{j}/comparison_temperature',abs(node['Tobs_K']-300.))
            result=response['result']
            if result is None:continue
            spec=manifest['presets'][request['preset']]
            check(key+'/frozen_source_policy',float(result['source_lambda']!=spec['source_lambdas'][str(ref['budget'])]))
            for j,point in enumerate(result['predictions']):
                scope.append(dict(id=key,context=ref['context'],variant=ref['variant'],input_tier=ref['input_tier'],budget=ref['budget'],
                    sample_label=table.iloc[j].sample_label,numeric=point['numeric'],valid=point['valid'],
                    physical_state_available=point.get('physical_state_available'),same_physical_state_for_output=point.get('same_physical_state_for_output'),
                    warnings=' | '.join(point.get('warnings',[])),warning_count=len(point.get('warnings',[]))))
                if point['valid']:
                    check(key+f'/{j}/coherent_derived_sigma',abs(point['log10_sigma_S_cm']-(np.log10(1.602176634e-19)+point['log10_nH_cm3']+point['log10_muH_cm2Vs'])),2e-9)
    for key,g in df[df.role.eq('query')].groupby(['context','variant','input_tier','budget']):
        for mode in ('all_query','common_query'):
            q=g if mode=='all_query' else g[g.common_query.eq(True)]
            s=summary[(summary.context.eq(key[0]))&summary.variant.eq(key[1])&summary.input_tier.eq(key[2])&summary.budget.eq(key[3])&summary.scope.eq(mode)]
            check('unique_summary/'+e.sha(out/'summary.csv')[:8]+'/'+str(key)+'/'+mode,abs(len(s)-1))
            if len(s)==1:
                if q.valid.all():check(str(key)+'/'+mode+'/MAE',abs(float(np.abs(q.predicted-q.observed).mean())-s.nH_MAE.iloc[0]),1e-12)
                else:check(str(key)+'/'+mode+'/no_partial_score',float(pd.notna(s.nH_MAE.iloc[0])))
    e.csvwrite(out/'operator_scope_and_warnings.csv',scope)
    value=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,point_rows=len(df),valid=int(df.valid.sum()),
        warned_points=sum(r['warning_count']>0 for r in scope),actual_Hall_conditions=12,approximate_same_Hall_chemistry_pairs=2,
        two_input_tasks_not_independent_conditions=True,not_a_pure_chemistry_ablation=True,
        shared_weights_or_sourcehyperparameters_refitted=False,independent_blind_or_lab_validation=False,
        goal_complete=False,code_sha256=e.sha(Path(__file__)))
    e.write(out/'audit.json',value)
    print('phase27 transfer audit',value['passed'],len(checks),'warned',value['warned_points'],flush=True)
    if not value['passed']:raise RuntimeError('Preserve failed source transfer audit before repair')


if __name__=='__main__':main()
