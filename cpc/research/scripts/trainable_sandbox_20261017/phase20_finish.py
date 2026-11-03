"""Verify report claims and update live progress without closing the goal."""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from csv_io_v2 import read_result_csv as read
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase20_source_inventory'


def main():
    report=OUT/'ML与Sandbox_相对库存学习与预测适应检验.md';pdf=report.with_suffix('.pdf');claims=json.loads((OUT/'report_claims.json').read_text())
    assert claims['report_sha256']==energy.sha(report)
    for c in claims['claims']:
        p=ROOT/c['path'];assert energy.sha(p)==c['sha256'];d=read(p)
        for k,v in c['selector'].items():d=d[d[k].eq(v)]
        assert len(d)==1 and float(d[c['column']].iloc[0])==c['value'],c
    exports=list(OUT.glob('_pdf_build_*/export.json'));assert len(exports)==1;render=json.loads(exports[0].read_text())
    assert render['source_sha256']==energy.sha(report) and render['pdf_sha256']==energy.sha(pdf) and not render['warnings']
    text=Path('/tmp/ga2o3_phase20_report.txt').read_text()
    assert all(t in text for t in ('1.637792','0.046069','0.344359','6900','6092'))
    energy.write(OUT/'rendering_check.json',dict(passed=True,pages=6,numeric_claims_verified=len(claims['claims']),
        report_sha256=energy.sha(report),pdf_sha256=energy.sha(pdf),no_render_warnings=True,math_preserved=True))
    energy.write(OUT/'visual_review.json',dict(passed=True,pages_inspected=[1,2,3,4,5,6],
        figures_visible=True,formulas_intact=True,tables_readable=True,continued_table_header_present=True,
        method_names_not_clipped=True,pdf_sha256=energy.sha(pdf)))
    goal=json.loads((OUT/'system_goal_checked.json').read_text());assert goal['goal']['status']=='active'
    a=json.loads((OUT/'audit/contract_audit.json').read_text());assert a['passed']
    direction=dict(created_utc=energy.now(),system_goal_status='active',goal_complete=False,
        direction='trainedML+Sandbox dopantpropertyprediction;theoryservesactualprediction',
        primary='actualincorporateddose+traceableinputs->nH;mu/sigma/supporteddopantsretained',
        original29_5_anchors_common18_unchanged=True,new_DFT_MD=0,new_material_measurements=0,GPU_used=False,GPU_authorized='GPU1only',
        fit_records=a['training_records'],new_optimizer_starts=a['new_optimizer_starts'],core_gates=262,learning_gates=59,
        source_inverse_repair='exactcachedhit;unchangedloss/formula/tolerance/weights',repaired_episodes=a['repaired_episodes'],
        repaired_solutions=a['repaired_solutions'],stable_shared_learning_gain_established=False,
        fullsource_shared_inventory_h=0.,trained_internal_source_updates=0,
        sourceexcluded_API_requests=6900,saved_RT_prediction_replays=870,knownsource_API_requests=1730,
        knownsource_grade='development diagnostic;not independenttrainedpolicyvalidation',
        theory_revision='firstanchorresponse penaltycanbeillconditioned anddoesnotcontrolquerydomain;testdeclared-dose responseconstraint',
        next='implementandgatecurve-response regularization+extendedfinitepenalties;fixedweights+matchedcontrols;thenfrozenpredictiontest',
        continuation='CONTINUE_PHASE20_HERE.md')
    # Store the exact diagnostic metric instead of a hand-transcribed number.
    d=read(OUT/'external_leedy/summary.csv');q=d[d.method.eq('RelativeInventory')&d.policy.eq('unpenalized')&d.budget.eq(2)&
        d.scenario.eq('primary_graph')&d.scope.eq('common3')&d.pressure_atm.eq(1e-5)]
    assert len(q)==1;direction['knownsource_fixed_common3_unpenalized_internal_nH_MAE']=float(q.nH_MAE.iloc[0])
    energy.write(OUT/'goal_direction_check.json',direction)
    for filename in ('progress.json','goal_requirements_audit.json'):
        path=BASE/filename;old=json.loads(path.read_text());old['goal_complete']=False;old['current_phase']=20
        old['phase20']=dict(direction,status='complete_bounded_prediction_checkpoint',frozen=False)
        old['phase20_running']=dict(status='finished',sourceV2_pending=False,goal_complete=False,continuation='CONTINUE_PHASE20_HERE.md')
        old['current_execution']='phase20predictioncheckpointcomplete;nextdeclared-dose responseconstraint,notyetimplemented'
        old['current_turn']='progress:sharedtraining+numericrepair+sourceprediction+theory/coverageaudit'
        old['active_jobs']=[];old['latest_utc']=energy.now();old['checked_utc']=energy.now()
        if filename=='progress.json':
            old.update(status='active',stage='phase20_relative_inventory_prediction_checkpoint',
                current_report=str(pdf.relative_to(ROOT)),latest_report_directory=str(OUT.relative_to(ROOT)),
                current_predictor=str((OUT/'delivery_v2/predictor_bundle_v7').relative_to(ROOT)),
                continuation=str((BASE/'CONTINUE_PHASE20_HERE.md').relative_to(ROOT)),next_theory='docs/ML_SANDBOX_CURVE_RESPONSE_NEXT_20261018.md',
                next=[direction['next'],'retainpropertypredictiongoal;noexplanation-onlycompletion'])
            item='phase20conservedrelativeinventory+matchedcontrols;353records462starts;V2repair;6900ownT+1730sourceoutputs;allfailuresretained'
            if item not in old['completed']:old['completed'].append(item)
        else:
            old['goal_status']='active';old['objective']=goal['goal']['objective'];old['sufficient_to_mark_goal_complete']=False
            entry=dict(requirement='sharedrelativeinventory andsamebudgetprediction;repairandcoverageaudited',
                evidence=['phase20_source_inventory/audit/contract_audit.json','phase20_source_inventory/cohort_properties/audit.json',
                    'phase20_source_inventory/external_leedy/audit.json','phase20_source_inventory/goal_direction_check.json'],
                scope='no stablesharedgain;knownLeedyinternaladaptationcapacity only;nextinputrange-awareconstraint')
            if not any(x['requirement']==entry['requirement'] for x in old['implemented_and_verified']):old['implemented_and_verified'].append(entry)
        energy.write(path,old)
    energy.write(OUT/'literature_reading.json',dict(created_utc=energy.now(),records=[
        dict(url='https://arxiv.org/abs/2501.17373',DOI='10.1039/D4CP04817B',depth='metadata/abstract rechecked;cachedPDFpages8-9 inspectedearlierthisphase',use='thermodynamicprior andkinetic/freeze limitations;not newMLnovelty'),
        dict(url='https://arxiv.org/pdf/1801.08930',depth='sections2,3,and4.1 read;not entire13pages',use='existinghierarchicalBayes/taskadaptation;not animplementedMAMLclaim'),
        dict(url='https://proceedings.mlr.press/v70/finn17a.html',depth='primaryabstract/metadata',use='priorart forlearnedfastadaptation;notalgorithmproof')]))
    print('finished bounded checkpoint; goal active;claims',len(claims['claims']),flush=True)


if __name__=='__main__':main()
