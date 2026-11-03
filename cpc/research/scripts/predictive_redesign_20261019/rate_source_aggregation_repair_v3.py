"""Finish unchanged source selection after a confirmed accounting-only failure.

Never edit the bound predictor, refit a model, or rerun an original runtime task.
Only missing derived artifacts are written, with a separate repair protocol.
"""
from pathlib import Path
import argparse,importlib,sys,time
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

MODULES={'31':'phase31_prediction','32':'phase32_prediction','trees':'strong_tree_prediction'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=MODULES)
    args=parser.parse_args();module=importlib.import_module(MODULES[args.phase])
    out=module.OUT;root=out.parent;fix=out/'repair_source_aggregation_v3'
    assert not (fix/'protocol_before.json').exists(),'Inspect original repair;do not overwrite'
    expected='KeyError: \'seconds\''
    log=root/'prediction.log'
    assert expected in log.read_text() and 'source_evaluate' in log.read_text()
    failed=(root/'pipeline_status.json') if args.phase=='trees' else (root/'after_training_failure.json')
    failure=c.document(failed)
    if args.phase=='trees':assert failure['prediction_status']==1
    assert not (out/'prediction_complete.json').exists()
    for name in ['source_compute.json','predictions.csv','prediction_diagnostics.json',
                 'source_choices.json','full_training_presets.json','summary.csv','per_family.csv',
                 'all_comparisons.csv','all_per_family.csv']:
        assert not (out/name).exists(),('Do not overwrite derived result',name)
    design=c.document(out/'design_before_prediction.json');c.verify_hashes(design)
    plan=c.document(out/'source_plan.json')
    workers=sorted((out/'source_runtime').glob('worker[0-9].json'))
    assert len(workers)==(2 if args.phase=='trees' else 4)
    old_plan=c.OLD/'prediction/plan.json'
    old_workers=sorted((c.OLD/'prediction/runtime').glob('worker[0-9].json'))
    inputs=[failed,log,out/'source_plan.json',out/'shared_selections.json',out/'zero_complete.json',
        out/'candidate_zero_predictions.csv',out/'training_audit.json',out/'design_before_prediction.json',
        old_plan,c.OLD/'data/conditions35.csv',c.OLD/'data/scenarios49.csv',c.OLD/'data/episodes35.csv',
        c.OLD/'prediction/predictions.csv',c.OUT/'predictions.csv',*workers,*old_workers]
    c.write(fix/'protocol_before.json',dict(created_utc=c.e.now(),phase=args.phase,
        failure_type='source accounting reads nonexistent top-level worker seconds',
        correction='sum each fresh task record seconds;all original task values and selection functions unchanged',
        writes_only_missing_derived_outputs=True,original_failure_retained=True,
        new_fits=0,new_model_or_source_evaluations=0,
        code_hashes=c.file_hashes([Path(__file__),Path(module.__file__),Path(c.__file__)]),
        input_hashes=c.file_hashes(inputs),goal_complete=False))
    before=time.perf_counter();records={}
    for path in workers:
        value=c.document(path);assert value['denied_reads']==[] and value['runtime_query_labels'] is False
        for r in value['records']:
            assert r['id'] not in records;records[r['id']]=r
    oldtasks={t['id']:t for t in c.document(old_plan)['tasks']}
    oldrecords={r['id']:r for p in old_workers for r in c.document(p)['records']}
    reused={};fresh=[]
    for task in plan['tasks']:
        if task['id'] in oldtasks and oldtasks[task['id']]==task:
            assert task['id'] not in records
            reused[task['id']]=oldrecords[task['id']]
        else:
            assert task['id'] in records;fresh.append(task)
    assert set(records)=={t['id'] for t in fresh}
    records.update(reused);assert len(records)==len(plan['tasks'])
    c.write(out/'source_compute.json',dict(tasks=len(plan['tasks']),new_tasks=len(fresh),
        reused_tasks=len(reused),reused_ids=sorted(reused),
        actual_new_physical_calls=sum(records[t['id']]['actual_physical_point_calls'] for t in fresh),
        actual_worker_seconds=sum(records[t['id']]['seconds'] for t in fresh),
        worker_seconds_definition='sum original fresh task seconds;runtime has no top-level elapsed field',
        prior_compute_not_counted_as_new=True,repair_new_model_evaluations=0,
        repaired_by=str(Path(__file__).relative_to(c.ROOT)),created_utc=c.e.now()))
    table=module.choose_sources(plan,records)
    label='strong_trees' if args.phase=='trees' else 'phase'+args.phase
    rows,families=module.summary(table,label)
    c.e.csvwrite(out/'summary.csv',rows);c.e.csvwrite(out/'per_family.csv',families)
    for study,path in [('phase29',c.OLD/'prediction/predictions.csv'),('phase30',c.OUT/'predictions.csv')]:
        a,b=module.summary(c.read(path),study);rows.extend(a);families.extend(b)
    c.e.csvwrite(out/'all_comparisons.csv',rows);c.e.csvwrite(out/'all_per_family.csv',families)
    c.verify_hashes(c.document(fix/'protocol_before.json'));c.verify_hashes(design)
    outputs=[out/v for v in ['source_compute.json','predictions.csv','prediction_diagnostics.json',
        'source_choices.json','full_training_presets.json','summary.csv','per_family.csv',
        'all_comparisons.csv','all_per_family.csv']]
    c.write(fix/'repair_complete.json',dict(created_utc=c.e.now(),phase=args.phase,
        new_fits=0,new_model_or_source_evaluations=0,original_failure_retained=True,
        original_runtime_tasks=len(records),rows=len(table),
        selection_functions_unmodified=True,code_hashes=c.file_hashes([Path(__file__),Path(module.__file__)]),
        output_hashes=c.file_hashes(outputs),seconds=time.perf_counter()-before,goal_complete=False))
    c.write(out/'prediction_complete.json',dict(created_utc=c.e.now(),variants=len(module.VARIANTS),
        rows=len(table),postprocess_seconds=time.perf_counter()-before,
        source_aggregation_repaired=True,original_prediction_terminal_status=1,
        repair_protocol_sha256=c.e.sha(fix/'protocol_before.json'),
        repair_receipt_sha256=c.e.sha(fix/'repair_complete.json'),
        independent_result_audit_still_required=True,goal_complete=False))
    print('Versioned aggregation repair',args.phase,'original tasks',len(records),'rows',len(table),
        'new model evaluations0',flush=True)


if __name__=='__main__':main()
