"""Re-evaluate complete observables of every saved candidate, with no fitting."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,sys,json,time,argparse,subprocess
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from sandbox.kroger_db import NPZ


def make_plan():
    table=c.cases();tasks=[];requests=[]
    for train in c.learning.contexts():
        if len(train)==5:continue
        for method in c.METHODS:
            for config in c.learning.configs(method):
                fit,path=c.fit(train,method,config)
                minimal=c.design.minimal_fit(fit)
                for held in c.units():
                    if held in train:continue
                    test=table[table.related_cluster.eq(held)]
                    nodes=[c.design.inputs(row) for row in test.itertuples()]
                    kind='internal' if method in c.design.PHYSICAL else 'offset'
                    task=dict(fit=minimal,kind=kind,nodes=nodes,
                        episodes=[dict(budget=0,anchor_condition_ids=[],
                            anchor_indices=[],anchor_labels=[])])
                    task['id']=c.digest(task)
                    tasks.append(task)
                    requests.append(dict(task_id=task['id'],fit_key=fit['key'],held=held,
                        method=method,config=config,train_units=list(train),
                        case_ids=test.case_id.tolist(),fit_path=str(path.relative_to(c.ROOT)),
                        fit_sha256=c.e.sha(path)))
    assert len(tasks)==2175 and len({t['id'] for t in tasks})==len(tasks)
    return dict(tasks=tasks,requests=requests)


def declare(plan):
    target=c.OUT/'protocol_before_zero.json'
    assert not target.exists(),'Preserve previous attempt before versioned repair'
    files=[c.OLD/'training_complete.json',c.OLD/'training_audit.json',
        c.OLD/'data/conditions35.csv',c.OLD/'data/scenarios49.csv',
        c.OLD/'data/episodes35.csv',c.OLD/'training_protocol.json',
        c.BASE/'goal_created.json',c.OLD/'prediction_design_protocol.json',NPZ,
        c.BASE/'phase30_theory_and_protocol.md']
    files += [c.ROOT/r['fit_path'] for r in plan['requests']]
    # Every reused input is checked against the already frozen artifact map.
    manifest=c.document(c.OLD.parent/'artifact_manifest_phase29.json')
    parent={v['path']:v['sha256'] for v in manifest['files']}
    for path in files:
        key=str(path.relative_to(c.ROOT))
        if key in parent:assert c.e.sha(path)==parent[key],key
    code=[Path(__file__),Path(c.__file__),
        c.OLD_SCRIPTS/'phase29_source_runtime.py',
        c.ROOT/'src/sandbox/joint_saved_model.py',
        c.ROOT/'src/sandbox/joint_inventory_scattering.py',
        c.ROOT/'src/sandbox/reference_scattering.py',
        c.ROOT/'src/sandbox/learnable_constant_tau.py']
    code += list((c.ROOT/'src/sandbox').glob('*.py'))
    code += [c.OLD_SCRIPTS/name for name in ('phase29_learning.py','phase29_data.py',
        'phase29_training_v2.py','phase29_prediction_design.py','csv_io_v2.py','phase2_energy.py')]
    value=dict(created_utc=c.e.now(),purpose='frozen-candidate complete-observable evaluation;no new training',
        primary=c.PRIMARY,methods=c.METHODS,policies=c.POLICIES,
        tasks=2175,source_workers=4,all_candidate_parameters_fixed=True,
        global_selection='training-inner related-group CV nH minimum;rho_safe or rho+mu constraints relative to training-inner chosen Reference205;fallback explicit',
        reference_selection='same legacy nH-only global CV across Reference205 gamma candidates',
        constraints_are_empirical_not_population_guarantees=True,
        source_selection='same prior finite Hall-only coordinate/lambda grid;constraints against no-adaptation inner predictions;no query mu given to source mapper',
        code_hashes=c.file_hashes(set(code)),input_hashes=c.file_hashes(set(files)),
        parent_manifest_sha256=c.e.sha(c.OLD.parent/'artifact_manifest_phase29.json'),
        retrospective_development=True,independent_blind_validation=False,
        new_training=0,goal_complete=False)
    c.write(target,value)
    return value


def run(plan,workers):
    packets=[[] for _ in range(workers)];loads=np.zeros(workers)
    for task in sorted(plan['tasks'],key=lambda t:(-len(t['nodes']),t['id'])):
        idx=int(np.argmin(loads));packets[idx].append(task);loads[idx]+=len(task['nodes'])
    runtime=c.OUT/'zero_runtime';runtime.mkdir(parents=True)
    def launch(index):
        packet=runtime/f'packet{index}.json';target=runtime/f'worker{index}.json'
        c.write(packet,dict(tasks=packets[index],lambdas=[None],physical_reference_sha256=c.e.sha(NPZ)))
        with (runtime/f'worker{index}.log').open('w') as log:
            job=subprocess.Popen([sys.executable,str(c.OLD_SCRIPTS/'phase29_source_runtime.py'),
                str(packet),str(target)],cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT)
            c.write(runtime/f'worker{index}_started.json',dict(
                created_utc=c.e.now(),pid=job.pid,packet_tasks=len(packets[index]),
                declared_input_nodes=int(loads[index])))
            code=job.wait()
        assert code==0,(index,code)
        value=c.document(target)
        assert value['denied_reads']==[],(index,value.keys())
        return value
    with ThreadPoolExecutor(max_workers=workers) as pool:
        values=list(pool.map(launch,range(workers)))
    records={r['id']:r for v in values for r in v['records']}
    assert len(records)==2175
    return records,values


def collate(plan,records):
    cases={r.case_id:r for r in c.cases().itertuples()}
    cache={}
    for p in (c.OLD/'prediction/zero_cache').glob('*.json'):
        v=c.document(p);cache[(v['fit_key'],v['held'])]=v
    rows=[];checks=[];errors=[]
    for request in plan['requests']:
        record=records[request['task_id']]
        solution=record['episodes'][0]['solutions'][0]
        assert len(solution['records'])==len(request['case_ids'])
        old=cache[(request['fit_key'],request['held'])]
        old_rows={r['case_id']:r for r in old['rows']}
        for case_id,p in zip(request['case_ids'],solution['records']):
            row=cases[case_id];old_row=old_rows[case_id]
            # A legacy invalid point is retained; group-wide refusal is not repaired.
            if p['numeric'] and old_row['numeric']:
                delta=abs(p['predicted']-old_row['predicted'])
                checks.append(delta<1e-9);errors.append(delta)
            elif p['numeric']!=old_row['numeric']:
                checks.append(False)
            q=dict(task_id=request['task_id'],fit_key=request['fit_key'],held=request['held'],
                method=request['method'],case_id=case_id,sample_id=row.sample_id,
                family=row.family,related_cluster=row.related_cluster,
                mobility_eligible=bool(row.mobility_eligible),observed=float(row.target),
                observed_log_mu=float(np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None,
                observed_log_sigma=float(np.log10(c.learning.E_CHARGE)+row.target+np.log10(row.mu_cm2Vs))
                    if row.mobility_eligible else None,
                old_nH_numeric=old_row['numeric'],old_nH_valid=old_row['valid'],**p)
            q.pop('diagnostics',None);q.pop('source_transport_derivative',None)
            rows.append(q)
    assert all(checks),('New zero readout differs from frozen nH',sum(not z for z in checks))
    c.e.csvwrite(c.OUT/'candidate_zero_predictions.csv',rows)
    c.write(c.OUT/'zero_replay_audit.json',dict(passed=all(checks),created_utc=c.e.now(),
        comparisons=len(checks),rows=len(rows),maximum_log_nH_difference=max(errors),
        all_candidate_parameters_unchanged=True,new_training=0,
        evidence_sha256=c.file_hashes([c.OUT/'candidate_zero_predictions.csv',
            c.OUT/'protocol_before_zero.json']),goal_complete=False))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();assert args.workers==4
    assert not (c.OUT/'zero_started.json').exists(),'Inspect original handles;do not restart'
    plan=make_plan();protocol=declare(plan);c.verify_hashes(protocol)
    c.write(c.OUT/'zero_plan.json',plan)
    c.write(c.OUT/'zero_started.json',dict(created_utc=c.e.now(),pid=os.getpid(),workers=args.workers))
    start=time.perf_counter();records,workers=run(plan,args.workers);rows=collate(plan,records)
    c.write(c.OUT/'zero_complete.json',dict(passed=True,created_utc=c.e.now(),
        tasks=len(records),rows=len(rows),seconds=time.perf_counter()-start,
        actual_physical_point_calls=sum(v['actual_physical_point_calls'] for v in records.values()),
        worker_seconds=sum(v['seconds'] for v in records.values()),
        new_optimizer_starts=0,goal_complete=False))
    print('Complete zero-observable re-evaluation finished',len(records),'tasks',len(rows),'rows',flush=True)


if __name__=='__main__':main()
