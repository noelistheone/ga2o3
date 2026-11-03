"""Freeze and train additional fair nonlinear controls on the current cohort."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,os,sys,time,subprocess,sklearn
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import strong_tree_learning as l


def declare():
    p=l.OUT/'protocol_before_training.json';assert not p.exists()
    files=[Path(__file__),Path(l.__file__),Path(__file__).with_name('strong_tree_saved.py'),
        Path(__file__).with_name('strong_tree_source_runtime.py'),Path(__file__).with_name('strong_tree_prediction.py'),
        Path(c.__file__),c.OLD_SCRIPTS/'phase29_learning.py',c.OLD_SCRIPTS/'phase29_prediction_design.py',
        c.OLD_SCRIPTS/'phase29_source_runtime.py',c.ROOT/'src/sandbox/context_saved_model.py',
        Path(__file__).with_name('strong_tree_gates.py')]
    files+=list((c.ROOT/'src/sandbox').glob('*.py'))
    inputs=[c.OLD/'data/conditions35.csv',c.OLD/'data/scenarios49.csv',c.OLD/'data/episodes35.csv',
        c.OUT/'candidate_zero_predictions.csv',c.OLD/'prediction/predictions.csv',c.OUT/'predictions.csv',
        l.OUT/'gates_v1/gates.json']
    assert c.document(l.OUT/'gates_v1/gates.json')['passed']
    c.write(p,dict(created_utc=c.e.now(),methods=list(l.METHODS),configs=l.configs(),
        contexts=[list(t) for t in c.learning.contexts()],planned_fit_records=288,
        n_estimators=l.N_ESTIMATORS,random_seed=l.SEED,rho_seed=l.SEED+1,sklearn_version=sklearn.__version__,
        reason='new primary-source review found process-feature nonlinear ML; previous tree control used old29cohort,notcurrent35fullchain',
        comparator_only=True,no_new_Sandbox_accuracy_claim=True,
        same10_input_features_as_original_DirectMultiRidge=True,
        growth_rate_not_supplied_no_replication_of_Chou2022_accuracy_claim=True,
        same_scenario_condition_DOI_relatedgroup_weights_as_current_models=True,
        residual_baseline='logC for nH;primitive rho from unitdose+CT for rho;no measured query nH',
        scenarios_not_independent_leaf_count_is_algorithmic_not_physical_sample_count=True,
        candidate_budget=9,source_lambdas=list(c.design.LAM),
        model_selection='zeroanchor groupmacro innerCV nH;both controls use nH-only selection and fullchain reporting',
        source_selection='same traininginner queryDOImacro nH;no outerlabels or querymu;no transport gate for direct controls',
        source_update='Halloffset only;paired rhohead unchanged;unpaired CT recalculated',
        all_unfavorable_results_preserved=True,independent_blind_validation=False,
        main_phase31_primary_and_bound_A_B_pipelines_unchanged=True,
        code_hashes=c.file_hashes(files),input_hashes=c.file_hashes(inputs),goal_complete=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();assert args.workers==2
    declare();c.verify_hashes(c.document(l.OUT/'protocol_before_training.json'))
    import strong_tree_prediction as prediction
    prediction.declare()
    c.write(l.OUT/'training_started.json',dict(pid=os.getpid(),workers=args.workers,created_utc=c.e.now()))
    tasks=[(t,m,q) for t in c.learning.contexts() for m in l.METHODS for q in l.configs()]
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:records=list(pool.map(l.fit,tasks))
    assert len(records)==288 and all(r['training_valid'] for r in records)
    c.verify_hashes(c.document(l.OUT/'protocol_before_training.json'))
    c.write(l.OUT/'training_complete.json',dict(created_utc=c.e.now(),fit_records=len(records),
        forest_fits=sum(r['forests'] for r in records),seconds=time.monotonic()-start,
        maximum_serialized_replay_error=max(r['serialization_error'] for r in records),goal_complete=False))
    print('Actual current-cohort nonlinear controls trained',len(records),flush=True)
    with (l.OUT/'prediction.log').open('x') as stream:
        process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('strong_tree_prediction.py')),
            'run','--workers','2'],cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        c.write(l.OUT/'prediction_child.json',dict(pid=process.pid,created_utc=c.e.now()))
        status=process.wait()
    c.write(l.OUT/'pipeline_status.json',dict(prediction_status=status,created_utc=c.e.now(),
        independent_result_audit_pending=True,goal_complete=False))
    assert status==0,'Preserve original tree prediction failure'


if __name__=='__main__':main()
