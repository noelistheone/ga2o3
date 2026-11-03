"""Run independent audits and actual API replay after the live prediction chain."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,os,time,subprocess

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e
import phase29_data as data
import phase29_delivery as delivery

OUT=data.OUT


def check(protocol):
    for name,digest in protocol['bound_files'].items():assert e.sha(ROOT/name)==digest,name


def run(script,logname,target,args=()):
    destination=OUT/target
    if destination.exists():
        assert json.loads(destination.read_text())['passed'];return dict(script=script,existing=True)
    logpath=OUT/logname
    assert not logpath.exists(),'inspect prior child handle/log before any versioned resume'
    with logpath.open('w') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name(script)),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        e.write(OUT/(logname+'_started.json'),dict(created_utc=e.now(),pid=proc.pid,script=script))
        code=proc.wait()
    e.write(OUT/(logname+'_finished.json'),dict(created_utc=e.now(),script=script,exit_code=code))
    if code:raise RuntimeError('Preserve failed independent stage '+script)
    assert json.loads(destination.read_text())['passed']
    return dict(script=script,exit_code=code)


def main():
    target=OUT/'post_prediction_protocol_before_wait.json'
    assert not target.exists(),'inspect actual post-prediction watcher before any versioned resume'
    files=[Path(__file__),*[Path(__file__).with_name(s) for s in (
        'phase29_prediction_audit.py','phase29_delivery.py','phase29_api_replay.py',
        'phase29_api_runtime.py','phase29_physical_scope.py','phase29_information_audit.py')],
        *delivery.core_files(),ROOT/'src/sandbox/transport_scope_diagnostic.py',
        ROOT/'docs/ML_SANDBOX_TRANSPORT_SCOPE_PHASE29_20261019.md',
        ROOT/'docs/ML_SANDBOX_SCENARIO_INFORMATION_PHASE29_20261019.md']
    protocol=dict(created_utc=e.now(),pid=os.getpid(),waiting_for_existing_pipeline_pid=2950397,
        bound_files={str(p.relative_to(ROOT)):e.sha(p) for p in files},
        no_model_retraining_or_prediction_changes=True,overall_goal_complete=False)
    e.write(target,protocol);print('Post-prediction watcher PID',os.getpid(),'waits for existing pipeline2950397',flush=True)
    while not (OUT/'pipeline_audit_and_prediction_complete.json').exists():
        path=Path('/proc/2950397/stat')
        if not path.exists() or path.read_text().split()[2]=='Z':
            raise RuntimeError('Original pipeline stopped without completion;inspect original errors,no blind restart')
        time.sleep(10.)
    check(protocol)
    run('phase29_prediction_audit.py','prediction_independent_audit.log','prediction/audit.json')
    check(protocol)
    run('phase29_delivery.py','delivery_export.log','delivery/audit.json')
    check(protocol)
    tasks=[('phase29_information_audit.py','information_audit.log','information_audit/audit.json',()),
        ('phase29_physical_scope.py','physical_scope.log','physical_scope/audit.json',()),
        ('phase29_api_replay.py','api_replay.log','api_replay/audit.json',('--workers','4'))]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,*args) for args in tasks]
        results=[f.result() for f in futures]
    check(protocol)
    e.write(OUT/'post_prediction_independent_checks_complete.json',dict(created_utc=e.now(),passed=True,
        stages=results,report_PDF_and_goal_requirement_audit_pending=True,
        outstanding=['all-results/theory report andPDF','independent rendered/numeric report audit','full goal audit and artifact freeze'],
        overall_goal_complete=False))
    print('Independent prediction/delivery/information/physics/API stages passed;report and full audit pending',flush=True)


if __name__=='__main__':main()
