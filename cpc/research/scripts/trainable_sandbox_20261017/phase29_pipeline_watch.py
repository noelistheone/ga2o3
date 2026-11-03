"""Continue the already authorized audit/prediction steps after live training."""
from pathlib import Path
import sys,json,os,time,subprocess

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e

OUT=e.BASE/'phase29_joint_scattering'


def check(protocol):
    for name,digest in protocol['bound_files'].items():assert e.sha(ROOT/name)==digest,name


def run(script,logname,args=()):
    logpath=OUT/logname
    assert not logpath.exists(),'inspect prior stage/session;never blindly rerun a failed writer'
    with logpath.open('w') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).with_name(script)),*args],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        e.write(OUT/(logname+'_started.json'),dict(created_utc=e.now(),pid=process.pid,script=script))
        code=process.wait()
    if code:raise RuntimeError('Preserve failed stage '+script+';see '+str(logpath))


def main():
    target=OUT/'pipeline_protocol_before_wait.json'
    assert not target.exists(),'inspect actual pipeline handle before a versioned resume'
    files=[Path(__file__),Path(__file__).with_name('phase29_training_audit.py'),
        Path(__file__).with_name('phase29_prediction.py'),Path(__file__).with_name('phase29_prediction_design.py'),
        Path(__file__).with_name('phase29_source_runtime.py'),ROOT/'src/sandbox/joint_saved_model.py',
        OUT/'training_protocol.json',OUT/'prediction_design_protocol.json']
    protocol=dict(created_utc=e.now(),pid=os.getpid(),continue_existing_training_only=True,
        no_training_restart=True,actual_global_goal_complete=False,
        bound_files={str(p.relative_to(ROOT)):e.sha(p) for p in files})
    e.write(target,protocol);started=json.loads((OUT/'training_started.json').read_text());check(protocol)
    print('Waiting for existing training PID',started['pid'],'pipeline PID',os.getpid(),flush=True)
    while not (OUT/'training_complete.json').exists():
        path=Path('/proc')/str(started['pid'])/'stat'
        if not path.exists() or path.read_text().split()[2]=='Z':
            raise RuntimeError('Training stopped without completion;preserve existing fits/logs,no blind restart')
        time.sleep(10.)
    check(protocol)
    if (OUT/'training_audit.json').exists():
        assert json.loads((OUT/'training_audit.json').read_text())['passed']
    else:run('phase29_training_audit.py','training_audit.log')
    check(protocol)
    if (OUT/'prediction/execution_complete.json').exists():
        print('Existing prediction completion preserved',flush=True)
    else:run('phase29_prediction.py','prediction_pipeline.log',('--workers','4'))
    check(protocol)
    e.write(OUT/'pipeline_audit_and_prediction_complete.json',dict(created_utc=e.now(),
        training_audit_passed=True,prediction_complete=True,
        prediction_protocol_sha256=e.sha(OUT/'prediction/protocol_before_prediction.json'),
        prediction_complete_sha256=e.sha(OUT/'prediction/execution_complete.json'),
        outstanding=['independent prediction audit/API replay','actual fitted-state physical scope audit',
            'full results/theory report/PDF','goal requirement audit and final artifact freeze'],
        overall_goal_complete=False))
    print('Phase29 audit/prediction complete;final independent audit and report pending',flush=True)


if __name__=='__main__':main()
