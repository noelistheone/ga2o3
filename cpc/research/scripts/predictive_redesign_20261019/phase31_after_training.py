"""Continue the original live training into the already frozen evaluation."""
from pathlib import Path
import os, sys, time, subprocess
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase31_learning as l


def main():
    target=l.OUT/'after_training_started.json'
    assert not target.exists(),'Inspect original watcher;never duplicate on timeout'
    original=c.document(l.OUT/'training_started.json')
    design=l.OUT/'prediction/design_before_prediction.json'
    c.verify_hashes(c.document(design))
    c.write(target,dict(pid=os.getpid(),training_pid=original['pid'],
        created_utc=c.e.now(),code_sha256=c.e.sha(__file__),
        design_sha256=c.e.sha(design),goal_complete=False))
    try:
        while not (l.OUT/'training_complete.json').exists():
            cmd=Path('/proc')/str(original['pid'])/'cmdline'
            if not cmd.exists() or b'phase31_training.py' not in cmd.read_bytes():
                raise RuntimeError('Original training stopped without completion;inspect preserved logs')
            time.sleep(15)
        with (l.OUT/'prediction.log').open('x') as stream:
            process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('phase31_prediction.py')),
                'run','--workers','4'],cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT)
            c.write(l.OUT/'after_training_child.json',dict(pid=process.pid,
                created_utc=c.e.now(),action='actual nested selection and0/1/2predictions'))
            status=process.wait()
        assert status==0,('Prediction failed;preserve before versioned repair',status)
        c.write(l.OUT/'after_training_complete.json',dict(created_utc=c.e.now(),
            prediction_terminal_status=status,independent_audit_pending=True,goal_complete=False))
    except BaseException as exc:
        c.write(l.OUT/'after_training_failure.json',dict(created_utc=c.e.now(),error=repr(exc),goal_complete=False))
        raise


if __name__=='__main__':main()
