"""Continue successful original pipelines into frozen conditional external data."""
from pathlib import Path
import os,sys,time,subprocess
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c


def command(args,logfile):
    with logfile.open('x') as stream:
        child=subprocess.Popen([sys.executable,*args],cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        c.write(logfile.with_suffix('.started.json'),dict(created_utc=c.e.now(),pid=child.pid,args=args))
        status=child.wait()
    assert status==0,(args,status)


def main():
    out=c.BASE/'external_rate_cohort_v2';target=out/'watcher_started.json';assert not target.exists()
    c.write(target,dict(created_utc=c.e.now(),pid=os.getpid(),code_sha256=c.e.sha(__file__),
        design_sha256=c.e.sha(out/'prediction/design_before_prediction.json'),goal_complete=False))
    try:
        for name in ['phase31_rate_redesign','phase32_rate_controls']:
            root=c.BASE/name
            while not (root/'prediction/independent_result_audit.json').exists():
                if (root/'after_training_failure.json').exists():
                    raise RuntimeError('Original main chain failed;inspect preserved '+name)
                auditwatch=c.document(c.BASE/'rate_audit_watch_started.json')
                process=Path('/proc')/str(auditwatch['pid'])/'cmdline'
                if not process.exists() or b'rate_audit_watch.py' not in process.read_bytes():
                    raise RuntimeError('Original audit watcher stopped without required results')
                time.sleep(15)
            assert c.document(root/'prediction/independent_result_audit.json')['passed']
        for phase in ['31','32']:
            command([str(Path(__file__).with_name('disorder_linearization_check.py')),'--phase',phase],out/f'linearization_phase{phase}.log')
        command([str(Path(__file__).with_name('external_rate_prediction_v2.py')),'run'],out/'prediction.log')
        c.write(out/'watcher_complete.json',dict(created_utc=c.e.now(),
            actual_prediction_done=True,scientific_interpretation_pending=True,goal_complete=False))
    except BaseException as exc:
        c.write(out/'watcher_failure.json',dict(created_utc=c.e.now(),error=repr(exc),goal_complete=False));raise


if __name__=='__main__':main()
