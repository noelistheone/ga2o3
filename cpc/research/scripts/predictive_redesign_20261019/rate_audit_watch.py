"""Audit each original prediction after its own existing watcher completes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,sys,time,subprocess
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c


def one(phase):
    name='phase31_rate_redesign' if phase==31 else 'phase32_rate_controls'
    out=c.BASE/name
    while not (out/'prediction/prediction_complete.json').exists():
        if (out/'after_training_failure.json').exists():
            raise RuntimeError(f'Original phase{phase} chain failed;preserve and inspect logs')
        time.sleep(15)
    logfile=out/'independent_audit.log'
    with logfile.open('x') as stream:
        child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('rate_result_audit.py')),
            '--phase',str(phase)],cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        c.write(out/'independent_audit_started.json',dict(pid=child.pid,created_utc=c.e.now()))
        status=child.wait()
    assert status==0,('Preserve actual audit failure',phase,status)
    return dict(phase=phase,status=status)


def main():
    target=c.BASE/'rate_audit_watch_started.json';assert not target.exists()
    audit=Path(__file__).with_name('rate_result_audit.py')
    c.write(target,dict(pid=os.getpid(),created_utc=c.e.now(),phases=[31,32],
        code_sha256=c.e.sha(__file__),audit_code_sha256=c.e.sha(audit),goal_complete=False))
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,[31,32]))
    assert c.document(target)['audit_code_sha256']==c.e.sha(audit)
    c.write(c.BASE/'rate_audit_watch_complete.json',dict(created_utc=c.e.now(),results=results,
        scientific_interpretation_pending=True,goal_complete=False))


if __name__=='__main__':main()
