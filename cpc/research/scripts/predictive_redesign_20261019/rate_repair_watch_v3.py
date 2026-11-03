"""Preserve original pipeline failures, reaggregate, then run unchanged audits."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,sys,time,subprocess
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

OUT=c.BASE/'downstream_watch_transition_v3'


def command(args,path):
    with path.open('x') as stream:
        child=subprocess.Popen([sys.executable,*args],cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        c.write(path.with_suffix('.started.json'),dict(pid=child.pid,args=args,created_utc=c.e.now()))
        status=child.wait()
    assert status==0,('Preserve repair/audit failure',args,status)


def one(phase):
    name='phase31_rate_redesign' if phase=='31' else 'phase32_rate_controls'
    root=c.BASE/name
    while not (root/'after_training_failure.json').exists():
        if (root/'after_training_complete.json').exists():break
        started=c.document(root/'training_started.json')
        child=root/'after_training_child.json'
        if child.exists():pid=c.document(child)['pid'];pattern=f'phase{phase}_prediction.py'
        else:pid=started['pid'];pattern=f'phase{phase}_training.py'
        proc=Path('/proc')/str(pid)/'cmdline'
        if not proc.exists() or pattern.encode() not in proc.read_bytes():
            # Allow original watcher to write its terminal receipt first.
            time.sleep(2)
            if not (root/'after_training_failure.json').exists() and not (root/'after_training_complete.json').exists():
                watcher=c.document(root/'after_training_started.json')
                assert (Path('/proc')/str(watcher['pid'])/'cmdline').exists(),'Original chain stopped without terminal receipt'
        time.sleep(15)
    if not (root/'prediction/prediction_complete.json').exists():
        command([str(Path(__file__).with_name('rate_source_aggregation_repair_v3.py')),'--phase',phase],OUT/f'phase{phase}_repair.log')
    command([str(Path(__file__).with_name('rate_result_audit.py')),'--phase',phase],OUT/f'phase{phase}_audit.log')
    assert c.document(root/'prediction/independent_result_audit.json')['passed']
    c.write(OUT/f'phase{phase}_complete.json',dict(created_utc=c.e.now(),
        original_failure_preserved=True,independent_audit_passed=True,goal_complete=False))
    return phase


def main():
    target=OUT/'repair_watcher_started.json';assert not target.exists()
    files=[Path(__file__),Path(__file__).with_name('rate_source_aggregation_repair_v3.py'),
           Path(__file__).with_name('rate_result_audit.py')]
    c.write(target,dict(pid=os.getpid(),created_utc=c.e.now(),code_hashes=c.file_hashes(files),
        stop_receipt_sha256=c.e.sha(OUT/'stop_receipt.json'),phases=['31','32'],goal_complete=False))
    with ThreadPoolExecutor(max_workers=2) as pool:phases=list(pool.map(one,['31','32']))
    c.verify_hashes(c.document(target))
    c.write(OUT/'repair_watcher_complete.json',dict(created_utc=c.e.now(),phases=phases,
        original_training_and_prediction_failures_preserved=True,goal_complete=False))


if __name__=='__main__':main()
