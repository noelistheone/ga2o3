"""Use the unchanged V2 external cohort after versioned main-output repair."""
from pathlib import Path
import os,sys,time,subprocess
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

OUT=c.BASE/'external_rate_cohort_v2/watch_v3'


def command(args,name):
    path=OUT/name
    with path.open('x') as stream:
        child=subprocess.Popen([sys.executable,*args],cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        c.write(path.with_suffix('.started.json'),dict(pid=child.pid,args=args,created_utc=c.e.now()))
        status=child.wait()
    assert status==0,('Preserve actual external failure',args,status)


def main():
    target=OUT/'watcher_started.json';assert not target.exists()
    design=c.BASE/'external_rate_cohort_v2/prediction/design_before_prediction.json'
    c.verify_hashes(c.document(design))
    repair=c.BASE/'downstream_watch_transition_v3/repair_watcher_started.json'
    assert repair.exists()
    c.write(target,dict(created_utc=c.e.now(),pid=os.getpid(),code_sha256=c.e.sha(__file__),
        unchanged_V2_external_design_sha256=c.e.sha(design),
        repair_watcher_sha256=c.e.sha(repair),goal_complete=False))
    try:
        for name in ['phase31_rate_redesign','phase32_rate_controls']:
            root=c.BASE/name
            while not (root/'prediction/independent_result_audit.json').exists():
                watcher=c.document(repair);proc=Path('/proc')/str(watcher['pid'])/'cmdline'
                assert proc.exists() and b'rate_repair_watch_v3.py' in proc.read_bytes(),'Inspect versioned main continuation failure'
                time.sleep(15)
            assert c.document(root/'prediction/independent_result_audit.json')['passed']
        for phase in ['31','32']:
            command([str(Path(__file__).with_name('disorder_linearization_check.py')),'--phase',phase],f'linearization_phase{phase}.log')
        command([str(Path(__file__).with_name('external_rate_prediction_v2.py')),'run'],'prediction.log')
        c.write(OUT/'watcher_complete.json',dict(created_utc=c.e.now(),
            actual_prediction_done=True,scientific_interpretation_pending=True,goal_complete=False))
    except BaseException as exc:
        c.write(OUT/'watcher_failure.json',dict(created_utc=c.e.now(),error=repr(exc),goal_complete=False));raise


if __name__=='__main__':main()
