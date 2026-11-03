"""Wait on the verified training job, then audit and execute frozen predictions."""
from pathlib import Path
import sys,json,time,os,argparse,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e


def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-pid',type=int,required=True);a=ap.parse_args()
    out=e.BASE/'phase26_pool_share';status=out/'after_training_status.json'
    e.write(status,dict(status='waiting_for_actual_training',training_pid=a.training_pid,updated_utc=e.now(),goal_complete=False))
    while not (out/'training_complete.json').exists():
        if not live(a.training_pid):
            time.sleep(2)
            if not (out/'training_complete.json').exists():raise RuntimeError('Training process terminal without completion;inspect logs before any repair')
        time.sleep(5)
    done=json.loads((out/'training_complete.json').read_text());assert done['fit_records']==1444
    for script,args in [('phase26_training_audit.py',[]),('phase26_prediction.py',['--workers','4'])]:
        e.write(status,dict(status='running_'+script,training_pid=a.training_pid,updated_utc=e.now(),goal_complete=False))
        subprocess.run([sys.executable,str(Path(__file__).parent/script),*args],cwd=ROOT,check=True)
    e.write(status,dict(status='complete',updated_utc=e.now(),goal_complete=False))
    print('phase26 training audit and actual source predictions complete;goal active',flush=True)


if __name__=='__main__':main()
