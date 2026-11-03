"""Wait on the actual source-prediction chain before mixed-model delivery."""
from pathlib import Path
import sys,json,time,os,argparse,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e


def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--chain-pid',type=int,required=True);a=ap.parse_args()
    out=e.BASE/'phase26_pool_share';dest=out/'prediction/execution_complete.json';status=out/'after_prediction_status.json'
    e.write(status,dict(status='waiting_for_actual_source_prediction',chain_pid=a.chain_pid,updated_utc=e.now(),goal_complete=False))
    while not dest.exists():
        if not live(a.chain_pid):
            time.sleep(2)
            if not dest.exists():raise RuntimeError('Source chain terminal without completed predictions;inspect logs')
        time.sleep(5)
    assert json.loads(dest.read_text())['prediction_rows']==3366
    for name,h in json.loads((out/'post_prediction_predeclared.json').read_text())['hashes'].items():assert e.sha(ROOT/name)==h,name
    for script in ('phase26_prediction_audit.py','phase26_export.py','phase26_cohort_replay.py'):
        e.write(status,dict(status='running_'+script,updated_utc=e.now(),goal_complete=False))
        subprocess.run([sys.executable,str(Path(__file__).parent/script)],cwd=ROOT,check=True)
    e.write(status,dict(status='complete',updated_utc=e.now(),goal_complete=False));print('phase26 mixed prediction delivery complete;goal active',flush=True)


if __name__=='__main__':main()
