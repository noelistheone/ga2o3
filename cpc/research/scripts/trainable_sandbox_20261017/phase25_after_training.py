"""Run the already authorized audits/predictions after the active fit job ends."""
from pathlib import Path
import os,sys,time,json,hashlib,argparse,subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/trainable_sandbox_20261017/phase25_shared_process'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-pid',type=int,required=True);a=ap.parse_args()
    prereg=json.loads((OUT/'pre_evaluation_protocol.json').read_text())
    def check():
        for name,h in prereg['hashes'].items():assert digest(ROOT/name)==h,name
    check()
    state=dict(training_pid=a.training_pid,status='waiting_for_current_training',start_utc=time.time(),goal_complete=False)
    (OUT/'after_training_status.json').write_text(json.dumps(state,indent=2))
    while True:
        p=OUT/'training_complete.json'
        if p.exists():
            try:
                d=json.loads(p.read_text())
                if d['fit_records']==1596:break
            except (json.JSONDecodeError,KeyError):pass
        proc=Path(f'/proc/{a.training_pid}/cmdline')
        if not proc.exists() or b'phase25_training.py' not in proc.read_bytes():
            raise RuntimeError('Training parent ended without the expected completion marker; preserve logs and inspect')
        time.sleep(15.)
    check()
    for script in ('phase25_training_audit.py','phase25_prediction.py'):
        state.update(status='running_'+script,updated_utc=time.time())
        (OUT/'after_training_status.json').write_text(json.dumps(state,indent=2))
        command=[sys.executable,str(ROOT/'scripts/trainable_sandbox_20261017'/script)]
        if script=='phase25_prediction.py':command+=['--workers','4']
        subprocess.run(command,cwd=ROOT,check=True)
    state.update(status='prediction_finished_pending_result_audit_and_report',updated_utc=time.time())
    (OUT/'after_training_status.json').write_text(json.dumps(state,indent=2))
    print('phase25 sequential training audit and actual source prediction complete;goal active',flush=True)


if __name__=='__main__':main()
