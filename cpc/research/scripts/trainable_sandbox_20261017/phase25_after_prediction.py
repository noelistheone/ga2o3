"""Sequential result audit, frozen-model export and full API replay."""
from pathlib import Path
import sys,time,json,hashlib,argparse,subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/trainable_sandbox_20261017/phase25_shared_process'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--chain-pid',type=int,required=True);a=ap.parse_args()
    protocol=json.loads((OUT/'post_prediction_predeclared.json').read_text())
    def check():
        for name,h in protocol['hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    check();state=dict(chain_pid=a.chain_pid,status='waiting_for_actual_prediction',start_utc=time.time(),goal_complete=False)
    dest=OUT/'after_prediction_status.json';dest.write_text(json.dumps(state,indent=2))
    while True:
        marker=OUT/'prediction/execution_complete.json'
        if marker.exists():
            try:
                v=json.loads(marker.read_text())
                if v['prediction_rows']==3762:break
            except (json.JSONDecodeError,KeyError):pass
        p=Path(f'/proc/{a.chain_pid}/cmdline')
        if not p.exists() or b'phase25_after_training.py' not in p.read_bytes():
            raise RuntimeError('Prediction chain ended without its completion marker;inspect existing logs,do not restart automatically')
        time.sleep(15.)
    check()
    for script in ('phase25_prediction_audit.py','phase25_export.py','phase25_cohort_replay.py'):
        state.update(status='running_'+script,updated_utc=time.time());dest.write_text(json.dumps(state,indent=2))
        subprocess.run([sys.executable,str(ROOT/'scripts/trainable_sandbox_20261017'/script)],cwd=ROOT,check=True)
    state.update(status='prediction_audit_export_and_API_replay_finished_report_pending',updated_utc=time.time())
    dest.write_text(json.dumps(state,indent=2))
    print('phase25 post-prediction actual audit/export/replay finished;goal active',flush=True)


if __name__=='__main__':main()
