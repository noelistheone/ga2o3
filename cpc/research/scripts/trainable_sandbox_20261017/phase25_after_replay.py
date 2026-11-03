"""Run frozen external transfer after actual saved-bundle replay has passed."""
from pathlib import Path
import sys,time,json,argparse,subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/trainable_sandbox_20261017/phase25_shared_process'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--replay-chain-pid',type=int,required=True);a=ap.parse_args()
    marker=OUT/'delivery/cohort_replay/audit.json';dest=OUT/'after_replay_status.json'
    state=dict(pid=a.replay_chain_pid,status='waiting_for_actual_API_replay',start_utc=time.time(),goal_complete=False)
    dest.write_text(json.dumps(state,indent=2))
    while True:
        if marker.exists():
            try:
                audit=json.loads(marker.read_text())
                if not audit['passed']:raise RuntimeError('API replay failed;preserve results and inspect')
                break
            except json.JSONDecodeError:pass
        proc=Path(f'/proc/{a.replay_chain_pid}/cmdline')
        if not proc.exists() or b'phase25_after_prediction.py' not in proc.read_bytes():
            raise RuntimeError('Replay chain ended without a passed audit;inspect current outputs')
        time.sleep(15.)
    for script in ('phase25_external.py','phase25_external_audit.py'):
        state.update(status='running_'+script,updated_utc=time.time());dest.write_text(json.dumps(state,indent=2))
        subprocess.run([sys.executable,str(ROOT/'scripts/trainable_sandbox_20261017'/script)],cwd=ROOT,check=True)
    state.update(status='transfer_finished_pending_synthesis_report_and_freeze',updated_utc=time.time());dest.write_text(json.dumps(state,indent=2))
    print('phase25 actual transfer and audit finished;goal active',flush=True)


if __name__=='__main__':main()
