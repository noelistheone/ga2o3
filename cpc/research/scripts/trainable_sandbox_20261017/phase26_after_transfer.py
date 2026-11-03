"""Run frozen read-only analyses only after actual transfer and its audit finish."""
from pathlib import Path
import sys,json,time,argparse,subprocess

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--transfer-chain-pid',type=int,required=True);a=ap.parse_args()
    out=e.BASE/'phase26_pool_share';marker=out/'external_Azizie/audit.json';status=out/'after_transfer_status.json'
    e.write(status,dict(status='waiting_for_actual_transfer_audit',pid=a.transfer_chain_pid,created_utc=e.now(),goal_complete=False))
    while True:
        if marker.exists():
            try:
                if not json.loads(marker.read_text())['passed']:raise RuntimeError('Actual external transfer audit failed;inspect preserved results')
                break
            except json.JSONDecodeError:pass
        proc=Path(f'/proc/{a.transfer_chain_pid}/cmdline')
        if not proc.exists() or b'phase26_after_replay.py' not in proc.read_bytes():
            raise RuntimeError('Transfer chain ended without passed audit;inspect outputs before repair')
        time.sleep(5)
    for name,h in json.loads((out/'analysis_predeclared.json').read_text())['hashes'].items():assert e.sha(ROOT/name)==h,name
    for script in ('phase26_analysis.py','phase26_physical_scope.py','phase26_delivery_final_check.py'):
        e.write(status,dict(status='running_'+script,updated_utc=e.now(),goal_complete=False))
        subprocess.run([sys.executable,str(Path(__file__).parent/script)],cwd=ROOT,check=True)
    e.write(status,dict(status='analyses_complete_pending_actual_result_review_report_and_freeze',updated_utc=e.now(),goal_complete=False))
    print('phase26 actual analyses finished;goal remains active,review results before conclusion',flush=True)


if __name__=='__main__':main()
