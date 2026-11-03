"""Run versioned exact-type repair replays before any subsequent transfer."""
from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e


def main():
    out=e.BASE/'phase25_shared_process/delivery/serialization_v2'
    paths=[ROOT/'src/sandbox/context_property_predictor_v2.py']+[Path(__file__).parent/n for n in
        ('phase25_api_runtime_v2.py','predict_context_properties_v2.py','phase25_cohort_replay_v2.py',
         'phase25_external_v2.py','phase25_external_audit_v2.py','phase25_label_sensitivity_v2.py','phase25_delivery_final_check.py')]
    p=out/'protocol_before_v2_replay.json';assert not p.exists()
    e.write(p,dict(created_utc=e.now(),repair='NumPy scalars/arrays to JSON native values only',
        primary_shared_fit_source_solve_and_scores_changed=False,original_replay_failure_retained=True,
        original_source='src/sandbox/context_property_predictor.py',source_sha256=e.sha(ROOT/'src/sandbox/context_property_predictor.py'),
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths},goal_complete=False))
    for script in ('phase25_cohort_replay_v2.py','phase25_external_v2.py','phase25_external_audit_v2.py',
                   'phase25_label_sensitivity_v2.py','phase25_delivery_final_check.py'):
        e.write(out/'status.json',dict(status='running',script=script,updated_utc=e.now(),goal_complete=False))
        subprocess.run([sys.executable,str(Path(__file__).parent/script)],cwd=ROOT,check=True)
    e.write(out/'status.json',dict(status='complete',updated_utc=e.now(),goal_complete=False))
    print('V2 replay/transfer/label-sensitivity/CLI chain complete;goal active',flush=True)


if __name__=='__main__':main()
