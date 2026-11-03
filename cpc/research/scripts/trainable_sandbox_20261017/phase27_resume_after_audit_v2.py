"""Resume only after verified terminal audit failure;never retrain or re-predict."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import phase2_energy as e

OUT = e.BASE / 'phase27_charge_feasible'
STAGES = ('phase27_prediction_audit_v2.py', 'phase27_export.py', 'phase27_cohort_replay.py',
    'phase27_external.py', 'phase27_external_audit.py', 'phase27_analysis.py',
    'phase27_physical_scope.py', 'phase27_delivery_final_check.py', 'phase27_information_audit.py')


def main():
    assert not (OUT / 'pipeline_resume_started_v2.json').exists(), 'Do not duplicate the resumed chain'
    protocol = json.loads((OUT / 'prediction_audit_resume_protocol_v2.json').read_text())
    for kind in ('code_hashes', 'input_hashes'):
        for path, digest in protocol[kind].items(): assert e.sha(ROOT / path) == digest, path
    old = json.loads((OUT / 'pipeline_events.json').read_text())
    assert len(old) == 3 and old[-1]['script'] == 'phase27_prediction_audit.py' and old[-1]['exit_code'] != 0
    assert all(r['exit_code'] == 0 for r in old[:2])
    prior = json.loads((OUT / 'pipeline_started.json').read_text())
    old_jobs = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            if len(argv) > 1 and Path(argv[0].decode()).name.startswith('python') and Path(argv[1].decode()).name in ('phase27_after_training.py', 'phase27_prediction.py', 'phase27_prediction_audit.py'):
                old_jobs.append(proc.name)
        except (OSError, UnicodeDecodeError): pass
    assert not old_jobs, ('Original chain is still live;do not duplicate', old_jobs)
    e.write(OUT / 'pipeline_resume_started_v2.json', dict(created_utc=e.now(), original_pipeline_terminal_exit=1,
        original_training_pid=prior['waits_for_training_pid'], replacement_scope='audit missing-value comparison only',
        training_or_prediction_restarted=False, protocol_sha256=e.sha(OUT / 'prediction_audit_resume_protocol_v2.json'),
        goal_complete=False))
    events = []
    for script in STAGES:
        event = dict(script=script, started_utc=e.now())
        e.write(OUT / 'pipeline_resume_current_v2.json', event)
        print('Starting actual resumed stage', script, flush=True)
        run = subprocess.run([sys.executable, str(Path(__file__).with_name(script))], cwd=ROOT)
        event.update(ended_utc=e.now(), exit_code=run.returncode); events.append(event)
        e.write(OUT / 'pipeline_resume_events_v2.json', events)
        if run.returncode:
            raise RuntimeError('Preserve the resumed failure before another version: ' + script)
    complete = dict(created_utc=e.now(), events=old + events,
        successful_events=[r for r in old + events if r['exit_code'] == 0],
        failed_events=[r for r in old + events if r['exit_code'] != 0],
        original_failed_audit_preserved=True, original_training_and_predictions_reused_unchanged=True,
        resume_protocol_sha256=e.sha(OUT / 'prediction_audit_resume_protocol_v2.json'),
        primary='FeasibleProcess_internal', predicted_rows=9504, external_rows=5760,
        supplemental_lowdose_annotations_conclusions_report_PDF_and_freeze_pending=True, goal_complete=False)
    e.write(OUT / 'pipeline_resume_complete_v2.json', complete)
    e.write(OUT / 'pipeline_complete.json', complete)
    print('Actual resumed main pipeline complete;initial failure retained;additional delivery work pending', flush=True)


if __name__ == '__main__': main()
