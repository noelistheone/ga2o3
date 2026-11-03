"""Continue the fixed prediction workflow after the actual training job finishes."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import phase2_energy as e

OUT = e.BASE / 'phase27_charge_feasible'


def main():
    assert not (OUT / 'pipeline_started.json').exists(), 'Do not duplicate this workflow'
    start = json.loads((OUT / 'training_started.json').read_text())
    assert (OUT / 'pre_evaluation_protocol.json').exists()
    e.write(OUT / 'pipeline_started.json', dict(created_utc=e.now(), waits_for_training_pid=start['pid'],
        primary='FeasibleProcess_internal', no_training_restart=True, goal_complete=False))
    while not (OUT / 'training_complete.json').exists():
        try:
            argv = (Path('/proc') / str(start['pid']) / 'cmdline').read_bytes().split(b'\0')
            live = len(argv) > 1 and Path(argv[1].decode()).name == 'phase27_training.py'
        except (OSError, UnicodeDecodeError): live = False
        if not live:
            raise RuntimeError('Actual training stopped without completion;preserve its log and partial fits')
        time.sleep(45)
    stages = [('phase27_training_audit.py', []), ('phase27_prediction.py', ['--workers', '4']),
        ('phase27_prediction_audit.py', []), ('phase27_export.py', []), ('phase27_cohort_replay.py', []),
        ('phase27_external.py', []), ('phase27_external_audit.py', []), ('phase27_analysis.py', []),
        ('phase27_physical_scope.py', []), ('phase27_delivery_final_check.py', []),
        ('phase27_information_audit.py', [])]
    events = []
    for script, args in stages:
        event = dict(script=script, started_utc=e.now())
        e.write(OUT / 'pipeline_current.json', event)
        print('Starting actual stage', script, flush=True)
        result = subprocess.run([sys.executable, str(Path(__file__).parent / script), *args], cwd=ROOT)
        event.update(ended_utc=e.now(), exit_code=result.returncode); events.append(event)
        e.write(OUT / 'pipeline_events.json', events)
        if result.returncode:
            raise RuntimeError('Preserve failed stage and evidence before a versioned repair: ' + script)
    e.write(OUT / 'pipeline_complete.json', dict(created_utc=e.now(), events=events,
        primary='FeasibleProcess_internal', predicted_rows=9504, external_rows=5760,
        report_notes_PDF_visual_review_and_freeze_pending=True, goal_complete=False))
    print('Phase27 actual predictions/replays/transfers complete;read full results before conclusions;goal active', flush=True)


if __name__ == '__main__': main()
