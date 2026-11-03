"""Independently verify every frozen parent entry without rewriting parents."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from freeze_phase21 import verify
import phase2_energy as e


def main():
    names = ['protected_inputs_before.json'] + [f'artifact_manifest_phase{i}.json' for i in
        (1, 2, 34, 56, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)]
    counts = {name: verify(e.BASE / name) for name in names}
    assert sum(counts.values()) == 52718
    assert e.sha(e.BASE / 'artifact_manifest_phase26.json') == '4489891819c035c45b3bae9c128f6a9e79b6f63cd79742c755d3a42eaacc6bcc'
    e.write(e.BASE / 'phase27_charge_feasible/parent_verification_working.json', dict(passed=True,
        created_utc=e.now(), entries=sum(counts.values()), counts=counts, changed_parent_files=0,
        parent_sha256={name: e.sha(e.BASE / name) for name in names}, goal_complete=False,
        code_sha256=e.sha(Path(__file__))))
    print('Verified actual phase27 parent entries', sum(counts.values()), flush=True)


if __name__ == '__main__': main()
