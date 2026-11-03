"""Verify frozen parents independently of running new-stage predictions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from freeze_phase21 import verify
import phase2_energy as e


def main():
    names=['protected_inputs_before.json']+[f'artifact_manifest_phase{i}.json' for i in
        (1,2,34,56,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)]
    counts={n:verify(e.BASE/n) for n in names}
    assert sum(counts.values())==47785
    assert e.sha(e.BASE/'artifact_manifest_phase24.json')=='7f33dcf124b336c902694f3b24280852dd2fc021b2bad955815bd4203fe4a5be'
    e.write(e.BASE/'phase25_shared_process/parent_verification.json',dict(passed=True,entries=sum(counts.values()),
        checked=counts,parent_sha256={n:e.sha(e.BASE/n) for n in names},created_utc=e.now(),
        changed_parent_files=0,goal_complete=False,code_sha256=e.sha(Path(__file__))))
    print('phase25 parents verified',sum(counts.values()),'entries',flush=True)


if __name__=='__main__':main()
