"""Regression for equal cached roots and the reproduced real calibration case."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.observation_adaptation import ObservationCoordinate as Old
from sandbox.observation_adaptation_v2 import ObservationCoordinate,solve_effect
import phase20_source as source
s=source.s;OUT=s.OUT/'inverse_repair'


def main():
    def rounded(z):return np.array([round(17.+z,10)]),np.ones(1),np.ones(1,dtype=bool)
    old=Old(rounded);new=ObservationCoordinate(rounded)
    for c in (old,new):
        c._raw(.123456789012);c._raw(.123456789013)
    effect=rounded(.123456789012)[0][0]-17.
    old_failed=False
    try:old.evaluate(effect)
    except ValueError as exc:old_failed='not ordered' in str(exc)
    fixed=new.evaluate(effect);synthetic_error=abs(fixed['predictions'][0]-17.-effect)
    case=json.loads((OUT/'reproduction.json').read_text());f=json.loads((s.OUT/'fits'/f"{case['fit_key']}.json").read_text())
    original=json.loads((source.OUT/'episodes'/f"{case['source_episode_key']}.json").read_text())
    d=source.table(original['held'],2);aa=d[d.role.eq('anchor')].sort_values(['incorporated_cm3','sample_id']);y=aa.target.to_numpy(float)
    coordinate=ObservationCoordinate(lambda z:s.predict(f,aa,source_kind='field',shift=z,require_observation=False))
    all_results=[]
    for prior in source.PRIORS:
        result=solve_effect(coordinate,y,[source.value(l) for l in source.LAM if l is not None],kind=prior)
        all_results += [dict(prior=prior,**r) for r in result['solutions']]
        for op in result['solutions']:
            if op.get('valid'):s.predict(f,d,source_kind='field',shift=float(op['raw_shift']),require_observation=False)
    errors=[r.get('first_anchor_identity_error',np.inf) for r in all_results]
    payload=dict(passed=bool(old_failed and synthetic_error==0. and all(r['valid'] for r in all_results) and max(errors)<2e-6),
        old_flat_cache_failure_reproduced=old_failed,equal_root_exact_error=float(synthetic_error),
        real_source_episode=case['source_episode_key'],real_solutions=len(all_results),all_real_solutions_valid=all(r['valid'] for r in all_results),
        maximum_real_firstanchor_error=float(max(errors)),anchor_calls=coordinate.calls,
        formula_and_tolerance_unchanged=True,code_sha256=s.energy.sha(Path(__file__)),
        fixed_code_sha256=s.energy.sha(ROOT/'src/sandbox/observation_adaptation_v2.py'))
    s.energy.write(OUT/'regression_v2.json',payload);s.energy.write(OUT/'real_case_v2_solutions.json',all_results)
    print(json.dumps(s.energy.clean(payload),ensure_ascii=False),flush=True);assert payload['passed']


if __name__=='__main__':main()
