"""Reproduce a frozen inverse-cache failure without changing the old solver."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.observation_adaptation import ObservationCoordinate,solve_effect
import phase20_source as source
s=source.s;OUT=s.OUT/'inverse_repair'


class DebugCoordinate(ObservationCoordinate):
    def evaluate(self,effect):
        try:return super().evaluate(effect)
        except ValueError as exc:
            if 'not ordered' not in str(exc):raise
            target=self.base[0]+float(effect)
            points=sorted((float(z),float(a[0][0])) for z,a in self.cache.items() if a is not None)
            below=[p for p in points if p[1]<=target];above=[p for p in points if p[1]>=target]
            lo=max(below,key=lambda p:p[0]);hi=min(above,key=lambda p:p[0]);nearest=min(points,key=lambda p:abs(p[1]-target))
            record=dict(effect=float(effect),target=float(target),lo=lo,hi=hi,nearest=nearest,
                nearest_observable_error=abs(nearest[1]-target),cache_points=len(points),
                maximum_sampled_drop=float(max(0.,-np.diff(np.array(points)[:,1]).min())),
                rows=points,error=repr(exc))
            s.energy.write(OUT/'cache_failure_snapshot.json',record);raise


def main():
    eps=[json.loads(p.read_text()) for p in (source.OUT/'episodes').glob('*.json')]
    failed=next(e for e in eps if e['source_kind']=='field' and e['budget']==2 and e['failures'])
    f=json.loads((s.OUT/'fits'/f"{failed['fit_key']}.json").read_text());d=source.table(failed['held'],2)
    aa=d[d.role.eq('anchor')].sort_values(['incorporated_cm3','sample_id']);y=aa.target.to_numpy(float)
    coordinate=DebugCoordinate(lambda z:s.predict(f,aa,source_kind='field',shift=z,require_observation=False));reproduced=False;errors=[]
    for prior in source.PRIORS:
        try:result=solve_effect(coordinate,y,[source.value(l) for l in source.LAM if l is not None],kind=prior)
        except ValueError as exc:
            errors.append(repr(exc));reproduced='not ordered' in str(exc);break
        # Reproduce the original intervening query-input evaluations. Query
        # labels do not enter the callback or this warm-state sequence.
        for op in result['solutions']:
            if op.get('valid'):s.predict(f,d,source_kind='field',shift=float(op['raw_shift']),require_observation=False)
    s.energy.write(OUT/'reproduction.json',dict(reproduced=reproduced,source_episode_key=failed['key'],fit_key=f['key'],
        source_episode_failures=failed['failures'],errors=errors,anchor_calls=coordinate.calls,
        code_sha256=s.energy.sha(Path(__file__)),old_code_unchanged=True))
    print('reproduced',reproduced,errors,flush=True)


if __name__=='__main__':main()
