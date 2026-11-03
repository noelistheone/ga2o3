"""Re-evaluate the affected internal source branch with exact cache hits.

All learned shared weights/global selections and8unaffected linear-source
variants remain frozen. V1 source results and the failed inverse are retained.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,shutil,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.observation_adaptation_v2 import ObservationCoordinate,solve_effect
import phase20_source as base
s=base.s;PARENT=base.OUT;OUT=s.OUT/'source_adaptation_v2'
base.OUT=OUT;base.ObservationCoordinate=ObservationCoordinate;base.solve_effect=solve_effect


def freeze():
    regression=json.loads((s.OUT/'inverse_repair/regression_v2.json').read_text());assert regression['passed']
    gates=json.loads((s.OUT/'inverse_repair/learning_gates_v2.json').read_text());assert gates['passed']
    assert gates['code_hashes']['src/sandbox/observation_adaptation_v2.py']==s.energy.sha(ROOT/'src/sandbox/observation_adaptation_v2.py')
    payload=dict(created_utc=s.energy.now(),code_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in
        (Path(__file__),Path(base.__file__),ROOT/'src/sandbox/observation_adaptation_v2.py')},
        input_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in (PARENT/'source_protocol.json',PARENT/'global_choices.json',
            s.OUT/'prediction_protocol.json',s.OUT/'inverse_repair/regression_v2.json',s.OUT/'inverse_repair/learning_gates_v2.json')},
        repair='return an exactly cached firstanchor rootbefore strictbracketing;no tolerance/model/loss/hypergrid change',
        affected_variant='RelativeInventory',inherited_variants=[v for v in base.VARIANTS if v!='RelativeInventory'],
        frozen_shared_weights=True,global_selections_unchanged=True,retained_v1=str(PARENT.relative_to(ROOT)),
        prior='same2families7lambda andfirstanchorobservable coordinate',goal_complete=False)
    p=OUT/'source_protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==payload[k],k
    else:s.energy.write(p,payload)
    q=OUT/'global_choices.json'
    if q.exists():assert s.energy.sha(q)==s.energy.sha(PARENT/'global_choices.json')
    else:q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/'global_choices.json',q)


def worker(job):
    base.OUT=OUT;base.ObservationCoordinate=ObservationCoordinate;base.solve_effect=solve_effect
    return base.select(job)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();freeze()
    inherited=[]
    for variant in base.VARIANTS:
        if variant=='RelativeInventory':continue
        for name in ('predictions.csv','selections.json'):
            src=PARENT/'variants'/variant/name;dest=OUT/'variants'/variant/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert s.energy.sha(dest)==s.energy.sha(src)
            else:shutil.copyfile(src,dest)
            inherited.append(dict(path=str(src.relative_to(ROOT)),sha256=s.energy.sha(src),copy=str(dest.relative_to(ROOT))))
    s.energy.write(OUT/'inherited_results.json',inherited)
    jobs=[('RelativeInventory',held) for held in s.FAMILIES]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(worker,jobs):pass
    folder=OUT/'variants/RelativeInventory'
    pd.concat([s.read(folder/'folds'/s.digest(held)/'predictions.csv') for held in s.FAMILIES],ignore_index=True).to_csv(folder/'predictions.csv',index=False)
    choices=[r for held in s.FAMILIES for r in json.loads((folder/'folds'/s.digest(held)/'selections.json').read_text())]
    s.energy.write(folder/'selections.json',choices)
    rows=[]
    for p in (OUT/'episodes').glob('*.json'):
        e=json.loads(p.read_text());f=json.loads((s.OUT/'fits'/f"{e['fit_key']}.json").read_text())
        d=base.table(e['held'],e['budget']);aa=d[d.role.eq('anchor')].sort_values(['incorporated_cm3','sample_id'])
        if len(aa):
            _,J,valid=s.predict(f,aa,source_kind='field',shift=0.,require_observation=False)
            rows.append(dict(episode_key=e['key'],held=e['held'],budget=e['budget'],
                first_jacobian=float(J[0]),all_anchor_information=float(J@J),
                relative_information=float((J@J)/(J[0]*J[0])) if J[0]!=0 else None))
    s.energy.csvwrite(OUT/'anchor_sensitivity.csv',rows)
    s.energy.write(OUT/'execution_complete.json',dict(created_utc=s.energy.now(),variants=9,recomputed_variants=1,
        inherited_variants=8,original_weights_unchanged=True,goal_complete=False))


if __name__=='__main__':main()
