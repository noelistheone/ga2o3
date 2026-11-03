"""Freeze the source/group comparison before any new outer predictions."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.kroger_db import NPZ
import phase26_prediction as p
import phase2_energy as e


def main():
    out=p.t.OUT;dest=out/'pre_evaluation_protocol.json'
    for name in ('source_gates.json','source_smoke/audit.json'):
        r=json.loads((out/name).read_text());assert r['passed']
        for path,h in r['code_hashes'].items():assert e.sha(ROOT/path)==h,path
    parent=json.loads((p.PARENT/'prediction/protocol.json').read_text())
    assert parent['input_hashes'][str(NPZ.relative_to(ROOT))]==e.sha(NPZ)
    paths=[Path(__file__),Path(__file__).parent/'phase26_prediction.py',Path(__file__).parent/'phase26_registry.py',
        Path(__file__).parent/'phase26_source_runtime.py',Path(__file__).parent/'phase26_training_audit.py',
        ROOT/'src/sandbox/pool_share_saved_model.py',ROOT/'src/sandbox/pool_share_model.py',ROOT/'src/sandbox/pool_share_controls.py',
        ROOT/'src/sandbox/absolute_pool_share.py',ROOT/'src/sandbox/context_source.py',ROOT/'src/sandbox/functional_source.py',
        ROOT/'docs/ML_SANDBOX_POOL_SHARE_ATTRIBUTION_ADDENDUM_20261018.md',
        out/'training_protocol.json',out/'source_gates.json',out/'source_smoke/audit.json',
        p.PARENT/'prediction/predictions.csv',p.PARENT/'prediction/summary.csv',
        p.PARENT/'data/conditions33.csv',p.PARENT/'data/episodes33.csv',NPZ]
    value=dict(created_utc=e.now(),primary='ShareEffectiveProcess_internal',variants=p.VARIANTS,
        new_variants=17,old_variants=19,combined_variants=36,new_rows=3366,combined_rows=7128,
        new_shared_choices=143,new_plans=221,new_source_selections=663,new_summary_rows=204,combined_summary_rows=432,
        selection='only corresponding training-inner labels;no outer queries used',
        episode_point_budget=32768,source_search='33cubicnodes+atmost3local searches80iterations;sampled zero-adjacent validbranch',
        qualification='invalid/unconverged training fits retained,unservable;no selective coverage MAE',
        cohort='unchanged33conditions/6DOI/5relatedgroups;old29/roles preserved;common20 and allquery both reported',
        metrics='DOI/group macro and worstDOI;0/1/2 perDOI,related2DOIgroup0/2/4total',
        attribution='same physical state has same source derivatives;matchedrho0/1 attribution not yet performed',
        retrospective=True,goal_complete=False,hashes={str(path.relative_to(ROOT)):e.sha(path) for path in paths})
    if dest.exists():
        old=json.loads(dest.read_text());assert old['hashes']==value['hashes']
        print('Verified existing phase26 source protocol',len(value['hashes']),flush=True)
    else:e.write(dest,value);print('Frozen phase26 source protocol',len(value['hashes']),flush=True)


if __name__=='__main__':main()
