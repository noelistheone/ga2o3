"""Freeze coverage/comparison/readback plans before new outer results exist."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e


def main():
    out=e.BASE/'phase26_pool_share';dest=out/'analysis_predeclared.json'
    paths=[Path(__file__)]+[Path(__file__).parent/n for n in ('phase26_analysis.py','phase26_physical_scope.py',
        'phase26_delivery_final_check.py','phase26_after_transfer.py','predict_pool_properties_v1.py')]
    paths+=[ROOT/'docs/ML_SANDBOX_GOAL_DIRECTION_RECONFIRM_20261018.md',
        ROOT/'docs/ML_SANDBOX_POOL_SHARE_ATTRIBUTION_ADDENDUM_20261018.md',out/'external_Azizie/protocol_before_transfer.json']
    value=dict(created_utc=e.now(),primary='ShareEffectiveProcess_internal',expected_all_method_metric_rows=432,
        comparisons='primary against every other35 methods x2schemes x3budgets on common20;DOI and relatedgroup macro plus completecoverage and leave-one-group sensitivity',
        physical_scope='sevennewphysicalfamilies x2fullsourcecontexts x33knownconditions;unqualifiedmodels retained and refused;not heldoutaccuracy',
        no_new_training_or_outer_method_selection=True,no_required_favorable_result=True,retrospective=True,goal_complete=False,
        hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths})
    if dest.exists():assert json.loads(dest.read_text())['hashes']==value['hashes']
    else:
        assert not (out/'prediction/execution_complete.json').exists()
        e.write(dest,value)
    print('Frozen allmethod analysis and physical readback plan',len(paths),'dependencies',flush=True)


if __name__=='__main__':main()
