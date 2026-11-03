"""Bind public delivery/replay checks before formal outer results exist."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e


def main():
    out=e.BASE/'phase26_pool_share';dest=out/'post_prediction_predeclared.json'
    assert json.loads((out/'interface_smoke/audit.json').read_text())['passed']
    code=[Path(__file__)]+[Path(__file__).parent/n for n in ('phase26_prediction_audit.py','phase26_export.py',
        'phase26_api_runtime.py','phase26_cohort_replay.py','predict_pool_properties_v1.py')]
    code+=[ROOT/'src/sandbox/pool_property_predictor.py',ROOT/'src/sandbox/pool_share_saved_model.py',
        ROOT/'src/sandbox/context_saved_model.py',ROOT/'src/sandbox/context_property_predictor_v2.py',out/'interface_smoke/audit.json']
    value=dict(created_utc=e.now(),scope='all36methods mixed-model public replay;newsourceheldout performance separate',
        expected_presets=468,expected_API_requests=1296,expected_point_rows=7128,
        checks='modelqualification,numeric/domainflags,allvalid3propertyvectors,invalidnumericcontinuations,readguard',
        raw_training_or_query_targets_in_model_bundle=False,retrospective=True,goal_complete=False,
        hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code})
    if dest.exists():assert json.loads(dest.read_text())['hashes']==value['hashes']
    else:e.write(dest,value)
    print('Frozen mixed-public delivery protocol',len(code),flush=True)


if __name__=='__main__':main()
