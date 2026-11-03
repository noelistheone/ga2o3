"""Audit the actual chain-selection results and expose fallback-based recovery."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c


def main():
    target=c.OUT/'independent_result_audit.json'
    assert not target.exists()
    c.verify_hashes(c.document(c.OUT/'protocol_before_zero.json'))
    c.verify_hashes(c.document(c.OUT/'selection_protocol_before.json'))
    table=c.read(c.OUT/'predictions.csv');summary=c.read(c.OUT/'summary.csv')
    cases={r.case_id:r for r in c.cases().itertuples()}
    roles={(r.sample_id,int(r.budget)):r.role for r in c.roles().itertuples()}
    choices=c.document(c.OUT/'shared_selections.json')
    checks=[];identity=[]
    for row in table.itertuples():
        actual=cases[row.case_id]
        checks.extend([row.sample_id==actual.sample_id,row.family==actual.family,
            row.related_cluster==actual.related_cluster,row.role==roles[(row.sample_id,int(row.budget))],
            abs(row.observed-actual.target)<1e-12])
        if row.mobility_eligible:
            checks.append(abs(row.observed_log_mu-np.log10(actual.mu_cm2Vs))<1e-12)
        if row.numeric:
            diff=abs(row.predicted_log_sigma-(np.log10(c.learning.E_CHARGE)+row.predicted+row.predicted_log_mu))
            checks.append(diff<1e-10);identity.append(diff)
    for row in choices:
        if row['outer_held'] is not None:checks.append(row['outer_held'] not in row['train_units'])
        for candidate in row['candidates']:
            for ref in candidate['validation_fit_refs']:
                checks.extend([ref['held'] in row['train_units'],ref['held'] not in ref['train_units'],
                    set(ref['train_units'])<set(row['train_units']),c.e.sha(c.ROOT/ref['path'])==ref['sha256']])
        if row['policy']!='nH' and not row['selected']['fallback_reference']:
            checks.append(c.safe(row['selected']['metrics'],row['reference']['metrics'],row['policy']))
    common=table[table.role.eq('query') & table.primary_common20.eq(True)]
    for _,part in common.groupby(['variant','budget']):
        checks.append(part.sample_id.nunique()==20)
    checks.extend([len(table)==5733,table.variant.nunique()==39,len(choices)==234,
        c.document(c.OUT/'nH_policy_replay_audit.json')['passed']])
    old=c.document(c.OLD.parent/'artifact_manifest_phase29.json')
    hashes={v['path']:v['sha256'] for v in old['files']}
    cachefiles=[c.OLD/'prediction/plan.json',*sorted((c.OLD/'prediction/runtime').glob('worker[0-9].json'))]
    for path in cachefiles:checks.append(c.e.sha(path)==hashes[str(path.relative_to(c.ROOT))])
    main=[r for r in choices if r['variant']==c.PRIMARY and r['outer_held'] is not None]
    fallback=sum(r['selected']['fallback_reference'] for r in main)
    c.write(target,dict(passed=all(checks),created_utc=c.e.now(),checks=len(checks),
        max_Hall_conductivity_identity_difference=max(identity),
        primary_outer_groups=len(main),primary_reference_fallback_groups=fallback,
        main_transport_query_groups=int(common[common.mobility_eligible.eq(True)].related_cluster.nunique()),
        original_labels_roles_and_fixed_common20_verified=True,
        source_cache_matches_old_frozen_manifest=True,
        no_predictive_increment_established_by_audit=True,
        goal_complete=False,evidence_sha256=c.file_hashes([c.OUT/'predictions.csv',
            c.OUT/'summary.csv',c.OUT/'shared_selections.json',c.OUT/'source_choices.json'])))
    assert all(checks),sum(not q for q in checks)
    print('Result audit',len(checks),'passed;primary fallback',fallback,'/',len(main),flush=True)


if __name__=='__main__':main()
