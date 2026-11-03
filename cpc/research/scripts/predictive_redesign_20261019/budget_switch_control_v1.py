"""Strong task-specific baseline: shared physics at k0, unit-dose anchors at k1/2."""
from pathlib import Path
import sys,itertools,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

OUT=c.BASE/'proposal_followup_20261021_v1/budget_switch_control'


def main():
    assert not (OUT/'protocol_before.json').exists()
    frozen=c.BASE/'proposal_followup_20261021_v1/profiled_adaptation/predictions.csv'
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),
        rule='budget0 frozenJointProcess_nHonly;budget1/2 unitdose+unshrunk mean anchoractivationoffset',
        baseline_not_new_internal_learning=True,rule_justification='user explicitly proposed separating prediction and calibration tasks;no queryselectedswitch',
        source_selection='none;fixed byavailableanchorbudget',retrospective=True,
        code_hashes=c.file_hashes([Path(__file__),Path(c.__file__)]),input_hashes=c.file_hashes([frozen,c.OLD/'data/scenarios49.csv',c.OLD/'data/episodes35.csv'])))
    table=c.cases();roles=c.roles();old=c.read(frozen);rows=[]
    zero={r.case_id:r.predicted for r in old[old.variant.eq('FrozenShared')&old.budget.eq(0)].itertuples()}
    for budget in (0,1,2):
        for family,source in table.groupby('family',sort=True):
            ep=roles[roles.family.eq(family)&roles.budget.eq(budget)];rolemap=dict(zip(ep.sample_id,ep.role))
            anchors=sorted(ep.loc[ep.role.eq('anchor'),'sample_id']);assert len(anchors)==budget
            offsets=[]
            for name in anchors:
                aa=source[source.sample_id.eq(name)]
                offsets.append(float(aa.target.iloc[0]-np.log10(aa.incorporated_cm3.to_numpy(float)).mean()))
            delta=float(np.mean(offsets)) if offsets else 0.
            for row in source.itertuples():
                for variant in ['BudgetSwitch_baseline','UnitDoseUnshrunk_baseline']:
                    y=zero[row.case_id] if budget==0 and variant=='BudgetSwitch_baseline' else np.log10(row.incorporated_cm3)+delta
                    rows.append(dict(variant=variant,budget=budget,case_id=row.case_id,sample_id=row.sample_id,family=family,
                        related_cluster=row.related_cluster,scenario_index=row.scenario_index,role=rolemap[row.sample_id],
                        primary_common20=bool(row.primary_common20),observed=float(row.target),predicted=float(y),numeric=True,valid=True,
                        mobility_eligible=False,source_anchor_ids=anchors,delta_dex=delta))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'predictions.csv',index=False);sums=[]
    for variant,budget,scope in itertools.product(frame.variant.unique(),(0,1,2),('all_query','common20')):
        q=frame[frame.variant.eq(variant)&frame.budget.eq(budget)&frame.role.eq('query')]
        if scope=='common20':q=q[q.primary_common20.eq(True)]
        sums.append(dict(variant=variant,budget=budget,scope=scope,conditions=q.sample_id.nunique(),
            nH_MAE=c.group_score(q.to_dict('records'),'nH',group_equal=False)))
    pd.DataFrame(sums).to_csv(OUT/'summary.csv',index=False)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'complete.json',dict(created_utc=c.e.now(),rows=len(frame),new_shared_fits=0,new_physical_calls=0,
        no_monotonicity_or_full_chain_claim=True,baseline_not_proof_new_architecture=True))
    print([s for s in sums if s['scope']=='common20'])


if __name__=='__main__':main()
