"""Audit the added matched controls, schema-only replay and grid refinement."""
from pathlib import Path
import sys,json,hashlib,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from audit_proposal_followup_v1 import macro,close

BASE=c.BASE/'proposal_followup_20261021_v1'
OUT=BASE/'additional_independent_audit_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    refined=BASE/'constrained_block_refinement_v2';controls=BASE/'overlay_nH_controls_v1';external=controls/'external_schema_replay_v2'
    for p in [refined,controls,external]:assert (p/'complete.json').exists();c.verify_hashes(c.document(p/'protocol_before.json'))
    files=[p/'summary.csv' for p in [refined,controls,external]]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes([Path(__file__)]),input_hashes=c.file_hashes(files)))
    checks=[]
    def check(name,passed,**detail):checks.append(dict(name=name,passed=bool(passed),**detail))
    for base in [refined,controls,external]:
        p=c.read(base/'predictions.csv');s=c.read(base/'summary.csv')
        for row in s.itertuples():
            q=p[p.variant.eq(row.variant)]
            if hasattr(row,'cohort'):q=q[q.cohort.eq(row.cohort)]
            if row.scope=='common20':q=q[q.primary_common20.eq(True)]
            if base==controls and getattr(row,'cohort','')=='external3_conditional':
                check('retained_initial_external_schema_refusal',q.numeric.eq(False).all(),rows=len(q));continue
            for endpoint,pred,obs in [('nH','predicted','observed'),('mu','predicted_log_mu','observed_log_mu'),('rho','predicted_log_sigma','observed_log_sigma')]:
                score=macro(q,pred,obs,endpoint!='nH')
                check(base.name+'_metric_'+endpoint+'_'+str(row.Index),close(score,getattr(row,endpoint+'_MAE')))
        q=p[p.numeric.eq(True)]
        if len(q):
            err=float(max(abs(q.predicted_log_sigma-q.predicted-q.predicted_log_mu-np.log10(c.learning.E_CHARGE))))
            check(base.name+'_Hall_identity',err<5e-10,max_error=err)
    exact_grid={tuple(x) for x in c.document(refined/'protocol_before.json')['grid']}
    probes=0
    for f in (refined/'records').glob('*.json'):
        j=c.document(f);probes+=len(j['probes'])
        check('grid_complete_'+f.stem,{tuple(x['beta']) for x in j['probes']}==exact_grid and len(j['probes'])==25)
        parent=c.document(c.ROOT/j['parent_path']);primary=c.document(c.ROOT/parent['primary_path'])
        check('grid_group_exclusion_'+f.stem,j['held'] not in primary['train_units'])
        if j['selected']:
            check('grid_feasible_'+f.stem,j['training_valid'] and j['min_constraint']>=-1e-7
                and j['training_max_nH_drift']<=.03000011 and j['training_nH_loss_change']<=.00010011)
            check('grid_not_worse_training_'+f.stem,j['selected']['objective']<=parent['selected']['objective']+1e-7)
    check('all_375_grid_nodes',probes==375,total=probes)
    fitrecords=list((controls/'fits').glob('*.json'));check('seven_matched_control_fits',len(fitrecords)==7)
    for f in fitrecords:
        j=c.document(f);check('control_fit_qualified_'+f.stem,j['training_valid'] and j['selected_success'] and j['config']['gamma']==0.)
        if j['held'] is not None:check('control_fit_exclusion_'+f.stem,j['held'] not in j['train_units'])
    src=c.read(c.BASE/'external_rate_cohort_v2/scenarios24.csv');p=c.read(external/'predictions.csv');lookup=src.set_index('case_id')
    check('external_complete',len(p)==48 and p.numeric.eq(True).all() and p.valid.eq(True).all())
    for row in p.itertuples():
        raw=lookup.loc[row.case_id];check('external_labels_'+str(row.Index),row.sample_id==raw.sample_id and abs(row.observed-raw.target)<1e-12)
    historical=c.document(c.ROOT.parent/'research_progress_20261021/evidence_index.json')
    for e in historical['entries']:
        p=c.ROOT.parent/e['path'];check('historical_unchanged_'+e['id'],hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'])
    p=c.ROOT.parent/historical['document']['path'];check('historical_report_unchanged',hashlib.sha256(p.read_bytes()).hexdigest()==historical['document']['sha256'])
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    result=dict(created_utc=c.e.now(),passed=all(x['passed'] for x in checks),checks=len(checks),failures=[x for x in checks if not x['passed']],
        new_fits=0,scope='bookkeeping/labels/metrics/constraints and previous-artifact preservation,not accuracy certification')
    c.write(OUT/'audit.json',result);print(result)


if __name__=='__main__':main()
