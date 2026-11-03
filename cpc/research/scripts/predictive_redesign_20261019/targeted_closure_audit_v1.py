"""Independently rebuild scores, rate regressions and upstream exclusions."""
from pathlib import Path
import sys,hashlib,itertools
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import targeted_closure_model_v1 as m
import targeted_closure_training_v1 as training
from audit_proposal_followup_v1 import macro,close

OUT=m.OUT/'independent_audit_v1';RATE=m.OUT/'rate_scale_v1';CHECKS=[]


def check(name,passed,**details):CHECKS.append(dict(name=name,passed=bool(passed),**details))


def independent_weight(rows):
    weights=[]
    groups=rows.related_cluster.nunique()
    for r in rows.itertuples():
        cluster=rows[rows.related_cluster.eq(r.related_cluster)]
        family=rows[rows.family.eq(r.family)]
        weights.append(r.scenario_weight/(groups*cluster.family.nunique()*family.sample_id.nunique()))
    return np.array(weights)


def main():
    assert not (OUT/'protocol_before.json').exists()
    for folder in (training.OUT,RATE):assert (folder/'complete.json').exists();c.verify_hashes(c.document(folder/'protocol_before.json'))
    files=[folder/name for folder,name in itertools.product((training.OUT,RATE),('predictions.csv','summary.csv'))]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),
        code_hashes=c.file_hashes([Path(__file__),Path(c.__file__)]),input_hashes=c.file_hashes(files),
        scope='independent score/regression/selection/exclusion reconstruction,not material accuracy certification'))
    table=c.cases();external=c.read(c.BASE/'external_rate_cohort_v2/scenarios24.csv')
    source={r.case_id:r for r in pd.concat([table,external],ignore_index=True).itertuples()}
    for folder in (training.OUT,RATE):
        p=c.read(folder/'predictions.csv');s=c.read(folder/'summary.csv')
        for row in s.itertuples():
            q=p[p.cohort.eq(row.cohort)&p.variant.eq(row.variant)]
            if row.scope=='common20':q=q[q.primary_common20.eq(True)]
            expected=(table[table.primary_common20.eq(True)] if row.scope=='common20' else table) if row.cohort=='old_cohort_heldout' else external
            complete=len(q)==len(expected) and set(q.case_id)==set(expected.case_id)
            check(folder.name+'_coverage_'+str(row.Index),complete and bool(row.complete))
            for name,pred,obs in [('nH','predicted','observed'),('mu','predicted_log_mu','observed_log_mu'),('rho','predicted_log_sigma','observed_log_sigma')]:
                check(folder.name+'_metric_'+name+'_'+str(row.Index),close(macro(q,pred,obs,name!='nH'),getattr(row,name+'_MAE')))
        for row in p.itertuples():
            r=source[row.case_id]
            check(folder.name+'_labels_'+str(row.Index),row.sample_id==r.sample_id and row.family==r.family and row.related_cluster==r.related_cluster and abs(row.observed-r.target)<1e-12)
            if row.mobility_eligible:check(folder.name+'_paired_label_'+str(row.Index),abs(row.observed_log_mu-np.log10(r.mu_cm2Vs))<1e-12)
        err=float(max(abs(p.predicted_log_sigma-p.predicted-p.predicted_log_mu-np.log10(c.learning.E_CHARGE))))
        check(folder.name+'_Hall_identity',err<5e-10,max_error=err)
    fits=list((training.OUT/'fits').glob('*.json'));check('sixteen_inventory_fits',len(fits)==16)
    for f in fits:
        j=c.document(f);train=j['train_units'];rows=table[table.related_cluster.isin(train)]
        check('training_ids_'+f.stem,set(j['training_case_ids'])==set(rows.case_id))
        best=min(t['loss'] for t in j['trials'] if t['loss'] is not None)
        selected=[t for t in j['trials'] if t['parameters']==j['parameters']]
        check('selected_fit_'+f.stem,j['selected_success'] and j['training_valid'] and any(close(t['loss'],best) and t['success'] for t in selected))
        warm=c.document(c.ROOT/j['initializer']['path'])
        check('initializer_isolation_'+f.stem,set(warm['train_units'])==set(train) and c.e.sha(c.ROOT/j['initializer']['path'])==j['initializer']['sha256'])
    cache={(tuple(sorted(j['train_units'].split('|'))),j['case_id']):np.array(j['values']) for j in c.document(RATE/'base_prediction_cache.json')}
    rawfits=c.document(RATE/'scale_fits.json');ratefits={}
    for k,j in enumerate(rawfits):
        train=tuple(sorted(j['train_units']));rows=table[table.related_cluster.isin(train)&table.mobility_eligible.eq(True)]
        primary=c.document(c.ROOT/j['primary_path']);check('rate_upstream_'+str(k),set(primary['train_units'])==set(train))
        check('rate_cases_'+str(k),set(j['training_case_ids'])==set(rows.case_id))
        w=independent_weight(rows);X=np.c_[np.ones(len(rows)),(np.log10(rows.incorporated_cm3.to_numpy(float))-18)/2,
            (rows.growth_T_C.to_numpy(float)-700)/300,rows.element.eq('Sn').to_numpy(float)]
        response=np.array([cache[(train,r.case_id)][1]-np.log10(r.mu_cm2Vs) for r in rows.itertuples()])
        beta=np.zeros(4)
        if j['variant']=='RateConstantMean':beta[0]=float(w@response)
        elif j['variant']=='RateConstantMedian':
            order=np.argsort(response,kind='stable');beta[0]=float(response[order[np.searchsorted(np.cumsum(w[order]),.5)]])
        else:
            # Solve augmented least squares independently of the centered normal equations.
            A=np.vstack([np.sqrt(w)[:,None]*X,np.c_[np.zeros(3),np.sqrt(j['alpha'])*np.eye(3)]])
            b=np.r_[np.sqrt(w)*response,np.zeros(3)];beta=np.linalg.lstsq(A,b,rcond=None)[0]
        error=float(max(abs(beta-np.array(j['coefficients']))));check('rate_coefficients_'+str(k),error<2e-9,max_error=error)
        ratefits[(train,j['variant'],j['alpha'])]=j
    for k,selection in enumerate(c.document(RATE/'selections.json')):
        train=tuple(sorted(selection['train_units']));scores=[]
        for candidate in selection['candidate_scores']:
            by=[]
            for fold in candidate['folds']:
                inner=tuple(sorted(fold['train_units']));held=fold['held']
                check('cv_exclusion_'+str(len(CHECKS)),held not in inner and set(inner)==set(train)-{held})
                j=ratefits[(inner,'RateFeatureRidge',candidate['alpha'])]
                rows=table[table.related_cluster.eq(held)&table.mobility_eligible.eq(True)]
                X=np.c_[np.ones(len(rows)),(np.log10(rows.incorporated_cm3.to_numpy(float))-18)/2,
                    (rows.growth_T_C.to_numpy(float)-700)/300,rows.element.eq('Sn').to_numpy(float)]
                pred=np.array([cache[(inner,r.case_id)][1] for r in rows.itertuples()])-X@np.array(j['coefficients'])
                per=rows[['family','sample_id']].copy();per['e']=abs(pred-np.log10(rows.mu_cm2Vs.to_numpy(float)))
                value=float(per.groupby(['family','sample_id']).e.mean().groupby(level='family').mean().mean())
                by.append(value);check('cv_score_'+str(len(CHECKS)),close(value,fold['score']))
            value=float(np.mean(by));check('cv_mean_'+str(len(CHECKS)),close(value,candidate['score']))
            scores.append((value,candidate['alpha']))
        alpha=min(scores,key=lambda r:(round(r[0],10),-r[1]))[1]
        check('selected_alpha_'+str(k),alpha==selection['selected_alpha'])
    p=c.read(RATE/'predictions.csv');check('zero_nH_drift',np.array_equal(p.predicted.to_numpy(),p.base_log_nH.to_numpy()))
    check('all_scales_positive_finite',bool(np.isfinite(p.positive_rate_scale).all() and (p.positive_rate_scale>0).all()))
    check('shared_scale_identities',max(abs(p.predicted_log_mu-p.base_log_mu+p.log10_rate_scale))<1e-12 and
        max(abs(p.predicted_log_sigma-p.base_log_sigma+p.log10_rate_scale))<1e-12)
    for variant,q in p[p.cohort.eq('old_cohort_heldout')].groupby('variant'):
        old=c.read(c.BASE/'proposal_followup_20261021_v1/constrained_block_v1/predictions.csv')
        old=old[old.variant.eq('FrozenPrimary')].set_index('case_id')
        check('exact_old_nH_'+variant,all(r.predicted==old.loc[r.case_id,'predicted'] for r in q.itertuples()))
    moment=c.document(RATE/'real_state_moment_checks.json');check('real_state_moment_identity',moment['passed'] and all(r['error_dex']<1e-10 for r in moment['records']))
    # Full previous delivery, including manuscripts and the live instruction snapshot.
    previous=c.document(c.BASE/'proposal_followup_20261021_v1/delivery_freeze_v1/manifest.json')
    for entry in previous['entries']:
        p=c.ROOT.parent/entry['path'];check('previous_unchanged_'+entry['path'],hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'])
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    result=dict(created_utc=c.e.now(),passed=all(r['passed'] for r in CHECKS),checks=len(CHECKS),
        failures=[r for r in CHECKS if not r['passed']],previous_files_verified=len(previous['entries']),
        no_new_material_validation_or_universal_gain_claim=True)
    c.write(OUT/'audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
