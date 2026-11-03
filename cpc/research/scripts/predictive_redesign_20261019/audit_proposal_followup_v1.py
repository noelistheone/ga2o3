"""Independent bookkeeping, score, label-isolation and trained-grid audits."""
from pathlib import Path
import sys,itertools,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from association_spatial_model_v1 import AssociationSpatialModel
from sandbox.coherent_fermi import published_coherent_profile
import phase34_learning_v1 as overlay

OUT=c.BASE/'proposal_followup_20261021_v1/independent_audit_v1'
HELD=c.BASE/'phase35_association_spatial/related_group_heldout_v1'
BLOCK=c.BASE/'proposal_followup_20261021_v1/constrained_block_v1'
CHECKS=[]


def check(name,passed,**details):CHECKS.append(dict(name=name,passed=bool(passed),**details))


def macro(rows,pred,obs,paired=False):
    q=rows[rows.mobility_eligible.eq(True)] if paired else rows
    if q.empty:return None
    if not q.numeric.eq(True).all() or not q.valid.eq(True).all():return None
    e=q.assign(error=abs(q[pred]-q[obs])).groupby(['family','sample_id']).error.mean()
    return float(e.groupby(level='family').mean().mean())


def close(a,b,tol=2e-10):
    aa=None if a is None or pd.isna(a) else float(a);bb=None if b is None or pd.isna(b) else float(b)
    return aa is None and bb is None or aa is not None and bb is not None and abs(aa-bb)<=tol


def table_checks(base,kind):
    c.verify_hashes(c.document(base/'protocol_before.json'));p=c.read(base/'predictions.csv');s=c.read(base/'summary.csv')
    roles=c.roles();rolemap={(r.sample_id,int(r.budget)):r.role for r in roles.itertuples()}
    actual={r.case_id:r for r in c.cases().itertuples()}
    if 'observed_log_sigma' in p:
        num=p[p.numeric.eq(True)]
        err=np.abs(num.predicted_log_sigma-num.predicted-num.predicted_log_mu-np.log10(c.learning.E_CHARGE))
        check(kind+'_Hall_identity',float(err.max())<5e-10,max_error=float(err.max()))
    for row in p.itertuples():
        x=actual[row.case_id]
        check(kind+'_role_'+str(row.Index),row.role==rolemap[(row.sample_id,int(row.budget))])
        check(kind+'_label_'+str(row.Index),abs(row.observed-x.target)<1e-12 and row.family==x.family and row.related_cluster==x.related_cluster)
    for row in s.itertuples():
        q=p[p.budget.eq(row.budget)&p.role.eq('query')] if hasattr(row,'budget') else p
        if kind=='phase35':q=q[q.method.eq(row.method)&q.alpha.eq(row.alpha)&q.adaptation.eq(row.adaptation)]
        else:q=q[q.variant.eq(row.variant)]
        if row.scope=='common20':q=q[q.primary_common20.eq(True)]
        expected=c.cases();ids=set(roles.loc[roles.budget.eq(row.budget)&roles.role.eq('query'),'sample_id']) if hasattr(row,'budget') else set(expected.sample_id)
        expected=expected[expected.sample_id.isin(ids)]
        if row.scope=='common20':expected=expected[expected.primary_common20.eq(True)]
        complete=len(q)==len(expected) and set(q.case_id)==set(expected.case_id)
        n=macro(q,'predicted','observed') if complete else None
        check(kind+'_score_nH_'+str(row.Index),close(n,row.nH_MAE),complete=complete)
        if hasattr(row,'mu_MAE'):
            check(kind+'_score_mu_'+str(row.Index),close(macro(q,'predicted_log_mu','observed_log_mu',True) if complete else None,row.mu_MAE))
            check(kind+'_score_rho_'+str(row.Index),close(macro(q,'predicted_log_sigma','observed_log_sigma',True) if complete else None,row.rho_MAE))
    return p,s


def grid_checks():
    p=c.read(HELD/'predictions.csv');p=p[p.budget.eq(0)&p.adaptation.eq('none')&p.alpha.eq(.1)&p.method.isin(['ClusterCoupled','ConstantTexture','UniformBackground'])]
    rows={r.case_id:r for r in c.cases().itertuples()};models={};results=[]
    for r in p.itertuples():
        fit=c.document(HELD/'fits'/(r.fit_key+'.json'))
        if r.method not in models:
            from phase35_learning_v1 import METHODS
            models[r.method]=AssociationSpatialModel(profile=published_coherent_profile(),db=overlay.database(),grid_size=256,period_nm=1000.,**METHODS[r.method])
        try:
            q=models[r.method].point(rows[r.case_id],fit['parameters'],require_observation=False);v=np.r_[q.values[0],q.transport_values]
            old=np.array([r.predicted,r.predicted_log_mu,r.predicted_log_sigma]);err=float(max(abs(v-old)))
            results.append(dict(method=r.method,case_id=r.case_id,fit_key=r.fit_key,max_grid_error_dex=err,
                refined_valid=bool(q.diagnostics['observation_domain_valid']),numeric=True,
                k_bad=q.diagnostics['conductivity_weighted_kell_bad'],rho_bad=q.diagnostics['resistivity_weighted_kell_bad']))
        except (ValueError,RuntimeError,ArithmeticError) as exc:results.append(dict(method=r.method,case_id=r.case_id,numeric=False,error=repr(exc)))
    pd.DataFrame(results).to_csv(OUT/'trained_grid128_256.csv',index=False)
    c.write(OUT/'trained_grid_summary.json',dict(created_utc=c.e.now(),cases=len(results),
        max_error_dex=max(r.get('max_grid_error_dex',0.) for r in results),
        failed=sum(not r['numeric'] for r in results),not_an_accuracy_or_Born_gate=True,
        full_grid_errors_reported_not_used_for_retrospective_model_selection=True))


def main():
    assert not (OUT/'protocol_before.json').exists()
    for b in [HELD,BLOCK,c.BASE/'proposal_followup_20261021_v1/profiled_adaptation_v2']:
        assert (b/'complete.json').exists(),b
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),purpose='independent aggregation,role/fit audits and grid refinement',
        code_hashes=c.file_hashes([Path(__file__),Path(c.__file__)]),
        input_hashes=c.file_hashes([HELD/'predictions.csv',HELD/'summary.csv',BLOCK/'predictions.csv',BLOCK/'summary.csv',
            c.BASE/'proposal_followup_20261021_v1/profiled_adaptation_v2/predictions.csv'])))
    p,s=table_checks(HELD,'phase35')
    for f in (HELD/'fits').glob('*.json'):
        fit=c.document(f);check('held_fit_exclusion_'+f.stem,fit['held_group'] not in fit['train_units'] and len(fit['train_units'])==4)
        finite=[r for r in fit['trials'] if r.get('loss') is not None]
        check('held_fit_selection_'+f.stem,not finite or close(fit['selected_loss'],min(r['loss'] for r in finite)))
    for version in ['profiled_adaptation','profiled_adaptation_v2']:
        base=c.BASE/'proposal_followup_20261021_v1'/version;pp,ss=table_checks(base,version)
        one=pp[pp.budget.eq(1)&pp.variant.eq('NestedProfiledGate')]
        check(version+'_one_anchor_no_internal_shape',one.internal_step_norm.max()==0 and one.information_rank.max()==0)
        two=pp[pp.budget.eq(2)&pp.variant.eq('NestedProfiledGate')]
        check(version+'_two_anchors_rank_limit',two.information_rank.max()<=1)
        zero=ss[ss.scope.eq('common20')&ss.budget.eq(0)&~ss.variant.eq('NestedUnitDose')]
        check(version+'_zero_exact_replay',all(abs(x-.3479041542002425)<2e-12 for x in zero.nH_MAE))
        for choice in c.document(base/'selections.json'):
            if not choice['budget'] or choice['variant']=='FrozenShared':continue
            vals=[(r['score'],r['policy']) for r in choice['candidate_scores'] if r['score'] is not None]
            check(version+'_selected_inner_min_'+str(len(CHECKS)),not vals or close(choice['inner_score'],min(x[0] for x in vals)))
    bp,bs=table_checks(BLOCK,'constrained_block')
    for f in (BLOCK/'primary_fits').glob('*.json'):
        j=c.document(f);check('block_primary_exclusion_'+f.stem,j['held'] not in j['train_units'] and len(j['train_units'])==4)
    for f in (BLOCK/'secondary_fits').glob('*.json'):
        j=c.document(f)
        if j.get('selected'):
            check('block_training_constraints_'+f.stem,j['prefix_frozen'] and j['training_full_constraints_valid']
                and j['training_max_nH_drift']<=.03000011 and j['training_nH_loss_change']<=.00010011)
            if not j['selected'].get('fallback',False):check('block_stopped_'+f.stem,j['selected']['success'])
    # Re-run one source after perturbing all forbidden query labels.
    import profiled_information_adaptation_v2 as gate
    selections=c.document(c.BASE/'proposal_followup_20261021_v1/profiled_adaptation_v2/selections.json')
    for held in c.units():
        choice=next(x for x in selections if x['outer_held']==held and x['budget']==2 and x['variant']=='NestedProfiledGate')
        fit,_=c.fit([u for u in c.units() if u!=held],'JointProcess',choice['shared_config'])
        table=c.cases();table=table[table.related_cluster.eq(held)];role=c.roles()
        source=next(g for fam,g in table.groupby('family',sort=True)
            if role[role.family.eq(fam)&role.budget.eq(2)&role.role.eq('query')].shape[0]>0)
        anchor=set(role.loc[role.family.eq(source.family.iloc[0])&role.budget.eq(2)&role.role.eq('anchor'),'sample_id'])
        altered=source.copy();altered.loc[~altered.sample_id.isin(anchor),'target']+=1234.
        a=gate.simulate(source,fit,2,*choice['policy']);b=gate.simulate(altered,fit,2,*choice['policy'])
        check('query_label_isolation_'+held,max(abs(x['predicted']-y['predicted']) for x,y in zip(a,b))<1e-12)
    grid_checks();c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'audit.json',dict(created_utc=c.e.now(),passed=all(r['passed'] for r in CHECKS),checks=len(CHECKS),
        failures=[r for r in CHECKS if not r['passed']],material_accuracy_or_novelty_not_certified=True,new_fits=0))
    print(dict(passed=all(r['passed'] for r in CHECKS),checks=len(CHECKS),failures=[r for r in CHECKS if not r['passed']]),flush=True)


if __name__=='__main__':main()
