"""Fixed prediction variants, nested-selection rules and source packet design."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.context_saved_model import INPUT_NAMES
from sandbox.activation_transport_control import ct_log_readout
import phase29_learning as l
import phase29_training_v2 as t
import phase29_data as data
import phase2_energy as e

LAM=(None,0.,.01,.1,1.,10.)
VARIANTS={m+'_offset':dict(method=m,kind='offset',selection='all',head='native') for m in l.METHODS}
PHYSICAL=tuple(m for m in l.NONLINEAR if l.SPECS[m]['kind']!='activation')
for m in PHYSICAL:
    VARIANTS[m+'_internal']=dict(method=m,kind='internal',selection='all',head='native')
for m in ('Reference205','DirectMultiRidge','UnitDose'):
    VARIANTS[m+'_affine']=dict(method=m,kind='affine',selection='all',head='native')
for m in ('JointProcess','Reference205'):
    for kind in ('offset','internal'):
        VARIANTS[m+'_nHonly_'+kind]=dict(method=m,kind=kind,selection='nHonly',head='native')
for m in ('JointProcess','Reference205','UnitDose','MedianActivation','DoseLine'):
    for head in ('native_mu_mean','native_mu_median','native_rho_mean','CT_mu_mean'):
        VARIANTS[m+'_nHonly_'+head]=dict(method=m,kind='internal' if m in PHYSICAL else 'offset',
            selection='nHonly',head=head)
assert len(VARIANTS)==48
PRIMARY='JointProcess_internal'
MINIMAL=('key','method','parameters','training_valid','selected_success','center','scale',
    'nH_coefficients','rho_coefficients','coefficients','offset_dex',
    'transport_head','transport_head_offset_dex')


def load_fit(units,method,config):
    key,path=t.fit_path(units,method,config)
    if not path.exists():raise RuntimeError('Prediction must not launch an undeclared fit: '+str(path))
    f=json.loads(path.read_text())
    assert f['key']==key and f['train_units']==sorted(units) and f['method']==method and f['config']==config
    return f,path


def minimal_fit(fit):return {k:e.clean(fit[k]) for k in MINIMAL if k in fit}


def inputs(row):return {k:e.clean(getattr(row,k)) for k in INPUT_NAMES}


def configs(method,selection):
    values=l.configs(method)
    if selection=='nHonly' and method in l.NONLINEAR:values=[v for v in values if v['gamma']==0.]
    assert selection in ('all','nHonly')
    return values


def config_tie(config):
    return (-float(config.get('alpha',config.get('ridge_alpha',0.))),float(config.get('gamma',0.)))


def lambda_tie(lam):return -np.inf if lam is None else -float(lam)


def condition_score(rows,*,endpoint='predicted'):
    if not rows:return np.inf
    if not all(r['numeric'] and r['valid'] for r in rows):return np.inf
    table=pd.DataFrame(rows)
    table['error']=abs(table[endpoint]-table.observed)
    # Every scenario of a reported condition is explicitly retained.
    percondition=table.groupby(['family','sample_id'],sort=True).error.mean()
    return float(percondition.groupby(level='family').mean().mean())


def source_design(family,cases,roles):
    table=cases[cases.family.eq(family)].copy()
    assert not table.empty and table.groupby('sample_id').size().nunique()==1
    episodes=[]
    for b in (0,1,2):
        selected=roles[roles.family.eq(family)&roles.budget.eq(b)]
        names=sorted(selected.loc[selected.role.eq('anchor'),'sample_id'])
        ai=[i for i,r in enumerate(table.itertuples()) if r.sample_id in names]
        assert len(names)==b
        episodes.append(dict(budget=b,anchor_condition_ids=names,anchor_indices=ai,
            anchor_labels=[float(table.iloc[i].target) for i in ai]))
    return [inputs(row) for row in table.itertuples()],table.case_id.tolist(),episodes


def with_transport_head(fit,table,head):
    result=dict(fit);result['transport_head']='native';result['transport_head_offset_dex']=0.
    if head=='native':return result,dict(head=head,paired_conditions=0,no_postfit=True)
    if not fit['training_valid'] or not fit['selected_success']:
        return result,dict(head=head,fit_unqualified=True)
    mask,rho=l.observed_pair(table);weights=l.weights(table,paired_only=True)[mask]
    if not mask.any():raise RuntimeError('Declared current fit context has no paired direction')
    values=np.asarray(fit['selected_training_values'],float)
    if head=='CT_mu_mean':
        prediction=np.array([ct_log_readout(g,300.)[0] for g in values[:,0]])
        error=np.log10(table.mu_cm2Vs.to_numpy(float)[mask])-prediction[mask]
        value=float(weights@error);kind='CT_offset'
    elif head=='native_rho_mean':
        error=-rho[mask]-values[mask,2];value=float(weights@error);kind='native_offset'
    else:
        error=np.log10(table.mu_cm2Vs.to_numpy(float)[mask])-values[mask,1]
        if head=='native_mu_mean':value=float(weights@error)
        elif head=='native_mu_median':value=l.weighted_median(error,weights)
        else:raise ValueError('Unknown fixed transport calibration')
        kind='native_offset'
    result.update(transport_head=kind,transport_head_offset_dex=value)
    return result,dict(head=head,transport_head=kind,offset_dex=value,
        paired_conditions=int(table.loc[mask,'sample_id'].nunique()),
        training_condition_ids=sorted(table.sample_id.unique()),
        calibration_uses_only_training_labels=True,no_query_mu=True,
        update_changes_latent_state=False)


def declare():
    """Freeze before any new outer prediction; training may still be active."""
    assert json.loads((data.OUT/'saved_source_gates_v1/gates.json').read_text())['passed']
    assert json.loads((data.OUT/'numerical_gates_v2/gates.json').read_text())['passed']
    assert (data.OUT/'training_protocol.json').exists()
    files=[Path(__file__),Path(__file__).with_name('phase29_source_runtime.py'),
        ROOT/'src/sandbox/joint_saved_model.py',ROOT/'src/sandbox/context_source.py',
        ROOT/'src/sandbox/functional_source.py']
    value=dict(created_utc=e.now(),primary=PRIMARY,variants=VARIANTS,lambdas=LAM,
        shared_selection='inner related-group held-out zero-anchor nH MAE;equalDOIthenconditionthenfixedscenario within each heldgroup;equalheldgroups',
        global_config_ties='round score10digits,prefer stronger alpha then smaller gamma;fixed variant identity',
        source_selection='training-inner query nH condition/DOI macroMAE;shared config fixed;None then largerlambda on ties',
        source_update='separateperDOI0/1/2Hallcondition anchors;8scenario repeats not8labels;no query mu',
        source_optimizer='same33cubicnodes,3localsearches,maxiter80,32768points,raw[-6,6];sampledzero-adjacentfiniteinputbranch',
        reference_internal_coordinate='relative log10 inventory odds;distinct from joint excessshare but equal oneparameter anchorbudget',
        direct_source_rule='updateHallhead,keeprhoheadfixed;mu followsidentity',
        physical_offset_rule='retain latent-state mobility,shiftHall andsigma;empirical output control',
        empirical_source_rule='shiftHall,recomputeCT,retain declared trained global transport offset',
        affine_controls='Reference205/DirectMultiRidge/UnitDose;sameanchors and finite response checks;two coefficients explicitly reported',
        transport_heads='predeclared four scalar controls on nH-only selected fits;calibrate training pairs only;no selection byoutermu',
        main_evaluation='originalcommon20 DOI-macro nH;groupmacro/allqueries andpairedfullchain separate',
        unknown_input_scoring='mean absolute error over fixed scenarios percondition,thenDOI;not a calibrated probability or selectbestscenario',
        fullcoverage_required_for_MAE=True,failed_and_nonstopped_fits_retained_but_not_served=True,
        retrospective=True,new_material_measurements=0,new_DFT_MD=0,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in files},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (
            data.OUT/'training_protocol.json',data.OUT/'data/qualification.json',
            data.OUT/'saved_source_gates_v1/gates.json',data.OUT/'saved_source_gates_v1/protocol_before.json',
            data.OUT/'numerical_gates_v2/gates.json')})
    target=data.OUT/'prediction_design_protocol.json'
    if target.exists():
        old=json.loads(target.read_text())
        for k in ('code_hashes','input_hashes','variants'):assert old[k]==e.clean(value[k]),k
        return old
    e.write(target,value);return value


if __name__=='__main__':print('Prediction design declared',len(declare()['variants']),flush=True)
