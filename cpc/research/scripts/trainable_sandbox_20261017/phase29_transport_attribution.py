"""Post hoc fixed-state scattering intervention; never a new fitted candidate.

Reconstruct actual saved primary transport, then change only mu0 to the prior
205. Charge/inventory states, trained weights, source policies and all main
scores are unchanged. The comparison order is explicit, not a physical causal
identification. Only zero-anchor held-source queries are covered.
"""
from pathlib import Path
from collections import defaultdict
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).parent))
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.learnable_constant_tau import learnable_fermi_transport
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE=e.BASE/'phase29_joint_scattering'
OUT=BASE/'transport_attribution_v1'
PRIMARY='JointProcess_internal'
CONTROL='ShareProcess205_internal'
REPLAY_TOL=2e-7


def main():
    assert not (OUT/'protocol_before.json').exists(),'preserve prior diagnostic before versioned repair'
    assert json.loads((BASE/'prediction/audit.json').read_text())['passed']
    files=[BASE/p for p in ('prediction/predictions.csv','prediction/diagnostics.json',
        'prediction/summary.csv','prediction/plan.json','prediction/audit.json',
        'delivery/predictor_bundle/manifest.json')]
    e.write(OUT/'protocol_before.json',dict(created_utc=e.now(),
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in
            (Path(__file__),ROOT/'src/sandbox/learnable_constant_tau.py',
             ROOT/'src/sandbox/coherent_transport.py')},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in files},
        scope='post hoc zero-anchor held-source fixed-state diagnostic after viewing main results',
        intervention='mu0=205cm2Vs;exact same saved free_n,Nq2,T,mass,epsilon and quadrature',
        comparison_order='fittedJoint -> same-state205 -> independentlytrainedShareProcess205',
        replay_tolerance_dex=REPLAY_TOL,monotonic_tolerance_dex=1e-9,
        no_new_training_or_source_adaptation=True,no_model_selection_or_main_score_changes=True,
        not_a_new_validated_predictor=True,goal_complete=False))
    table=read(BASE/'prediction/predictions.csv')
    primary=table[table.variant.eq(PRIMARY)&table.budget.eq(0)]
    control=table[table.variant.eq(CONTROL)&table.budget.eq(0)].set_index('case_id')
    assert len(primary)==49 and primary.case_id.nunique()==49
    diag={x['row_id']:x['diagnostics'] for x in json.loads((BASE/'prediction/diagnostics.json').read_text())}
    profile=published_coherent_profile();rows=[];checks=[]
    for r in primary.itertuples():
        q=diag[r.row_id];tr=q['coherent_transport'];c=control.loc[r.case_id]
        assert q['transport_head']=='native' and q['source_shift']==0
        assert not tr['unit_Hall_control'] and tr['extra_mobility_cm2Vs']>0
        assert abs(tr['mass_ratio']-profile.electron_mass_ratio)<1e-14
        kwargs=dict(Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,
            mass_ratio=tr['mass_ratio'],epsilon_r=tr['epsilon_r'],hall=True,
            order=tr['quadrature_order'],tail_width=tr['tail_width'])
        state=(np.log10(tr['free_n_cm3']),np.log10(tr['Nq2_cm3']),tr['temperature_K'])
        actual=learnable_fermi_transport(*state,log10_extra_mobility=np.log10(tr['extra_mobility_cm2Vs']),**kwargs)
        counter=learnable_fermi_transport(*state,log10_extra_mobility=np.log10(205.),**kwargs)
        saved=np.array([r.predicted,r.predicted_log_mu,r.predicted_log_sigma],float)
        reference=np.array([c.predicted,c.predicted_log_mu,c.predicted_log_sigma],float)
        error=float(np.max(abs(actual.values-saved)))
        checks.append(dict(name='actual_saved_transport_reconstruction',case_id=r.case_id,
            error=error,passed=error<=REPLAY_TOL))
        du=float(np.log10(tr['extra_mobility_cm2Vs']/205.))
        difference=actual.values-counter.values
        signs=np.array([-difference[0]*du,difference[1]*du,difference[2]*du])
        checks.append(dict(name='fixed_state_positive_channel_monotonicity',case_id=r.case_id,
            minimum_signed_change=float(signs.min()),passed=bool((signs>=-1e-9).all())))
        record=dict(case_id=r.case_id,sample_id=r.sample_id,family=r.family,
            related_cluster=r.related_cluster,role=r.role,
            primary_common20=bool(r.primary_common20),new_Sn=bool(r.new_Sn),
            mobility_eligible=bool(r.mobility_eligible),
            original_valid=bool(r.numeric and r.valid),
            counterfactual_valid=bool(r.numeric and r.valid and counter.values[0]>=13.),
            control_valid=bool(c.numeric and c.valid),
            fixed_free_n_cm3=tr['free_n_cm3'],fixed_Nq2_cm3=tr['Nq2_cm3'],
            fitted_mu0_cm2Vs=tr['extra_mobility_cm2Vs'],intervened_mu0_cm2Vs=205.,
            original_hall_factor=tr['readout_hall_factor'],
            counterfactual_hall_factor=counter.diagnostics['readout_hall_factor'])
        for j,name in enumerate(('nH','mu','sigma')):
            observed=float(getattr(r,{'nH':'observed','mu':'observed_log_mu','sigma':'observed_log_sigma'}[name]))
            record.update({name+'_observed':observed,name+'_original':saved[j],
                name+'_same_state205':float(counter.values[j]),name+'_control':reference[j],
                name+'_scattering_delta':float(saved[j]-counter.values[j]),
                name+'_remaining_delta':float(counter.values[j]-reference[j])})
        rows.append(record)
    e.csvwrite(OUT/'points.csv',rows)
    frame=pd.DataFrame(rows);summaries=[]
    def macro(part,column,target,valid):
        if part.empty or not part[valid].all():return None
        x=part.assign(error=abs(part[column]-part[target]))
        return float(x.groupby(['family','sample_id']).error.mean().groupby(level='family').mean().mean())
    for scope,mask in (
            ('all_query',frame.role.eq('query')),
            ('common20',frame.role.eq('query')&frame.primary_common20),
            ('newSn_query',frame.role.eq('query')&frame.new_Sn)):
        for name in ('nH','mu','sigma'):
            selected=mask if name=='nH' else mask&frame.mobility_eligible
            part=frame[selected];scores={}
            for version,valid in (('original','original_valid'),('same_state205','counterfactual_valid'),('control','control_valid')):
                scores[version]=macro(part,name+'_'+version,name+'_observed',valid)
            row=dict(scope=scope,property=name,conditions=part.sample_id.nunique(),
                DOIs=part.family.nunique(),scenario_rows=len(part),**scores)
            if all(v is not None for v in scores.values()):
                row.update(total_MAE_delta=scores['original']-scores['control'],
                    scattering_MAE_delta=scores['original']-scores['same_state205'],
                    remaining_fit_and_selection_MAE_delta=scores['same_state205']-scores['control'])
                residual=abs(row['total_MAE_delta']-row['scattering_MAE_delta']-
                    row['remaining_fit_and_selection_MAE_delta'])
                checks.append(dict(name='MAE_telescoping_identity',scope=scope,property=name,
                    error=residual,passed=residual<2e-12))
            summaries.append(row)
    e.csvwrite(OUT/'summary.csv',summaries)
    original_summary=read(BASE/'prediction/summary.csv')
    for r in summaries:
        original=original_summary[original_summary.variant.eq(PRIMARY)&original_summary.budget.eq(0)&original_summary.scope.eq(r['scope'])].iloc[0]
        matched=original_summary[original_summary.variant.eq(CONTROL)&original_summary.budget.eq(0)&original_summary.scope.eq(r['scope'])].iloc[0]
        for version,line in (('original',original),('control',matched)):
            actual=line[r['property']+'_MAE'];value=r[version]
            good=(value is None and pd.isna(actual)) or (value is not None and abs(value-actual)<2e-12)
            checks.append(dict(name='frozen_main_metric_replay',scope=r['scope'],property=r['property'],
                version=version,passed=bool(good)))
    common=[r for r in summaries if r['scope']=='common20']
    fig,axes=plt.subplots(1,3,figsize=(9.5,3.5))
    for ax,name in zip(axes,('nH','mu','sigma')):
        r=next(x for x in common if x['property']==name)
        ax.bar(range(3),[r[k] for k in ('original','same_state205','control')],color=['C3','C1','C0'])
        ax.set(xticks=range(3),xticklabels=['Fitted joint','Same state\nmu0=205','Retrained\nShare205'],
            ylabel='MAE (dex)',title=name+';'+str(r['conditions'])+' conditions')
        ax.tick_params(axis='x',labelsize=7);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Post hoc zero-anchor diagnostic;not a validated new candidate',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'fixed_state_intervention.png',dpi=170)
    fig.savefig(OUT/'fixed_state_intervention.pdf');plt.close(fig)
    result=dict(passed=all(x['passed'] for x in checks),created_utc=e.now(),rows=checks,
        checks=len(checks),actual_saved_states=len(rows),transport_evaluations=2*len(rows),
        new_hot_or_cold_solves=0,new_fits=0,new_source_calibrations=0,
        maximum_original_transport_replay_error_dex=max(x['error'] for x in checks if x['name']=='actual_saved_transport_reconstruction'),
        total_charge_state_and_inventory_deliberately_fixed=True,
        counterfactual_not_a_trained_or_validated_predictor=True,
        decomposition_order_dependent_not_microscopic_causal_identification=True,
        original_scores_and_serving_weights_unchanged=True,goal_complete=False)
    e.write(OUT/'audit.json',result)
    print('Fixed-state diagnostic',result['passed'],'checks',len(checks),'states',len(rows),flush=True)
    print(pd.DataFrame(common).to_string(index=False),flush=True)
    if not result['passed']:raise RuntimeError('Preserve original diagnostic failure')


if __name__=='__main__':main()
