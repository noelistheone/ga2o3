"""Portable score, reading-interval and readout-sensitivity analysis; no refits.

All comparisons retain the original source/specimen weighting. Interval extrema
use one shared observation for a pair of predictions, at fixed nominal T. They
are deterministic digitization-box bounds, not confidence intervals.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

M = Path(__file__).resolve().parents[1]
D, V, TAB = M/'source_data', M/'verification', M/'tables'
CHECKS = []


def check(name, ok, **detail):
    CHECKS.append(dict(name=name, passed=bool(ok), **detail))


def read(name):
    return pd.read_csv(D/(name+'.csv'))


def mean(d, col):
    return float(d.groupby(['family', 'sample_id'])[col].mean().groupby('family').mean().mean())


def score(d):
    return mean(d.assign(error=abs(d.predicted-d.observed)), 'error')


def table(name, caption, label, columns, rows):
    s = ['\\begin{table}[htbp]\\centering', '\\caption{'+caption+'}', '\\label{'+label+'}',
         '\\small', '\\begin{tabular}{'+'l'+'r'*(len(columns)-1)+'}', '\\toprule',
         ' & '.join(columns)+r' \\ \midrule']
    s += [' & '.join(map(str,r))+r' \\' for r in rows]
    s += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    (TAB/(name+'.tex')).write_text('\n'.join(s))


def main():
    V.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    room=read('room_temperature_predictions')
    control=read('capacity_control_predictions').rename(columns={'target':'observed','method':'variant'})
    records=[]; source=[]
    for data in [room,control]:
        for variant, d in data.groupby('variant'):
            for scope in ['common20','all35']:
                s=d[d.primary_common20.eq(True)] if scope=='common20' else d
                check('room_query_coverage',len(s)==len(room[(room.variant.eq('LearnedNHInventory')) & (room.primary_common20.eq(True) if scope=='common20' else True)]),method=variant,scope=scope)
                records.append(dict(method=variant,scope=scope,MAE=score(s),conditions=s.sample_id.nunique(),sources=s.family.nunique()))
                for fam, g in s.groupby('family'):
                    source.append(dict(method=variant,scope=scope,family=fam,MAE=score(g)))
    pd.DataFrame(records).to_csv(D/'capacity_comparison_scores.csv',index=False)
    pd.DataFrame(source).to_csv(D/'capacity_comparison_sources.csv',index=False)
    rc=pd.DataFrame(records).pivot(index='method',columns='scope',values='MAE')
    labels={'FixedNHReference':'Fixed full inventory (2)', 'LearnedNHInventory':'Learned full inventory (6)',
            'ReducedInventory6':'Reduced inventory (6)', 'DirectQuadratic6':'Direct quadratic (6)'}
    table('capacity_comparison','Capacity controls for zero-anchor carrier prediction. Parentheses give the number of fitted coefficients; errors are source-macro MAE in dex. The reduced model retains the primary Fermi-moment Hall readout.','tab:Scapacity',
          ['Model (parameters)','Common 20','All 35'],[(labels[k],f'{rc.loc[k,"common20"]:.5f}',f'{rc.loc[k,"all35"]:.5f}') for k in labels])

    tp=read('temperature_predictions'); q=tp[tp.budget.eq(0)&tp.role.eq('query')]
    wide=read('thermal_bound_predictions')
    thermal=[]; bysource=[]; hall=[]
    for method,label in [('fixed_joint_offset','Fixed donor'),('shared_joint_offset','Primary learned'),('direct_joint_offset','Direct Ridge'),('ExpandedBounds','Expanded bounds')]:
        d=wide if method=='ExpandedBounds' else q[q.method.eq(method)]
        check('thermal_142_ids',set(d.obs_id)==set(wide.obs_id),method=method)
        for scope in ['all142','common80']:
            s=d[d.common40_125.eq(True)] if scope=='common80' else d
            thermal.append(dict(method=label,scope=scope,MAE=score(s),reads=len(s)))
            for fam, g in s.groupby('family'):
                bysource.append(dict(method=label,scope=scope,family=fam,MAE=score(g),reads=len(g)))
        if method in ('fixed_joint_offset','shared_joint_offset'):
            for factor in [1.,1.25,1.5,1.93]:
                shifted=d.assign(predicted=d.predicted-np.log10(factor))
                for scope in ['all142','common80']:
                    s=shifted[shifted.common40_125.eq(True)] if scope=='common80' else shifted
                    hall.append(dict(method=label,assumed_rH=factor,scope=scope,MAE=score(s)))
    ts=pd.DataFrame(thermal); ts.to_csv(D/'thermal_robustness_scores.csv',index=False)
    pd.DataFrame(bysource).to_csv(D/'thermal_robustness_sources.csv',index=False)
    pd.DataFrame(hall).to_csv(D/'thermal_hall_sensitivity.csv',index=False)
    tab=ts.pivot(index='method',columns='scope',values='MAE')
    table('thermal_robustness','Sensitivity to expanded training bounds. The same two source-exclusion folds are refitted after lowering the background bound to $10^{12}\\,\\mathrm{cm}^{-3}$ and raising the dilute donor-energy bound to $0.3\\,\\mathrm{eV}$.','tab:Sbounds',
          ['Model','Full 142','Common 80'],[(k,f'{tab.loc[k,"all142"]:.5f}',f'{tab.loc[k,"common80"]:.5f}') for k in ['Fixed donor','Direct Ridge','Primary learned','Expanded bounds']])

    intervals=read('thermal_reading_intervals')
    pair=[]
    for budget,left,right,label in [(0,'shared_joint_offset','direct_joint_offset','Learned $-$ Ridge, zero anchors'),(2,'shared_joint_Tf_E','shared_joint_offset','Internal $-$ shift, two anchors'),(2,'shared_joint_Tf_E','direct_joint_affine','Internal $-$ Ridge, two anchors')]:
        a=tp[(tp.method==left)&(tp.budget==budget)&(tp.role=='query')]
        b=tp[(tp.method==right)&(tp.budget==budget)&(tp.role=='query')]
        d=a.merge(b[['obs_id','predicted','observed']],on='obs_id',suffixes=('','_baseline'),validate='one_to_one').merge(intervals[['obs_id','reading_lower','reading_upper']],on='obs_id',validate='one_to_one')
        check('paired_same_observations',np.max(abs(d.observed-d.observed_baseline))<1e-12,budget=budget,comparison=label)
        l=d.reading_lower.to_numpy();u=d.reading_upper.to_numpy();a=d.predicted.to_numpy();b=d.predicted_baseline.to_numpy()
        check('observations_inside_digitization_box',np.all(l<=d.observed)&np.all(d.observed<=u),budget=budget)
        candidates=np.stack([l,u,np.clip(a,l,u),np.clip(b,l,u)],axis=1)
        delta=abs(a[:,None]-candidates)-abs(b[:,None]-candidates)
        d=d.assign(delta_lower=delta.min(axis=1),delta_upper=delta.max(axis=1),delta_nominal=abs(a-d.observed)-abs(b-d.observed))
        for scope in ['all_query','common80']:
            s=d[d.common40_125.eq(True)] if scope=='common80' else d
            row=dict(comparison=label,budget=budget,scope=scope,reads=len(s),nominal=mean(s,'delta_nominal'),lower=mean(s,'delta_lower'),upper=mean(s,'delta_upper'))
            check('paired_interval_contains_nominal',row['lower']-1e-12<=row['nominal']<=row['upper']+1e-12,comparison=label,scope=scope)
            pair.append(row)
    pd.DataFrame(pair).to_csv(D/'thermal_digitization_loss_bounds.csv',index=False)
    table('digitization_bounds','Paired source-macro MAE differences over the archived carrier-reading intervals at fixed nominal temperatures (dex). Negative values favor internal learning. Each pair uses the same perturbed reading; interval extrema are deterministic bounds, not confidence intervals.','tab:Spixels',
          ['Comparison','Queries','Nominal','Lower','Upper'],[(x['comparison'],x['reads'],f"{x['nominal']:.5f}",f"{x['lower']:.5f}",f"{x['upper']:.5f}") for x in pair])

    ap=read('acquisition_transfer_predictions'); ac=read('acquisition_transfer_scores'); states=read('acquisition_selected_states')
    check('acquisition_full_coverage',len(ap)==160 and len(ac)==20)
    check('acquisition_physical_coverage',ap.physical_valid.eq(True).all())
    for r in ac.itertuples():
        g=ap[(ap.element==r.element)&(ap.policy==r.policy)&(ap.budget==r.budget)]
        value=float(abs(g.predicted-g.full_catalogue).mean())
        check('acquisition_recomputed_mean',abs(value-r.mean)<1e-12,element=r.element,policy=r.policy,budget=r.budget)
    primary_inputs=read('validation_room_inputs')
    all_groups=set(primary_inputs.related_cluster)
    table('acquisition_states','First two acquired charge-state coordinates in each masked-dopant test. Each selected parent contains one dopant atom and has $\\Delta N_{\\mathrm{Ga}}=-2$, $\\Delta N_{\\mathrm O}=0$. Sensitivities are absolute derivatives of $\\log_{10}n_H$ with respect to the relative coordinate $\\xi$ (dex/eV), evaluated on the design set immediately before acquisition. Parent IDs refer to the supplied catalogue.','tab:Sstates',
          ['Dopant','Step','Parent','$q$','$q_0$','Mean sensitivity','Maximum'],[(r.element,r.step,r.parent,r.q,r.q_reference,f'{r.mean_abs_sensitivity_dex_per_eV:.4f}',f'{r.max_abs_sensitivity_dex_per_eV:.4f}') for r in states.itertuples()])
    # Replay checks depend only on the new fit/verification records, not fitting.
    fits=json.loads((V/'capacity_control_fits.json').read_text()) if (V/'capacity_control_fits.json').exists() else None
    if fits:
        check('capacity_130_fits',len(fits['fits'])==130 and fits['all_fits_converged'])
        check('capacity_derivative',fits['analytic_gradient_max_error']<1e-7)
        check('capacity_Hall_interpolation',fits['hall_interpolation_max_error_dex']<2e-7)
        check('ten_outer_selections',len(fits['selections'])==10)
        for s in fits['selections']:
            check('nested_training_score_minimized',abs(s['inner_MAE']-min(x[0] for x in s['grid']))<1e-12,held=s['held'],method=s['method'])
            matching=[f for f in fits['fits'] if f['method']==s['method'] and f['alpha']==s['selected_alpha'] and set(f['train'])==all_groups-{s['held']}]
            check('selected_outer_fit_excludes_test_group',len(matching)==1,held=s['held'],method=s['method'])
    if (V/'thermal_bound_refits.json').exists():
        refits=json.loads((V/'thermal_bound_refits.json').read_text())['fits']
        thermal_inputs=read('validation_thermal_inputs').set_index('obs_id')
        for fit in refits:
            tr=thermal_inputs.loc[fit['train_observations']]
            check('expanded_thermal_whole_source_exclusion',not tr.family.eq(fit['held']).any(),held=fit['held'])
            check('expanded_thermal_two_converged_starts',len(fit['trials'])==2 and all(x['success'] for x in fit['trials']),held=fit['held'])
    result=dict(checks=len(CHECKS),passed=sum(x['passed'] for x in CHECKS),failed=[x for x in CHECKS if not x['passed']],items=CHECKS)
    (V/'submission_validation_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(ts.to_string(index=False)); print(pd.DataFrame(hall).to_string(index=False));print(pd.DataFrame(pair).to_string(index=False))
    print(json.dumps({k:v for k,v in result.items() if k!='items'}))
    if result['failed']:raise SystemExit(1)


def figures():
    import matplotlib.pyplot as plt
    from make_figures import TEAL, ORANGE, BLUE, GRAY, clean, panel, save
    ac=read('acquisition_transfer_scores')
    fig,axs=plt.subplots(1,2,figsize=(7.25,2.9),layout='constrained')
    for ax,el,letter in zip(axs,['Si','Sn'],['a','b']):
        panel(ax,letter,el+': frozen acquisition at new temperatures')
        for pol,lab,col,ls in [('goal','Property-directed',TEAL,'-'),('energy_variance','Energy variance',ORANGE,'--')]:
            s=ac[(ac.element==el)&(ac.policy==pol)].sort_values('budget')
            ax.plot(s.budget,s['mean'],'o'+ls,color=col,ms=3,label=lab)
        ax.set(xlabel='Revealed labels per dopant',ylabel='Discrepancy from full catalogue (dex)',xticks=range(5),ylim=(-.02,.65));clean(ax)
        ax.legend(frameon=False,fontsize=7)
    save(fig,'figS3_acquisition_transfer')
    h=read('thermal_hall_sensitivity');t=read('thermal_robustness_scores')
    fig,axs=plt.subplots(1,2,figsize=(7.25,2.9),layout='constrained')
    ax=axs[0];panel(ax,'a','Source-excluded refits with expanded bounds')
    names=['Direct Ridge','Primary learned','Expanded bounds'];x=np.arange(3)
    for scope,off,col,lab in [('all142',-.18,TEAL,'Full 142 readings'),('common80',.18,BLUE,'Common 80 queries')]:
        values=[float(t[(t.method==n)&(t.scope==scope)].MAE.iloc[0]) for n in names]
        ax.bar(x+off,values,width=.34,color=col,label=lab)
    ax.set(xticks=x,xticklabels=['Direct\nRidge','Primary\nlearned','Expanded\nbounds'],ylabel='Carrier MAE (dex)',ylim=(0,.45));clean(ax)
    ax.legend(frameon=False,fontsize=7)
    ax=axs[1];panel(ax,'b','Frozen-prediction Hall-factor sensitivity')
    for method,col in [('Fixed donor',GRAY),('Primary learned',TEAL)]:
        for scope,ls in [('all142','-'),('common80','--')]:
            s=h[(h.method==method)&(h.scope==scope)].sort_values('assumed_rH')
            ax.plot(s.assumed_rH,s.MAE,'o'+ls,color=col,ms=3,label=method+(' / full' if scope=='all142' else ' / common'))
    ax.set(xlabel=r'Assumed constant $r_H$ (no refitting)',ylabel='Carrier MAE (dex)',xticks=[1,1.25,1.5,1.93]);clean(ax)
    ax.legend(frameon=False,fontsize=6.5)
    save(fig,'figS4_thermal_sensitivity')


if __name__=='__main__':
    main()
    figures()
