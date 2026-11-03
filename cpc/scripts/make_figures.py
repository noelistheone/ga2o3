"""Original scientific artwork from curated, frozen numerical evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from ase.io import read

ROOT=Path(__file__).resolve().parents[2]
OUT=(ROOT/'paper/npj/ml_sandbox_manuscript_v3' if Path(__file__).resolve().parent.name=='manuscript_ml_sandbox_v3'
     else Path(__file__).resolve().parent.parent)
D=OUT/'source_data'; F=OUT/'figures';F.mkdir(exist_ok=True)
(OUT/'audit').mkdir(exist_ok=True)
TEAL='#008979'; BLUE='#3b6c9e'; PURPLE='#9464a7'; ORANGE='#d57c39'
GRAY='#8a939b'; INK='#25313d'; RED='#bf5b59'
COLORS=['#008979','#d57c39','#3b6c9e','#9464a7','#b5a138','#bf5b59','#4b9baa']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
    'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.6,'lines.linewidth':1.4,'pdf.fonttype':42,
    'savefig.facecolor':'white','figure.facecolor':'white','text.color':INK,
    'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})

def load(name):return pd.read_csv(D/(name+'.csv'),dtype={'fit_key':str,'base_fit_key':str})
def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight',pad_inches=.035)
    fig.savefig(F/(name+'.png'),dpi=250,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
def panel(ax,letter,title=None):
    ax.text(-.12,1.07,letter,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
    if title:ax.set_title(title,loc='left',pad=7)
def clean(ax):ax.grid(axis='y',alpha=.13,lw=.5);ax.set_axisbelow(True)
def macro(d):
    return d.assign(ae=abs(d.predicted-d.observed)).groupby(
        ['family','sample_id']).ae.mean().groupby('family').mean().mean()

def rotate(points):
    a=np.deg2rad(20);b=np.deg2rad(-17)
    rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    rx=np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
    return points@rz.T@rx.T

def ballstick(ax,xyz,symbols,bonds,central=None,scale=1.):
    p=rotate(np.asarray(xyz)); p=p-p.mean(axis=0)
    for i,j in bonds:
        ax.plot(p[[i,j],0],p[[i,j],1],color='#c4c9cd',lw=1.8*scale,zorder=-5)
    order=np.argsort(p[:,2]);cs={'Ga':BLUE,'O':RED,'Si':TEAL}
    for i in order:
        radius=(55 if symbols[i]=='Ga' else 27)*scale**2
        ax.scatter(p[i,0],p[i,1],s=radius,c=cs[symbols[i]],edgecolor='white',linewidth=.35,zorder=2+p[i,2]*.01)
        ax.scatter(p[i,0]-.035,p[i,1]+.035,s=radius*.18,c='white',alpha=.24,edgecolor='none',zorder=3+p[i,2]*.01)
    ax.set_aspect('equal');ax.axis('off');ax.margins(.1)

def architecture():
    from figure1_architecture import architecture as draw
    # Restore the shared plot style after the architecture-specific style.
    with plt.rc_context():
        draw()


def energy():
    es=load('energy_learning_summary');es=es[es.stratum.eq('all')].set_index('method')
    ps=load('energy_property_summary');ps=ps[ps.scope.eq('all_query')&ps.budget.eq(0)].set_index('method')
    a=load('energy_acquisition_summary')
    names={'extra_trees':'Extra Trees','ridge':'Ridge','rbf':'RBF','global_mean':'Global mean','charge_pair_mean':'Charge-pair mean'}
    fig,axs=plt.subplots(2,2,figsize=(7.25,5.4),layout='constrained')
    ax=axs[0,0];panel(ax,'a','Energy accuracy does not rank carrier accuracy')
    offsets={'extra_trees':(7,3),'ridge':(-27,-13),'rbf':(6,-2),'global_mean':(-53,-15),'charge_pair_mean':(5,5)}
    for k,col in zip(names,[PURPLE,TEAL,ORANGE,GRAY,BLUE]):
        x=es.loc[k,'element_parent_MAE_eV'];y=ps.loc[k,'MAE']
        ax.scatter(x,y,s=35,c=col,zorder=4)
        ax.annotate(names[k],(x,y),xytext=offsets[k],textcoords='offset points',fontsize=7,color=col)
    ax.set(xlabel='Formation-intercept MAE (eV)',ylabel='Carrier MAE (dex)',xlim=(1.04,1.65),ylim=(.28,.445));clean(ax)
    ax=axs[0,1];panel(ax,'b','Prediction against published carrier data')
    for pref,label,color,style in [('ET_goal','Property-directed',TEAL,'-'),('ET_variance','Energy variance',ORANGE,'--')]:
        yy=[ps.loc['extra_trees','MAE']]+[ps.loc[f'{pref}_k{k}','MAE'] for k in [2,4]]
        ax.plot([0,2,4],yy,'o'+style,color=color,label=label,ms=4)
    for seed in [0,1]:
        yy=[ps.loc['extra_trees','MAE']]+[ps.loc[f'ET_random{seed}_k{k}','MAE'] for k in [2,4]]
        ax.plot([0,2,4],yy,'o:',color=GRAY,alpha=.7,label='Random: two replays' if seed==0 else None,ms=3)
    ax.axhline(ps.loc['oracle_full','MAE'],color=INK,lw=.9,ls='--',label='Full supplied catalogue')
    ax.scatter([2],[ps.loc['Ridge_diagonal_xi_k2','MAE']],marker='D',s=22,color=PURPLE,label='Ridge + diagonal sensitivity',zorder=7)
    ax.set(xlabel='Additional revealed energy labels per dopant',ylabel='Carrier MAE (dex)',xticks=[0,2,4],ylim=(.245,.37))
    ax.legend(loc='upper right',fontsize=6.2,frameon=False);clean(ax)
    for ax,el,letter in zip(axs[1],['Si','Sn'],['c','d']):
        panel(ax,letter,el+': separate virtual property queries')
        sub=a[a.element.eq(el)&a.model.eq('extra_trees')]
        for pol,label,col,style in [('goal','Property-directed',TEAL,'-'),('energy_variance','Energy variance',ORANGE,'--'),('random','Random',GRAY,':')]:
            s=sub[sub.policy.eq(pol)].sort_values('budget')
            ax.plot(s.budget,s.MAE_mean,'o'+style,color=col,ms=3,label=label)
            if pol=='random':ax.fill_between(s.budget,s.MAE_min,s.MAE_max,color=GRAY,alpha=.16,label='32-replay range')
        ax.set(xlabel='Additional revealed energy labels',ylabel='Discrepancy from full catalogue (dex)',xticks=[0,2,4,6,8],ylim=(-.025,float(sub.MAE_max.max())+.05))
        ax.legend(loc='upper right',frameon=False,fontsize=6.5);clean(ax)
    save(fig,'fig2_property_directed')

def room_temperature():
    rt=load('room_temperature_condition_predictions');src=load('room_temperature_source_scores')
    labels={'10.1002/aelm.202300679':'Tarntair et al.', '10.1007/s10853-015-9693-6':'Baldini et al.',
        '10.1007/s40843-022-2167-x':'Xiang et al.', '10.1063/5.0027870':'Mauze et al.',
        '10.1063/1.5047214':'Leedy et al.', '10.1063/5.0101132':'McCandless et al.'}
    families=list(labels)
    fig,axs=plt.subplots(2,2,figsize=(7.25,5.5),layout='constrained')
    ax=axs[0,0];panel(ax,'a','Prediction without target-source Hall labels')
    ctrl=load('original_prediction_controls');ctrl=ctrl[ctrl.scope.eq('common20')&ctrl.budget.eq(0)].set_index('variant')
    cap=load('capacity_control_scores');cap=cap[cap.scope.eq('common20')].set_index('method')
    vals=[cap.loc['DirectQuadratic6','MAE'],cap.loc['ReducedInventory6','MAE'],
        ctrl.loc['DirectMultiRidge_offset','nH_MAE'],ctrl.loc['ActivationProcessPair_offset','nH_MAE'],
        src[src.variant.eq('FixedNHReference')].MAE.mean(),src[src.variant.eq('LearnedNHInventory')].MAE.mean()]
    names=['Direct quadratic (6)','Reduced inventory (6)','Direct Ridge','Empirical process','Fixed inventory (2)','Learned inventory (6)']
    ax.barh(np.arange(6),vals,color=[GRAY,GRAY,GRAY,BLUE,BLUE,TEAL],height=.65)
    for j,v in enumerate(vals):ax.text(v+.01,j,f'{v:.3f}',va='center',fontsize=7)
    ax.set(yticks=np.arange(6),yticklabels=names,xlabel='Source-macro carrier MAE (dex)',xlim=(0,.76));ax.invert_yaxis();ax.grid(axis='x',alpha=.13)
    ax=axs[0,1];panel(ax,'b','All 20 common-query chemical conditions')
    sub=rt[rt.variant.eq('LearnedNHInventory')]
    ax.plot([15.5,21],[15.5,21],color=GRAY,lw=.8)
    for family,col in zip(families,COLORS):
        s=sub[sub.family.eq(family)];ax.scatter(s.observed,s.predicted,c=col,s=25,edgecolor='white',lw=.4)
    ax.set(xlabel=r'Observed $\log_{10} n_H$ (cm$^{-3}$)',ylabel=r'Predicted $\log_{10} n_H$ (cm$^{-3}$)',xlim=(15.5,21),ylim=(15.5,21));ax.set_aspect('equal');clean(ax)
    ax=axs[1,0];panel(ax,'c','Source-resolved gain and heterogeneity')
    for j,(family,col) in enumerate(zip(families,COLORS)):
        s=src[src.family.eq(family)].set_index('variant');v=[s.loc['FixedNHReference','MAE'],s.loc['LearnedNHInventory','MAE']]
        ax.plot(v,[j,j],color=col,lw=1.8)
        ax.scatter([v[0]],[j],marker='o',facecolor='white',edgecolor=col,s=32,zorder=4)
        ax.scatter([v[1]],[j],marker='o',color=col,s=24,zorder=5)
    ax.set(yticks=np.arange(6),yticklabels=[labels[f] for f in families],xlabel='Carrier MAE (dex)',xlim=(0,1.35));ax.invert_yaxis();ax.grid(axis='x',alpha=.13)
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color=INK,mfc='white',label='Fixed inventory'),Line2D([],[],marker='o',ls='',color=INK,label='Learned inventory')],loc='lower right',frameon=False,fontsize=6.5)
    ax=axs[1,1];panel(ax,'d','Electrical activation over chemical dose')
    for family,col in zip(families,COLORS):
        s=sub[sub.family.eq(family)].sort_values('log_dose')
        ax.scatter(s.log_dose,s.observed-s.log_dose,s=23,facecolor='white',edgecolor=col,lw=.9)
        ax.scatter(s.log_dose,s.predicted-s.log_dose,s=17,marker='x',color=col,lw=1.)
        for row in s.itertuples():ax.plot([row.log_dose]*2,[row.observed-row.log_dose,row.predicted-row.log_dose],color=col,alpha=.35,lw=.6)
    ax.axhline(0,color=GRAY,lw=.7,ls=':')
    ax.set(xlabel=r'$\log_{10} C_X$ (cm$^{-3}$)',ylabel=r'$\log_{10}(n_H/C_X)$',xlim=(17.4,20.9))
    ax.legend(handles=[Line2D([],[],marker='o',ls='',mfc='white',color=INK,label='Measured'),Line2D([],[],marker='x',ls='',color=INK,label='Learned')],frameon=False,fontsize=6.5,loc='lower left');clean(ax)
    save(fig,'fig3_cross_source')

def temperature():
    tp=load('temperature_predictions');obs=load('temperature_observations');pars=load('temperature_parameters')
    samples=list(tp.sample_id.drop_duplicates()); fams=list(tp.family.drop_duplicates())
    fig,axs=plt.subplots(2,2,figsize=(7.25,5.5),layout='constrained')
    for ax,fam,letter,title in zip(axs[0],fams,['a','b'],['Xiang et al.: four Si specimens','McCandless et al.: three Si specimens']):
        panel(ax,letter,title)
        ids=[s for s in samples if s.startswith(fam+'::')]
        for sid,col in zip(ids,COLORS):
            data=tp[tp.sample_id.eq(sid)&tp.method.eq('shared_joint_offset')&tp.budget.eq(0)].sort_values('Tobs_K')
            data=data[data.Tobs_K.between(39.9,301)]
            # Plot the distinct published readings once, not repeated per model.
            ax.scatter(1000/data.Tobs_K,data.observed,c=col,s=12,edgecolor='white',lw=.25,zorder=4)
            ax.plot(1000/data.Tobs_K,data.predicted,color=col,label=sid.split('::')[-1])
            fixed=tp[tp.sample_id.eq(sid)&tp.method.eq('fixed_joint_offset')&tp.budget.eq(0)&tp.Tobs_K.between(39.9,301)].sort_values('Tobs_K')
            ax.plot(1000/fixed.Tobs_K,fixed.predicted,color=col,alpha=.65,ls=':',lw=1)
        ax.axvspan(1000/125,1000/40,color=TEAL,alpha=.045,zorder=-5)
        ax.set(xlabel=r'$1000/T$ (K$^{-1}$)',ylabel=r'$\log_{10} n_H$ (cm$^{-3}$)',xticks=[5,10,15,20,25])
        ax.legend(title='Specimen',title_fontsize=7,frameon=False,ncol=2,loc='lower left',fontsize=6.5);clean(ax)
    ax=axs[1,0];panel(ax,'c','A learned concentration-dependent donor scale')
    thermal_doses=obs[obs.kind.eq('curve')].drop_duplicates('sample_id').incorporated_cm3.to_numpy()
    lo,hi=np.log10(thermal_doses.min()),np.log10(thermal_doses.max())
    logC=np.linspace(min(16.1,lo-.15),20,300)
    ax.axvspan(lo,hi,color=GRAY,alpha=.09,zorder=-5)
    marker_records=[]
    for fam,col,lab in zip(fams,[TEAL,BLUE],['Xiang held out','McCandless held out']):
        row=pars[pars.training_sources.eq(4)&pars.method.eq('learned_screen_joint')&pars.excluded_sources.eq(fam)].iloc[0]
        E=1000*row.binding_E0_eV*np.exp(-row.concentration_s*(10**logC/1e18)**(1/3))
        ax.plot(logC,E,c=col,ls='--',lw=1.,alpha=.65)
        within=(logC>=lo)&(logC<=hi)
        ax.plot(logC[within],E[within],c=col,label=lab)
        used=obs[obs.family.eq(fam)&obs.kind.eq('curve')].drop_duplicates('sample_id')
        xx=np.log10(used.incorporated_cm3);yy=1000*row.binding_E0_eV*np.exp(-row.concentration_s*(10**xx/1e18)**(1/3))
        ax.scatter(xx,yy,c=col,s=22,edgecolor='white',lw=.4,zorder=4)
        marker_records.extend(dict(sample_id=sid,log10_C=float(cx),energy_meV=float(ey))
                              for sid,cx,ey in zip(used.sample_id,xx,yy))
    ax.axhline(36.6,color=GRAY,ls=':',lw=1,label='Fixed donor scale')
    ax.set(xlabel=r'$\log_{10} C_{\rm Si}$ (cm$^{-3}$)',ylabel=r'Effective $E_D(C)$ (meV)',ylim=(-2,110))
    assert len(marker_records)==7 and all(-2<r['energy_meV']<110 for r in marker_records)
    (OUT/'audit/figure4_domain.json').write_text(json.dumps(dict(
        physical_specimens=7,dose_min_cm3=float(min(thermal_doses)),dose_max_cm3=float(max(thermal_doses)),
        plotted_energy_limits_meV=[-2,110],all_markers_visible=True,markers=marker_records,
        shading='range of seven thermal specimen doses',dashed_curves='extrapolation outside that range'),indent=2)+'\n')
    ax.legend(frameon=False,fontsize=6.5);clean(ax)
    ax=axs[1,1];panel(ax,'d','Source-excluded temperature prediction')
    all_scores=load('temperature_scores');all_scores=all_scores[all_scores.task.eq('temperature')&all_scores.budget.eq(0)]
    scores=all_scores[all_scores.scope.eq('common40_125')].set_index('method')
    # The constant-energy ablation is from the same phase9 joint-input protocol.
    phase9=load('temperature_base_model_scores')
    val=.5195078607714807
    if 'method' in phase9 and 'scope' in phase9:
        z=phase9[phase9.method.eq('learned_binding_joint')&phase9.scope.eq('common40_125')&phase9.budget.eq(0)]
        if len(z):val=float(z.iloc[0]['MAE'])
    values=[scores.loc['fixed_joint_offset','MAE'],val,scores.loc['direct_joint_affine','MAE'],scores.loc['shared_joint_offset','MAE']]
    full=all_scores[all_scores.scope.eq('all_query')].set_index('method')
    const=phase9[phase9.method.eq('learned_binding_joint')&phase9.scope.eq('all_query')&phase9.budget.eq(0)].iloc[0].MAE
    values_full=[full.loc['fixed_joint_offset','MAE'],const,full.loc['direct_joint_affine','MAE'],full.loc['shared_joint_offset','MAE']]
    for j,(v,vf,col) in enumerate(zip(values,values_full,[GRAY,BLUE,PURPLE,TEAL])):
        ax.barh(j-.17,vf,color=col,height=.30)
        ax.barh(j+.17,v,color=col,alpha=.34,height=.30)
        ax.text(vf+.012,j-.17,f'{vf:.3f}',va='center',fontsize=6.4)
        ax.text(v+.012,j+.17,f'{v:.3f}',va='center',fontsize=6.4,color=GRAY)
    ax.set(yticks=range(4),yticklabels=['Fixed donor energy','Learned constant energy','Tuned direct Ridge','Learned $E_D(C)$'],xlabel='Carrier MAE (dex)',xlim=(0,1.03));ax.invert_yaxis();ax.grid(axis='x',alpha=.13)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=INK,label='Full: 142 readings'),Patch(facecolor=INK,alpha=.34,label='Common: 80 readings')],loc='lower right',frameon=False,fontsize=6)
    save(fig,'fig4_temperature')

def adaptation():
    tp=load('temperature_predictions');sens=load('temperature_sensitivities')
    scores=load('temperature_scores');scores=scores[scores.task.eq('temperature')&scores.scope.eq('common40_125')]
    fig=plt.figure(figsize=(7.25,3.0));gs=fig.add_gridspec(1,3,width_ratios=[1.05,1.1,1],wspace=.52)
    ax=fig.add_subplot(gs[0]);panel(ax,'a','Internal sensitivity')
    low=sens[sens.kind.eq('curve')].copy(); ids=list(low.sample_id)
    a=[];b=[]
    for sid in ids:
        a.append(abs(sens[sens.sample_id.eq(sid)&sens.kind.eq('RT')].iloc[0].dlogn_dlogE0))
        b.append(abs(low[low.sample_id.eq(sid)].iloc[0].dlogn_dlogE0))
    x=np.arange(len(ids));ax.bar(x-.17,np.maximum(a,1e-8),width=.34,color=GRAY,label='300 K');ax.bar(x+.17,b,width=.34,color=TEAL,label='Lowest measured $T$')
    ax.set(yscale='log',xticks=x,xticklabels=[s.split('::')[-1] for s in ids],ylabel=r'$|\partial\log_{10} n_H/\partial\log_{10} E_0|$',xlabel='Si specimen (A–D; 1–3)')
    ax.legend(frameon=False,fontsize=6.5,loc='lower left');clean(ax)
    ax=fig.add_subplot(gs[1]);panel(ax,'b','Internal adaptation')
    ms=[('shared_joint_offset','Output shift',GRAY,'s'),('shared_joint_Tf_E','Internal $T_f,E_0$',TEAL,'o'),('direct_joint_affine','Direct Ridge + affine',PURPLE,'^')]
    for method,label,col,marker in ms:
        sub=scores[scores.method.eq(method)].sort_values('budget')
        ax.plot(sub.budget,sub.MAE,color=col,marker=marker,ms=4,label=label)
    ax.scatter([2],[.2305200770971113],marker='D',color=ORANGE,s=28,label=r'Anchor-only $a+b/T$',zorder=5)
    ax.set(xticks=[0,1,2],xlabel='Temperature anchors per specimen',ylabel='Common-query carrier MAE (dex)',ylim=(.185,.355))
    ax.legend(frameon=False,fontsize=6.0,loc='upper right');clean(ax)
    ax=fig.add_subplot(gs[2]);panel(ax,'c','Held-out sources')
    labels=['Xiang','McCandless'];fams=list(tp.family.drop_duplicates());xx=np.arange(2)
    for j,(m,lab,col) in enumerate([('shared_joint_offset','Output shift',GRAY),('direct_joint_affine','Ridge + affine',PURPLE),('shared_joint_Tf_E','Internal $T_f,E_0$',TEAL)]):
        s=tp[tp.method.eq(m)&tp.budget.eq(2)&tp.common40_125.eq(True)&tp.role.eq('query')].assign(ae=lambda d:abs(d.predicted-d.observed))
        y=[s[s.family.eq(f)].groupby('sample_id').ae.mean().mean() for f in fams]
        ax.bar(xx+(j-1)*.23,y,width=.21,color=col,label=lab)
    ax.set(xticks=xx,xticklabels=labels,ylabel='Two-anchor MAE (dex)',ylim=(0,.34));ax.legend(frameon=False,fontsize=6.0,loc='upper right');clean(ax)
    fig.subplots_adjust(left=.08,right=.985,top=.85,bottom=.20)
    save(fig,'fig5_internal_adaptation')

def supplementary():
    tp=load('temperature_predictions');ids=list(tp.sample_id.drop_duplicates())
    fig,axs=plt.subplots(2,4,figsize=(7.25,4.5),layout='constrained')
    for ax,sid,letter in zip(axs.flat,ids,'abcdefg'):
        panel(ax,letter,('Xiang ' if 's40843' in sid else 'McCandless ')+sid.split('::')[-1])
        for m,k,col,ls in [('shared_joint_offset',0,BLUE,':'),('shared_joint_offset',2,GRAY,'--'),('shared_joint_Tf_E',2,TEAL,'-')]:
            d=tp[tp.sample_id.eq(sid)&tp.method.eq(m)&tp.budget.eq(k)].sort_values('Tobs_K')
            ax.plot(1000/d.Tobs_K,d.predicted,color=col,ls=ls)
            if m=='shared_joint_Tf_E':
                ax.scatter(1000/d.Tobs_K,d.observed,s=9,c=INK,edgecolor='white',lw=.2)
                a=d[d.role.eq('anchor')];ax.scatter(1000/a.Tobs_K,a.observed,s=38,marker='*',c=ORANGE,zorder=5)
        ax.set(xlabel=r'$1000/T$ (K$^{-1}$)',ylabel=r'$\log_{10} n_H$');clean(ax)
    axs.flat[-1].axis('off');axs.flat[-1].legend(handles=[
        Line2D([],[],color=BLUE,ls=':',label='No target-temperature anchors'),
        Line2D([],[],color=GRAY,ls='--',label='Two anchors: output shift'),
        Line2D([],[],color=TEAL,label='Two anchors: internal adaptation'),
        Line2D([],[],ls='',marker='o',color=INK,label='Published observations'),
        Line2D([],[],ls='',marker='*',color=ORANGE,label='Calibration anchors')],loc='center',frameon=False,fontsize=7)
    save(fig,'figS1_all_temperature_adaptations')
    fig,axs=plt.subplots(1,2,figsize=(7.25,2.8),layout='constrained')
    d=load('room_temperature_anchor_controls');print('anchor schema',list(d.columns))
    ax=axs[0];panel(ax,'a','Room-temperature sample anchors')
    if {'variant','scope','budget','nH_MAE'}.issubset(d.columns):
        for m,lab,col in [('FrozenShared','Frozen shared',BLUE),('NestedShift','Output shift',GRAY),('NestedProfiledGate','Profiled information gate',TEAL),('NestedPhysicsOrDose','Training-only branch choice',PURPLE),('NestedUnitDose','Unit-slope dose',ORANGE)]:
            s=d[d.variant.eq(m)&d.scope.eq('common20')].sort_values('budget')
            if len(s):ax.plot(s.budget,s.nH_MAE,'o-',ms=3,color=col,label=lab)
    ax.set(xlabel='Room-temperature sample anchors',ylabel='Common-query MAE (dex)',xticks=[0,1,2]);ax.legend(frameon=False,fontsize=6);clean(ax)
    ax=axs[1];panel(ax,'b','Closing controls under the same prediction contract')
    h=load('hinge_control_inventory_training_v1');h=h[h.cohort.eq('old_cohort_heldout')&h.scope.eq('common20')]
    if len(h):
        h=h.set_index('variant'); names=['FixedNHReference','LinearShared','DiluteHinge']
        vals=[h.loc[k,'nH_MAE'] for k in names];ax.bar(range(3),vals,color=[GRAY,TEAL,ORANGE]);ax.set(xticks=range(3),xticklabels=['Fixed','Linear share','Soft hinge'],ylabel='Carrier MAE (dex)',ylim=(0,.5))
        for j,v in enumerate(vals):ax.text(j,v+.01,f'{v:.3f}',ha='center',fontsize=7)
    clean(ax);save(fig,'figS2_room_anchor_and_hinge')

if __name__=='__main__':
    architecture();energy();room_temperature();temperature();adaptation();supplementary()
    from summarize_submission_validation import figures as validation_figures
    validation_figures()
    print('Generated',len(list(F.glob('*.pdf'))),'vector figures')
