"""Generate manuscript supplementary tables from curated source CSV files."""
from pathlib import Path
import sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/predictive_redesign_20261019'))
OUT=(ROOT/'paper/npj/ml_sandbox_manuscript_v3' if Path(__file__).resolve().parent.name=='manuscript_ml_sandbox_v3'
     else Path(__file__).resolve().parent.parent)
D=OUT/'source_data';T=OUT/'tables';T.mkdir(exist_ok=True)
def load(n):
    return pd.read_csv(D/(n+'.csv'),dtype={k:str for k in (
        'fit_key','base_fit_key','sample_id','case_id','family','related_cluster',
        'excluded_sources','variant','method','model','scope','cohort')})
def esc(s):return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')
def f(v,n=4):return '---' if pd.isna(v) else f'{float(v):.{n}f}'
def table(name,heads,rows,caption,label,align=None,long=False):
    align=align or 'l'+'r'*(len(heads)-1)
    body=' & '.join(heads)+r' \\'+'\n'+r'\midrule'+'\n'
    body+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    if long:
        # The curated tables fit on one page; a floating unit keeps the
        # caption/header with its data instead of splitting it at a page foot.
        text=r'\begin{table}[htbp]\centering\footnotesize'+'\n'+r'\caption{'+caption+r'}\label{'+label+r'}'+'\n'+r'\begin{tabular}{'+align+'}\n'+r'\toprule'+'\n'+body+'\n'+r'\bottomrule\end{tabular}\end{table}'
    else:
        text=r'\begin{table}[htbp]\centering\small'+'\n'+r'\caption{'+caption+r'}\label{'+label+r'}'+'\n'+r'\begin{tabular}{'+align+'}\n'+r'\toprule'+'\n'+body+'\n'+r'\bottomrule\end{tabular}\end{table}'
    (T/(name+'.tex')).write_text(text+'\n')

names={'10.1002/aelm.202300679':'Tarntair et al.', '10.1007/s10853-015-9693-6':'Baldini et al.',
       '10.1007/s40843-022-2167-x':'Xiang et al.', '10.1063/5.0027870':'Mauze et al.',
       '10.1063/1.5047214':'Leedy et al.', '10.1063/5.0101132':'McCandless et al.',
       '10.1039/c5ce01106j':'Gogova et al.'}

d=load('room_temperature_sample_contract');rows=[]
for fam,s in d.groupby('family',sort=True):
    rows.append([names[fam],str(len(s)),str(int(s.primary_common20.eq(True).sum())),
       '/'.join(s.element.unique()),f'{s.log_dose.min():.2f}--{s.log_dose.max():.2f}',
       ('AFRL/Cornell' if s.related_cluster.iloc[0]=='related_AFRL_Cornell_Wright' else
        'IKZ' if s.related_cluster.iloc[0]=='10.1007/s10853-015-9693-6' else names[fam].replace(' et al.',''))])
table('sample_sources',['Source','All','Common','Dopant',r'$\log_{10} C_X$','Exclusion group'],rows,
 r'Literature sources, doping-condition counts, incorporated-dose ranges and related-source exclusion groups for the room-temperature cohort. The All and Common columns report distinct physical doping conditions; sample identities and primary extraction locators are archived in the companion CSV.',
 'tab:Ssources',align='lrrlll')

ctrl=load('original_prediction_controls');ctrl=ctrl[ctrl.scope.eq('common20')].set_index(['variant','budget'])
labels=[('UnitDose_offset','Unit-slope dose'),('MedianActivation_offset','Median activation'),('DoseLine_offset','Fitted dose line'),('ActivationProcessPair_offset','Empirical process'),('DirectMultiRidge_affine','Direct Ridge + affine'),('Reference205_nHonly_internal','Earlier fixed inventory, carrier-only'),('JointConstant_internal','Earlier constant allocation, joint'),('ShareDose205_internal','Earlier dose allocation, joint'),('JointProcess_nHonly_internal','Earlier process allocation, carrier-only'),('JointProcess_internal','Earlier process allocation, joint')]
rows=[]
for k,label in labels:rows.append([label]+[f(ctrl.loc[(k,b),'nH_MAE']) for b in [0,1,2]])
trees=load('strong_tree_controls');trees=trees[trees.scope.eq('common20')]
for k,label in [('DirectMultiExtraTrees__nH','Direct Extra Trees'),('DoseResidualExtraTrees__nH','Dose-residual Extra Trees')]:
    s=trees[trees.variant.eq(k)].set_index('budget');rows.append([label]+[f(s.loc[b,'nH_MAE']) for b in [0,1,2]])
rows += [['Matched overlay fixed inventory',f(.41476808105339447),'---','---'],['Matched overlay learned inventory',f(.3377212625661608),'---','---']]
table('room_controls',['Model','0 anchors','1 anchor','2 anchors'],rows,
 r'Source-macro carrier MAE (dex) on the 20 common room-temperature queries across target-source sample-anchor budgets. Earlier baselines follow their frozen energetic and tuning protocols; the final two rows compare fixed and learned inventory allocation under an identical multi-donor overlay and carrier-only supervision.',
 'tab:Sroom',align='lrrr')

e=load('energy_property_summary');e=e[e.scope.eq('all_query')];rows=[]
for m,s in e.groupby('method'):
    s=s.set_index('budget');rows.append([esc(m)]+[f(s.loc[b,'MAE']) for b in [0,1,2]])
table('energy_controls',['Energetic model / acquisition','0 anchors','1 anchor','2 anchors'],rows,
 r'Source-macro carrier MAE (dex) across all energetic surrogates and label-acquisition policies on the 29-condition cohort. Columns correspond to target-source carrier-anchor budgets of 0, 1 and 2 (29, 23 and 18 query conditions), whereas suffixes k2 and k4 denote the number of revealed energy labels per dopant.',
 'tab:Senergy',align='lrrr',long=True)

z=load('temperature_scores');z=z[z.task.eq('temperature')]
rows=[]
for m,s in z.groupby('method'):
    s=s.set_index(['scope','budget']);rows.append([esc(m)]+[f(s.loc[(scope,b),'MAE']) for scope in ['common40_125','all_query'] for b in [0,1,2]])
table('temperature_controls',['Model','Common 0','Common 1','Common 2','Full 0','Full 1','Full 2'],rows,
 r'Source-macro carrier MAE (dex) for temperature-dependent prediction and adaptation. Common columns evaluate the fixed 80 low-temperature readings (40--125 K) at every budget; Full columns evaluate all 142, 138 and 131 non-anchor query readings at budgets 0, 1 and 2.',
 'tab:Sthermal',align='lrrrrrr',long=True)

p=load('temperature_parameters');p=p[p.training_sources.eq(4)&p.method.eq('learned_screen_joint')&p.excluded_sources.isin(['10.1007/s40843-022-2167-x','10.1063/5.0101132'])]
rows=[[names[r.excluded_sources],f(r.Tf_K,1),f(r.log10_background,2),f(r.binding_E0_eV,5),f(r.concentration_s,5)] for r in p.itertuples()]
table('temperature_fits'.strip(),['Excluded source',r'$T_f$ (K)',r'$\log_{10}N_b$',r'$E_0$ (eV)',r'$s$'],rows,
 r'Fitted shared parameters of the concentration-dependent donor model under each outer publication exclusion. In both folds $\log_{10}N_b$ settles at its lower bound ($14.00$), and in the Xiang-excluded fold $E_0$ approaches its upper bound ($0.20\,\mathrm{eV}$). The two effective $E_D(C_X)$ curves approach one another at the higher sampled doses and differ at the dilute end; main-text Fig.~5(c) displays the full seven-specimen dose range.',
 'tab:Stparams',align='lrrrr')

p=load('spatial_candidate_summary');p=p[p.scope.eq('common20')&p.budget.eq(0)&p.adaptation.eq('none')]
rows=[[esc(r.method),f(r.alpha,1),f(r.nH_MAE),f(r.mu_MAE)] for r in p.itertuples()]
table('spatial_controls',['Model',r'$\alpha$',r'$n_H$ MAE',r'$\mu_H$ MAE'],rows,
 r'Zero-anchor carrier and mobility MAE (dex) on the common20 cohort for jointly supervised association and spatial-texture models at two response-prior weights $\alpha$. Within common20, four conditions across two related-source groups include paired Hall mobility measurements.',
 'tab:Sspatial',align='lrrr')

p=load('rate_scale_control_rate_scale_v1');p=p[p.scope.eq('common20')|p.cohort.eq('external3_conditional')]
rows=[[('Common20' if r.scope=='common20' else 'External3'),esc(r.variant),f(r.nH_MAE),f(r.mu_MAE)] for r in p.itertuples()]
table('rate_controls',['Cohort','Rate-scale model',r'$n_H$ MAE',r'$\mu_H$ MAE'],rows,
 r'Evaluation of common-rate scattering multipliers on the common20 and three-condition external cohorts. Scaling all relaxation rates at a given condition by a common factor leaves $r_H$ and $n_H$ invariant while rescaling $\mu_H$.',
 'tab:Srate',align='llrr')

p=load('conditional_mobility_ablation')
rows=[[esc(r.model),f(r.source_equal_MAE_dex),str(int(r.improved_families_vs_CT))] for r in p.itertuples()]
table('conditional_mobility',['Conditional mobility model','MAE (dex)','Sources improved / 11'],rows,
 r'Leave-one-publication-out Hall mobility MAE (dex) on the 52-pair, 11-publication conditional transport cohort, where each model receives the measured Hall carrier density $n_H$ alongside growth metadata.',
 'tab:Sconditional',align='lrr',long=True)
print('Generated',len(list(T.glob('*.tex'))),'supplementary tables.')
