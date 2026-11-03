"""Render the numerical-performance figure from complete benchmark records."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output', type=Path, help='Directory for the rendered figure')
args=parser.parse_args()
OUT=args.output or BASE/'figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
    'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
    'axes.spines.right':False,'savefig.bbox':'tight'})
root=BASE/'verification/cpc_numerics'
t=pd.read_csv(root/'timings.csv');g=pd.read_csv(root/'gradient_checks.csv')
s=pd.read_csv(root/'timing_summary.csv');fit=json.loads((BASE/'verification/cpc_workflow/training_comparison.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(7.4,5.5),layout='constrained')
colors={'IFT':'#147D92','FD':'#B75D43'}
for method in ('FD','IFT'):
    d=t[t.method.eq(method)].groupby('parameters').seconds
    med=d.median()*1000;lo=d.quantile(.25)*1000;hi=d.quantile(.75)*1000
    axs[0,0].plot(med.index,med.values,'o-',lw=1.7,color=colors[method],label=method)
    axs[0,0].fill_between(med.index,lo,hi,color=colors[method],alpha=.15)
axs[0,0].set(xlabel='Energetic coordinates, p',ylabel='Jacobian wall time (ms)',yscale='log',title='a  Cost of the state derivative')
axs[0,0].legend(frameon=False,ncol=2)
paths=pd.read_csv(BASE/'verification/gradient_paths_20261031/settings.csv')
settings=[(1100.,3e17),(1100.,3e19),(1500.,3e17),(1500.,3e19)]
for element,offset,color in [('Si',-.18,colors['IFT']),('Sn',.18,colors['FD'])]:
    d=paths[paths.element.eq(element)].set_index(['Tf_K','C_cm3'])
    vals=[100*d.loc[k,'full_vs_fixed_relative_norm'] for k in settings]
    bars=axs[0,1].barh(np.arange(4)+offset,vals,height=.32,color=color,label=element)
    for bar,value in zip(bars,vals):
        axs[0,1].text(value-2,bar.get_y()+bar.get_height()/2,f'{value:.1f}',
                      ha='right',va='center',color='white',fontsize=8)
axs[0,1].set(yticks=np.arange(4),yticklabels=['1100 K, low','1100 K, high','1500 K, low','1500 K, high'],
             xlabel='Fixed-inventory Jacobian error (%)',xlim=(0,105),ylim=(3.65,-1),
             title='b  Preparation-stage sensitivity')
axs[0,1].legend(frameon=False,ncol=2,loc='upper left',fontsize=8)
for label,values in [('Maximum',g.groupby('step').scaled_error.max()),('Median',g.groupby('step').scaled_error.median())]:
    axs[1,0].loglog(values.index,values.values,'o-',label=label,lw=1.5)
axs[1,0].set(xlabel='Central-difference step (eV)',ylabel='Scaled gradient discrepancy',title='c  Accuracy under step refinement')
axs[1,0].legend(frameon=False)
v=[fit['training_ift_seconds'],fit['training_fd_seconds']]
axs[1,1].bar(['IFT','FD'],v,color=[colors['IFT'],colors['FD']],width=.52)
for i,a in enumerate(v):axs[1,1].text(i,a+.8,f'{a:.2f} s',ha='center',fontsize=9)
axs[1,1].set(ylabel='Training wall time (s)',ylim=(0,max(v)*1.32),title='d  Same two-start thermal fit')
axs[1,1].text(.03,.95,f"{fit['speedup']:.2f}× faster\nQuery difference < 2×10⁻⁹ dex",transform=axs[1,1].transAxes,va='top',fontsize=8)
for ax in axs.flat:ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig.savefig(OUT/'fig_cpc_numerics.pdf');fig.savefig(OUT/'fig_cpc_numerics.png',dpi=180)
print(OUT/'fig_cpc_numerics.pdf')
