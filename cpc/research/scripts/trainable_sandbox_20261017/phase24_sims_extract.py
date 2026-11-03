"""Read twelve chemical layer interiors from the original Fig3 raster; no Hall labels."""
from pathlib import Path
import sys,json,io
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from PIL import Image
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';DEST=OUT/'extraction'
PDF=BASE/'phase22_predictive_inference/source_leads/Azizie2023.pdf'
PANELS={
    'm':dict(fig='3b',x_pixels=[903.,1632.],x_um=[0.,7.],y_pixels=[17.,583.],y_logC=[20.,13.],velocity_um_h=.86,
        Tsource_nominal_C=[1282.3,1237.1,1194.7,1152.9,1116.6,1080.4],
        doped_windows_um=[[.50,1.08],[1.48,2.06],[2.43,3.01],[3.35,3.94],[4.26,4.86],[5.11,5.70]]),
    'n':dict(fig='3d',x_pixels=[900.,1632.],x_um=[0.,10.],y_pixels=[738.,1305.],y_logC=[20.,13.],velocity_um_h=1.3,
        Tsource_nominal_C=[1285.4,1238.1,1193.7,1150.3,1112.5,1074.9],
        doped_windows_um=[[.60,1.26],[1.86,2.52],[3.08,3.78],[4.34,5.03],[5.60,6.32],[6.88,7.56]])}
TARGET=[1e19,2.5e18,6.3e17,1.5e17,4e16,1e16]


def freeze(image_bytes):
    import hashlib
    p=dict(created_utc=e.now(),previous_goal_turn='progress:phase23 actualnewsourceforecasts and422artifactfreeze',
        purpose='independent chemical supervision before process-to-property learning',
        input_hashes={str(PDF.relative_to(ROOT)):e.sha(PDF),str((BASE/'artifact_manifest_phase23.json').relative_to(ROOT)):e.sha(BASE/'artifact_manifest_phase23.json')},
        code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))},figure_page_1indexed=7,native_image_size=[1649,1427],
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),panels=PANELS,nominal_target_cm3=TARGET,
        scenarios={'central50':.50,'central25':.25,'central75':.75},primary='central50',
        color_rule='R>150,75<G<225,B<110,R>1.1G,G>1.5B in panel ROI;median y percolumn;all raw spans retained',
        label='log10 arithmeticmean concentration of uniformly spaced image columns in declared interior',
        source_temperature='published nominal settings in targeted-layer schematic,not independently traced instrumentvalues',
        uncertainty='curveinteriorvariation pluspixelstroke/axis±1px envelopes;not experimentalCI',
        sample_structure='12 layers/2stacks/1DOI,not12 independentgrowthsources',
        Hall_labels_read=False,no_clock_learning_from_steady_levels=True,not_blind=True,goal_complete=False)
    path=DEST/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['code_hashes']==p['code_hashes'] and old['input_hashes']==p['input_hashes']
    else:e.write(path,p)


def logC(y,p,dy0=0.,dy1=0.):
    a,b=p['y_pixels'];lo,hi=p['y_logC'];return lo+(np.asarray(y)-a-dy0)*(hi-lo)/(b+dy1-a-dy0)


def main():
    d=fitz.open(PDF);items=d[6].get_images();assert len(items)==1;im=d.extract_image(items[0][0]);blob=im['image'];a=np.asarray(Image.open(io.BytesIO(blob)).convert('RGB'),float)
    assert a.shape==(1427,1649,3);freeze(blob);(DEST/'Fig3_original.png').write_bytes(blob)
    rows=[];traces=[];rawpoints=[];checks=[];fig,axes=plt.subplots(2,1,figsize=(10,7),sharey=True)
    for ax,(stack,p) in zip(axes,PANELS.items()):
        x0,x1=p['x_pixels'];xmin,xmax=p['x_um'];roi=(max(0,int(p['y_pixels'][0])),min(a.shape[0],int(p['y_pixels'][1])+18))
        mask=(a[:,:,0]>150)&(a[:,:,1]>75)&(a[:,:,1]<225)&(a[:,:,2]<110)&(a[:,:,0]>1.1*a[:,:,1])&(a[:,:,1]>1.5*a[:,:,2])
        trace=[]
        for x in range(int(x0),int(x1)+1):
            yy=np.flatnonzero(mask[roi[0]:roi[1],x])+roi[0]
            if not len(yy):continue
            z=xmin+(x-x0)*(xmax-xmin)/(x1-x0);y=float(np.median(yy))
            row=dict(stack_id=stack,pixel_x=x,pixel_y=y,pixel_y_min=float(yy.min()),pixel_y_max=float(yy.max()),depth_um=z,
                log10_C= float(logC(y,p)),stroke_pixels=len(yy),stroke_span=float(np.ptp(yy)),no_Hall_target_used=True)
            trace.append(row);traces.append(row);rawpoints.extend(dict(stack_id=stack,pixel_x=x,pixel_y=int(v),R=int(a[v,x,0]),G=int(a[v,x,1]),B=int(a[v,x,2])) for v in yy)
        ax.plot([r['depth_um'] for r in trace],[r['log10_C'] for r in trace],color='#d69b00',lw=1,label='Si raster trace')
        for i,(T,win) in enumerate(zip(p['Tsource_nominal_C'],p['doped_windows_um'])):
            mid=float(np.mean(win));width=win[1]-win[0]
            for scenario,fraction in [('central50',.50),('central25',.25),('central75',.75)]:
                lo=mid-width*fraction/2;hi=mid+width*fraction/2;q=[r for r in trace if lo<=r['depth_um']<=hi]
                assert len(q)>3,(stack,i,scenario)
                y=np.array([r['pixel_y'] for r in q]);mu=float(np.log10(np.mean(10**logC(y,p))))
                bounds=[]
                for dy0 in (-1.,1.):
                    for dy1 in (-1.,1.):
                        bounds.append(float(np.log10(np.mean(10**logC([r['pixel_y_min'] for r in q],p,dy0,dy1)))))
                        bounds.append(float(np.log10(np.mean(10**logC([r['pixel_y_max'] for r in q],p,dy0,dy1)))))
                expected=int(np.floor(x0+(hi-xmin)*(x1-x0)/(xmax-xmin))-np.ceil(x0+(lo-xmin)*(x1-x0)/(xmax-xmin))+1)
                coverage=len(q)/expected;checks.append(dict(stack_id=stack,layer=i,scenario=scenario,coverage=coverage,passed=coverage>=.85))
                rows.append(dict(layer_id=f'10.1063/5.0139622::{stack}::layer_surface_{i+1}',stack_id=stack,source_DOI='10.1063/5.0139622',
                    scenario=scenario,surface_order=i+1,growth_order=6-i,Tsource_nominal_C=T,velocity_um_h=p['velocity_um_h'],
                    Tsub_reported_C=525.,oxygen_source='distilledO3;not equilibriumO2',nominal_target_cm3=TARGET[i],
                    interior_min_um=lo,interior_max_um=hi,columns=len(q),coverage=coverage,
                    log10_C_chemical=mu,C_chemical_cm3=10**mu,log10_C_median=float(np.median(logC(y,p))),
                    log10_reading_lower=min(bounds),log10_reading_upper=max(bounds),within_interior_log_range=float(np.ptp(logC(y,p))),
                    observed_to_target_ratio=10**mu/TARGET[i],chemical_grade='SIMS graph interior;not raw calibrated values or same-Hall specimen',
                    source_temperature_grade='targeted layer nominal setting;actual logged history not paired',
                    source_locator=f'PDFp7 Fig{p["fig"]} and schematic3{chr(ord("a")+2*(stack=="n"))}',independent_physical_stacks=2))
                if scenario=='central50':
                    ax.axvspan(lo,hi,color='#2864a0',alpha=.12);ax.errorbar([mid],[mu],yerr=[[mu-min(bounds)],[max(bounds)-mu]],fmt='o',color='#2864a0',ms=4)
                    ax.annotate(f'{T:.1f} C',(mid,mu),xytext=(0,9),textcoords='offset points',ha='center',fontsize=7)
        ax.set(xlim=(0,7 if stack=='m' else 10),ylim=(13,20),ylabel=r'$\log_{10} C_{\rm SIMS}\,[{\rm cm}^{-3}]$',title=f'Stack {stack}: six layer interiors; nominal source settings')
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Sputtered depth (um)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(DEST/f'chemical_digitization.{ext}',dpi=180)
    plt.close(fig);e.csvwrite(DEST/'layer_readings.csv',rows);e.csvwrite(DEST/'Si_trace.csv',traces);e.csvwrite(DEST/'Si_pixels.csv',rawpoints)
    e.write(DEST/'extraction_audit.json',dict(passed=len(rows)==36 and all(q['passed'] for q in checks),checks=checks,rows=36,chemical_conditions=12,
        independent_stacks=2,independent_DOI=1,Hall_targets_used=False,new_carrier_labels=0,goal_complete=False))
    print('SIMS extraction',len(rows),'scenario layerreadings,12chemicalconditions/2stacks/1DOI',flush=True)
    for r in rows:
        if r['scenario']=='central50':print(r['stack_id'],r['Tsource_nominal_C'],f"{r['C_chemical_cm3']:.6g}",'targetratio',round(r['observed_to_target_ratio'],3),flush=True)


if __name__=='__main__':main()
