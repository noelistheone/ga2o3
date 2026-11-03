"""Vector architecture figure: physical computation, learning actions and data flow.

Geometry comes from the archived host CIF. Miniature curves are schematic.
No experimental values, model weights or relaxed defect structures are generated.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
from matplotlib.path import Path as MPath
from scipy.spatial import ConvexHull
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
OUT = (ROOT/'paper/npj/ml_sandbox_manuscript_v3'
       if Path(__file__).resolve().parent.name == 'manuscript_ml_sandbox_v3'
       else Path(__file__).resolve().parent.parent)

INK = '#20364b'
MUTED = '#617381'
LINE = '#ccd6dd'
PHYS = '#3b6c9e'
ENERGY = '#8661a4'
ALLOC = '#008979'
ION = '#ce7836'
OXYGEN = '#ca716a'
FEEDBACK = '#b55d46'

def architecture():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':7.2,
                         'text.color':INK, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    width, height = 7.25, 6.08
    fig = plt.figure(figsize=(width, height), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0)); ax.axis('off')

    def text(x, y, s, size=7.2, color=INK, weight='normal', ha='left', va='center', **kw):
        return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,
                       ha=ha,va=va,linespacing=1.35,**kw)

    def card(x,y,w,h,edge=LINE,fill='white',lw=.75,r=.06):
        obj=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                          facecolor=fill,edgecolor=edge,linewidth=lw,zorder=1)
        ax.add_patch(obj);return obj

    def arrow(points, color=INK, dash=False, lw=1., head=8, z=3):
        pts=np.asarray(points,float)
        path=MPath(pts,[MPath.MOVETO]+[MPath.LINETO]*(len(pts)-1))
        obj=FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=head,
                            lw=lw,color=color,linestyle=(0,(3.2,2.3)) if dash else '-',
                            capstyle='round',joinstyle='round',zorder=z)
        ax.add_patch(obj);return obj

    def badge(x,y,label,color,r=.082):
        ax.add_patch(Circle((x,y),r,fc=color,ec='white',lw=.55,zorder=6))
        text(x,y,label,6.1,'white','bold',ha='center',zorder=7)

    def panel(x,y,label,title):
        text(x,y,label,10.2,weight='bold')
        text(x+.22,y,title,8.6,weight='bold')

    # a. Compact physical context, catalogued information and specimen inputs.
    panel(.10,.17,'a','Material context and supplied inputs')
    card(.10,.36,2.15,.97,fill='#f8fafc')
    card(2.42,.36,2.06,.97,fill='#f8fafc')
    card(4.65,.36,2.50,.97,fill='#f8fafc')
    text(.22,.51,r'$\beta$-Ga$_2$O$_3$',8.0,weight='bold')
    text(2.56,.52,'Energetic catalogue',8.0,weight='bold')
    text(4.79,.52,'Doping and measurement',8.0,weight='bold')
    text(2.56,.77,'Parent structures + charge states',6.9)
    text(2.56,1.00,r'$\{A_{iq},\ q,\ B_{Xi}\}$',9.3,color=ENERGY)
    text(4.79,.77,r'$X,\ C_X,\ T_g,\ p_{\mathrm{O}_2}$',9.3)
    text(4.79,1.02,r'$T$ + documented impurity inputs',7.0)
    text(2.56,1.20,'873 supplied charge states',6.5,color=MUTED)
    text(4.79,1.20,'Chemistry, preparation, observation',6.5,color=MUTED)

    atoms=read(OUT/'source_data/beta_ga2o3_host.cif')
    symbols=np.asarray(atoms.get_chemical_symbols()); xyz=atoms.positions
    dist=atoms.get_all_distances(mic=True)
    ga=np.flatnonzero(symbols=='Ga')
    cn={int(i):int(np.sum((symbols=='O')&(dist[i]<2.35))) for i in ga}
    assert set(cn.values())=={4,6}
    a,b=np.deg2rad(20),np.deg2rad(-17)
    rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    rx=np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
    rot=rz.T@rx.T

    def atomic(points,syms,edges,box,polyhedron=False):
        xx,yy,ww,hh=box
        p=np.asarray(points)@rot
        scale=min(ww/max(np.ptp(p[:,0]),1e-9),hh/max(np.ptp(p[:,1]),1e-9))*.86
        xy=(p[:,:2]-(p[:,:2].min(0)+p[:,:2].max(0))/2)*scale
        xy[:,1]*=-1;xy+=np.array([xx+ww/2,yy+hh/2])
        if polyhedron:
            hull=ConvexHull(np.asarray(points)[1:])
            for tri in hull.simplices:
                ax.add_patch(Polygon(xy[tri+1],fc=PHYS,alpha=.10,ec=PHYS,lw=.4,zorder=2))
        for i,j in edges:
            ax.plot(xy[[i,j],0],xy[[i,j],1],color='#a7b6c2',lw=.75,zorder=3)
        for i in np.argsort(p[:,2]):
            r=(.021 if syms[i]=='Ga' else .015) if len(points)>20 else (.040 if syms[i]=='Ga' else .026)
            c=PHYS if syms[i]=='Ga' else OXYGEN
            z=4+float((p[i,2]-p[:,2].min())/max(np.ptp(p[:,2]),1e-9))*.5
            ax.add_patch(Circle(xy[i],r,fc=c,ec='white',lw=.28,zorder=z))
            ax.add_patch(Circle(xy[i]+[-r*.25,-r*.25],r*.3,fc='white',alpha=.27,ec='none',zorder=z+.01))

    edges=[(int(i),int(j)) for i in ga for j in np.flatnonzero((symbols=='O')&(dist[i]<2.35))
           if np.linalg.norm(xyz[i]-xyz[j])<2.35]
    atomic(xyz,symbols,edges,(.18,.65,.65,.55))
    for x,n,label in [(1.00,4,'Ga(I)'),(1.60,6,'Ga(II)')]:
        i=next(i for i,v in cn.items() if v==n)
        neigh=np.flatnonzero((symbols=='O')&(dist[i]<2.35))
        local=np.vstack([np.zeros(3),[atoms.get_distance(i,j,mic=True,vector=True) for j in neigh]])
        atomic(local,['Ga']+['O']*n,[(0,k) for k in range(1,n+1)],(x,.65,.49,.46),True)
        text(x+.245,1.21,label,6.5,ha='center')
    text(.50,1.21,'80 atoms',6.0,color=MUTED,ha='center')

    # b. The forward map occupies the largest area. Colored ports identify the
    # separately tested substitutions; they are not a single joint network.
    panel(.10,1.61,'b','Differentiable physical sandbox')
    arrow([(3.46,1.34),(3.46,1.94)],color=MUTED,lw=.85)
    arrow([(5.90,1.34),(5.90,1.94)],color=MUTED,lw=.85)
    text(3.58,1.67,'energetics',6.5,color=MUTED)
    text(6.02,1.67,'conditions',6.5,color=MUTED)
    card(.10,1.95,7.05,2.65,edge='#b7c8d5',fill='#f5f8fb',lw=.8,r=.08)
    text(.28,2.10,'FORWARD: FORMATION → FREEZE → MEASUREMENT',7.0,color=PHYS,weight='bold')
    xs=[.40,2.12,3.84,5.56]; ws=[1.40,1.40,1.40,1.40]
    y,h=2.46,1.30
    titles=['Preparation reference','Conserved inventory','Electronic equilibrium','Hall readout']
    for x,w,title in zip(xs,ws,titles):
        card(x,y,w,h,edge='#a9bdcd',fill='white')
        text(x+w/2,y+.16,title,7.1,weight='bold',ha='center')
    # Numbered physical injection points, with shared energy access at both roots.
    for x in [xs[0]+.56,xs[2]+.56]:
        badge(x,2.31,'I',ENERGY);badge(x+.27,2.31,'III',ION)
        arrow([(x+.13,2.40),(x+.13,2.46)],color=MUTED,lw=.65,head=5)
    badge(xs[1]+.7,2.31,'II',ALLOC)
    arrow([(xs[1]+.7,2.40),(xs[1]+.7,2.46)],color=ALLOC,lw=.65,head=5)
    for j in range(3):
        arrow([(xs[j]+ws[j]+.025,3.07),(xs[j+1]-.035,3.07)],lw=1.1)
    text(1.96,2.89,r'$c^{\rm ref}$',7.4,ha='center')
    text(3.68,2.89,r'$c_i$',7.4,ha='center',color=ALLOC)
    text(5.40,2.89,r'$n,\pi$',7.4,ha='center')
    # Hot chemical equilibrium: schematic formation branches and solved energy.
    x=xs[0]
    ax.plot([x+.22,x+1.17],[3.29,3.29],c=LINE,lw=.6)
    ax.plot([x+.22,x+.22],[3.29,2.85],c=LINE,lw=.6)
    for yy,dy,c in [(3.16,-.28,ENERGY),(2.94,.24,PHYS),(3.07,.02,ION)]:
        ax.plot([x+.27,x+1.13],[yy,yy+dy],c=c,lw=1.15)
    ax.plot([x+.73,x+.73],[2.86,3.29],c=MUTED,lw=.6,ls=(0,(2,2)))
    text(x+.7,3.48,r'$F_h(s_h;\theta,x)=0$',8.0,ha='center')
    text(x+.7,3.64,r'Effective reference $T_f$',6.4,color=MUTED,ha='center')
    # Allocation keeps the total number of dopant atoms fixed.
    x=xs[1]
    for yy,left in [(2.91,.46),(3.17,.72)]:
        ax.add_patch(Rectangle((x+.22,yy),left,.11,fc=ALLOC,ec='none'))
        ax.add_patch(Rectangle((x+.22+left,yy),.96-left,.11,fc=ENERGY,ec='none'))
    text(x+.21,2.80,'reference',5.9,color=MUTED)
    text(x+.21,3.07,'learned',5.9,color=MUTED)
    text(x+.7,3.48,r'$B_Xc=C_X,\quad c_i\geq0$',8.0,ha='center')
    text(x+.7,3.64,'Freeze parent totals',6.4,color=ALLOC,ha='center')
    # Cold charge root and representative occupations.
    x=xs[2];u=np.linspace(0,1,70)
    ax.plot(x+.24+.91*u,3.27-.40/(1+np.exp(9*(u-.48))),c=PHYS,lw=1.2)
    ax.plot([x+.22,x+1.18],[3.30,3.30],c=LINE,lw=.6)
    for j,v in enumerate([.85,.20,.52]):
        ax.add_patch(Circle((x+.38+j*.30,3.38),.027,fc=PHYS,alpha=v,ec='none'))
    text(x+.7,3.50,r'$Q(E_F;c,T)=0$',8.0,ha='center')
    text(x+.7,3.65,'Re-equilibrate charge',6.4,color=MUTED,ha='center')
    # Hall observable: representative temperature-response icon (schematic).
    x=xs[3]
    ax.plot([x+.24,x+1.16],[3.29,3.29],c=LINE,lw=.6)
    ax.plot([x+.24,x+.24],[3.29,2.83],c=LINE,lw=.6)
    ax.plot(x+.28+.84*u,3.25-.36*(1-np.exp(-3*u)),color=ALLOC,lw=1.5)
    text(x+.7,3.50,r'$n_H=n/r_H$',9.,ha='center',color=ALLOC)
    text(x+.7,3.66,r'$\widehat y=\log_{10}[n_H/(1\,\mathrm{cm}^{-3})]$',6.6,ha='center')
    # Actual implicit derivative information feeds actions c; it does not
    # differentiate tree-fitting weights or consume outer-test Hall labels.
    card(2.13,4.02,3.35,.41,edge='#d2beb8',fill='#fcf6f3',r=.045)
    text(3.80,4.17,r'$D_\theta\widehat{y}=g_\theta-g_s\mathcal{F}_s^{-1}\mathcal{F}_\theta$',10.,ha='center',color=FEEDBACK)
    text(3.80,4.36,'implicit linear solve at the equilibrium root',6.0,ha='center',color=MUTED)
    arrow([(6.26,3.79),(6.26,4.23),(5.53,4.23)],color=FEEDBACK,dash=True)
    text(6.30,4.03,'sensitivity',6.3,color=FEEDBACK)
    card(.43,4.02,1.38,.41,edge='#d2beb8',fill='#fcf6f3',r=.045)
    text(1.12,4.15,'Task-specific learning',6.9,weight='bold',ha='center',color=FEEDBACK)
    text(1.12,4.33,'select label / update law',6.1,ha='center',color=FEEDBACK)
    arrow([(2.08,4.23),(1.86,4.23)],color=FEEDBACK,dash=True)
    arrow([(.43,4.23),(.23,4.23),(.23,3.07),(.37,3.07)],color=FEEDBACK,dash=True,lw=.9)
    text(.28,3.91,'recompute',5.7,color=FEEDBACK,rotation=90)

    # c. Same badges expand into the three actual learning procedures.
    panel(.10,4.84,'c','Three learning interfaces')
    card(.10,5.06,2.23,.92,edge=ENERGY,fill='#f7f3fa',r=.055)
    card(2.51,5.06,2.23,.92,edge=ALLOC,fill='#f0f9f6',r=.055)
    card(4.92,5.06,2.23,.92,edge=ION,fill='#fff7ef',r=.055)
    for x,title,col,mark in [(.10,'Property-directed energetics',ENERGY,'I'),
                              (2.51,'Conserved dopant allocation',ALLOC,'II'),
                              (4.92,'Donor ionization',ION,'III')]:
        badge(x+.15,5.23,mark,col)
        text(x+.29,5.23,title,6.7,weight='bold',color=col)
    text(.24,5.48,r'Energy labels $\longrightarrow$ surrogate $\longrightarrow\ \widehat{\xi}$',7.0)
    text(.24,5.70,r'$J\Sigma J^{\mathsf{T}}\ \longrightarrow$ next energy label',7.0,color=ENERGY)
    text(.24,5.88,'Energetic supervision + acquisition  ·  Fig. 3',6.0,color=MUTED)
    text(2.65,5.48,r'$(X,C_X,T_g;\theta)\ \longmapsto\ f_{\rm iso}$',8.4)
    text(2.65,5.70,r'Hall loss $\longrightarrow\ \theta$ through the solve',7.0,color=ALLOC)
    text(2.65,5.88,'Source-excluded carrier learning  ·  Fig. 4',6.0,color=MUTED)
    text(5.06,5.48,r'$C_X\ \longmapsto\ E_D(C_X;E_0,s)$',8.4)
    text(5.06,5.70,r'Hall curves $\longrightarrow$ donor statistics',7.0,color=ION)
    text(5.06,5.88,'Thermal prediction + adaptation  ·  Figs. 5, 6',6.0,color=MUTED)

    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for extension in ['pdf','svg','png']:
        fig.savefig(dest/f'fig1_architecture.{extension}',dpi=300,
                    facecolor='white',metadata={'Creator':'Matplotlib; original vector artwork'} if extension=='pdf' else None)
    # Check that each label remains on the canvas; precise spacing is reviewed
    # after rendering and again at final two-column publication width.
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    canvas=fig.bbox
    outside=[]
    for item in ax.texts:
        bb=item.get_window_extent(renderer)
        if bb.x0<canvas.x0-1 or bb.y0<canvas.y0-1 or bb.x1>canvas.x1+1 or bb.y1>canvas.y1+1:
            outside.append(item.get_text())
    report={'vector_outputs':['fig1_architecture.pdf','fig1_architecture.svg'],
            'host_atoms':len(atoms),'coordination_numbers':sorted(set(cn.values())),
            'numeric_results_changed':False,'miniature_curves':'schematic, not a new fit',
            'text_outside_canvas':outside,'learning_routes':{
                'I':'energy-label-supervised surrogate; property-Jacobian-guided acquisition',
                'II':'Hall-supervised inventory parameters, implicit differentiation',
                'III':'Hall-supervised donor parameters, implicit differentiation'},
            'explicit_scope':'three separate tested instances of the common computation',
            'structure_source':'source_data/beta_ga2o3_host.cif'}
    (OUT/'audit').mkdir(exist_ok=True)
    (OUT/'audit/figure1_structure.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.close(fig)
    assert not outside, outside
    print('Figure 1: vector PDF/SVG and 300 dpi PNG written.')

if __name__=='__main__':
    architecture()
