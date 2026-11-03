"""Small convex, finite-design empirical source-correction controls.

Whitening makes the penalty the mean squared response change on the known
inputs. Rank-two controls deliberately have more source capacity than the
single physical inventory coordinate; they use the same anchor labels.
"""
import numpy as np
from scipy.optimize import brentq


def basis(log_dose,kind):
    x=np.asarray(log_dose,float)-18.
    if kind=='slope':return x[:,None]
    if kind=='affine':return np.c_[np.ones(len(x)),x]
    raise ValueError('Declared slope or affine basis required')


def fit_correction(base,log_dose,anchors,observed,lam,*,kind='affine',floor=13.,radius=6.):
    base=np.asarray(base,float);idx=np.asarray(anchors,int);obs=np.asarray(observed,float);V=basis(log_dose,kind);lam=float(lam)
    if len(base)!=len(V) or obs.shape!=idx.shape or not np.isfinite(np.r_[base,obs,V.ravel()]).all():raise ValueError('Finite consistent input/anchor arrays required')
    if np.any(base<floor):raise ValueError('Reference fails the finite-design carrier floor')
    if np.isinf(lam) or not len(idx):return dict(valid=True,coefficients=np.zeros(V.shape[1]),corrections=np.zeros(len(base)),objective=float(np.sum((base[idx]-obs)**2)),penalty=0.,rank=0,status='no_update')
    if lam<0:raise ValueError('Nonnegative regularization required')
    G=V.T@V/len(V);val,U=np.linalg.eigh(G);keep=val>max(float(val.max()),1.)*1e-12
    T=U[:,keep]/np.sqrt(val[keep])[None,:];W=V@T;rank=W.shape[1]
    if rank==0:return dict(valid=True,coefficients=np.zeros(V.shape[1]),corrections=np.zeros(len(base)),objective=float(np.sum((base[idx]-obs)**2)),penalty=0.,rank=0,status='unobservable_basis')
    A=W[idx];residual=obs-base[idx];H=A.T@A+lam*np.eye(rank);b=A.T@residual;lower=floor-base
    def objective(z):return float(np.sum((A@z-residual)**2)+lam*(z@z))
    def feasible(z):return bool(np.isfinite(z).all() and np.linalg.norm(z)<=radius+1e-8 and np.min(W@z-lower)>=-1e-8)
    candidates=[np.zeros(rank)]
    if rank==1:
        lo,hi=-radius,radius
        for w,c in zip(W[:,0],lower):
            if abs(w)<1e-14:
                if c>1e-12:raise ValueError('No feasible finite-design correction')
            elif w>0:lo=max(lo,float(c/w))
            else:hi=min(hi,float(c/w))
        if lo>hi+1e-10:raise ValueError('Empty feasible correction interval')
        z=float(b[0]/H[0,0]) if H[0,0]>1e-14 else 0.
        candidates.append(np.array([np.clip(z,lo,hi)]))
    else:
        unconstrained=np.linalg.lstsq(H,b,rcond=1e-12)[0];candidates.append(unconstrained)
        # Disk-only optimum; H is PSD, and b belongs to its range at lambda0.
        if np.linalg.norm(unconstrained)>radius:
            ev,R=np.linalg.eigh(H);projection=R.T@b
            def point(tau):return R@np.divide(projection,ev+tau,out=np.zeros_like(projection),where=ev+tau>1e-14)
            upper=1.
            while np.linalg.norm(point(upper))>radius:upper*=10.
            tau=brentq(lambda t:np.linalg.norm(point(t))-radius,0.,upper,xtol=1e-12,rtol=1e-12)
            candidates.append(point(tau))
        # Every other constrained optimum lies on at least one active line.
        # Minimize on each feasible line segment intersected with the disk.
        for wi,ci in zip(W,lower):
            norm=float(wi@wi)
            if norm<1e-24:continue
            center=ci*wi/norm;remaining=radius*radius-float(center@center)
            if remaining < -1e-10:continue
            direction=np.array([-wi[1],wi[0]])/np.sqrt(norm);extent=np.sqrt(max(remaining,0.));lo,hi=-extent,extent
            for wj,cj in zip(W,lower):
                a=float(wj@direction);c=float(cj-wj@center)
                if abs(a)<1e-12:
                    if c>1e-9:lo,hi=1.,0.;break
                elif a>0:lo=max(lo,c/a)
                else:hi=min(hi,c/a)
            if lo>hi+1e-9:continue
            lo=min(lo,hi);curvature=float(direction@H@direction);slope=float(direction@(H@center-b))
            t=-slope/curvature if curvature>1e-14 else 0. if abs(slope)<1e-10 else lo if slope>0 else hi
            candidates.append(center+np.clip(t,lo,hi)*direction)
    candidates=[z for z in candidates if feasible(z)]
    if not candidates:raise ValueError('No feasible source-correction candidate')
    z=min(candidates,key=lambda z:(round(objective(z),11),float(z@z),tuple(z)))
    original=z.copy();backoff=0
    # Floating-point active-edge intersections can be a few ulps outside.
    # Move coefficients toward the feasible reference, never clip predictions.
    while (np.any(base+W@z<floor) or np.linalg.norm(z)>radius) and backoff<40:
        z*=1-1e-9 if backoff==0 else .5;backoff+=1
    if np.any(base+W@z<floor) or np.linalg.norm(z)>radius:raise ValueError('Numerical interior backoff failed')
    return dict(valid=True,coefficients=T@z,corrections=W@z,objective=objective(z),penalty=float(z@z),rank=rank,
        source_dimensions=V.shape[1],metric_condition=float(val[keep].max()/val[keep].min()),
        numeric_backoff_steps=backoff,numeric_backoff_norm=float(np.linalg.norm(original-z)),
        minimum_node_log_nH=float(np.min(base+W@z)),solver='convex disk/active-edge enumeration',
        coefficient_gauge='minimum-response-norm on identifiable finite-design subspace')
