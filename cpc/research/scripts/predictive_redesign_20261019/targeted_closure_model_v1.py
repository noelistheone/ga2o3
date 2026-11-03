"""Fixed-capacity soft-hinge inventory and exact positive-rate scaling controls.

The low-dose derivative vanishes; this does not recover the physical reference.
All changes wrap the frozen point map and preserve its original implementation.
"""
from pathlib import Path
from dataclasses import replace
import sys
import numpy as np
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase34_learning_v1 as overlay

OUT=c.BASE/'targeted_closure_20261021_v1'
THRESHOLD=18.4
WIDTH=.25
ALPHA=.1


def clean(row):
    fields={'target','mu_cm2Vs','carrier_cm3','observed_log_nH','nH_cm3','log10_nH_cm3'}
    return row._replace(**{k:np.nan for k in fields if k in row._fields})


def hinge(logc):
    """Centered smooth hinge; high-dose derivative matches old feature 1/2."""
    z=(float(logc)-THRESHOLD)/WIDTH
    h=.5*WIDTH*(np.logaddexp(0.,z)-np.logaddexp(0.,(18.-THRESHOLD)/WIDTH))
    return float(h),float(.5*expit(z))


class HingeInventory:
    def __init__(self):
        self.base=overlay.model('OverlayShareProcess')
        self.parameter_names=('logit_Tf','log10_background','share_intercept',
            'share_hinge_coefficient','share_growth_coefficient','share_Sn_coefficient')

    def point(self,row,parameters,*,require_observation=False):
        w=np.asarray(parameters,float)
        if w.shape!=(6,) or not np.isfinite(w).all():raise ValueError('Six finite hinge parameters required')
        h,dh=hinge(np.log10(row.incorporated_cm3))
        eta=w.copy();eta[2]=w[2]+w[3]*h;eta[3]=0.
        p=self.base.point(clean(row),eta,require_observation=require_observation)
        chain=np.eye(6);chain[2,3]=h;chain[3]=0.
        ij=p.input_jacobian.copy();tij=p.transport_input_jacobian.copy()
        k=p.input_names.index('log10_dose_cm3')
        ij[:,k]+=p.jacobian[:,2]*w[3]*dh
        tij[:,k]+=p.transport_jacobian[:,2]*w[3]*dh
        return replace(p,jacobian=p.jacobian@chain,transport_jacobian=p.transport_jacobian@chain,
            input_jacobian=ij,transport_input_jacobian=tij,
            diagnostics=dict(p.diagnostics,hinge_threshold_log10_cm3=THRESHOLD,
                hinge_width_dex=WIDTH,hinge_feature=h,hinge_logdose_derivative=dh,
                hinge_coefficients=w[2:].tolist(),low_dose_recovers_reference_not_guaranteed=True,
                no_constraint_on_sign_of_hinge_coefficient=True))


MODELS={}
def model(method):
    if method not in MODELS:
        MODELS[method]=HingeInventory() if method=='DiluteHinge' else overlay.model('OverlayShareProcess')
    return MODELS[method]


def point_values(p):return np.r_[p.values[0],p.transport_values]


def forward(method,w,rows,*,penalty=True):
    values=[];jac=[];prior=[];pJ=[];valid=[];cache={}
    for row in rows.itertuples():
        key=c.learning.input_key(row,'process')
        if key not in cache:
            p=model(method).point(clean(row),w,require_observation=False)
            v=point_values(p);J=np.r_[p.jacobian[[0]],p.transport_jacobian]
            da=0.;dJ=np.zeros(6)
            if penalty:
                r=overlay.reference_model().point(clean(row),np.asarray(w)[:2],require_observation=False)
                da=v[0]-r.values[0];dJ=J[0].copy();dJ[:2]-=r.jacobian[0]
            cache[key]=(v,J,da,dJ,bool(p.diagnostics['observation_domain_valid']))
        v,J,da,dJ,ok=cache[key]
        values.append(v);jac.append(J);prior.append(da);pJ.append(dJ);valid.append(ok)
    n=len(rows)
    return dict(values=np.asarray(values),jacobian=np.asarray(jac),prior_nH=np.asarray(prior),
        prior_nH_jacobian=np.asarray(pJ),prior_mu=np.zeros(n),prior_mu_jacobian=np.zeros((n,6)),
        valid=np.asarray(valid,bool),unique_physical_inputs=len(cache))


def loss_arrays(method,w,rows):
    q=forward(method,w,rows)
    weight=c.learning.weights(rows);sw=np.sqrt(weight);reg=np.sqrt(ALPHA*weight)
    r=np.r_[sw*(q['values'][:,0]-rows.target.to_numpy(float)),reg*q['prior_nH']]
    J=np.r_[sw[:,None]*q['jacobian'][:,0],reg[:,None]*q['prior_nH_jacobian']]
    return r,J,q


def initializers(train,method):
    starts,metadata=overlay.initializer(train,'OverlayShareProcess',dict(gamma=0.,alpha=ALPHA))
    if method=='DiluteHinge':
        median=float(np.median(np.log10(overlay.subset(train).incorporated_cm3)))
        h,dh=hinge(median);out=[]
        for w in starts:
            v=np.asarray(w).copy();v[3]=np.clip(w[3]/max(2*dh,.1),-12.,12.)
            v[2]=np.clip(w[2]+w[3]*(median-18.)/2.-v[3]*h,-12.,12.);out.append(v)
        return out,dict(metadata,hinge_initialization='training-median value/slope projection;box clipped',training_median_logdose=median)
    return starts,metadata


def rate_features(rows):
    return np.c_[np.ones(len(rows)),(np.log10(rows.incorporated_cm3.to_numpy(float))-18.)/2.,
        (rows.growth_T_C.to_numpy(float)-700.)/300.,rows.element.eq('Sn').to_numpy(float)]


def scale_observables(base_values,log10_rate_scale):
    v=np.asarray(base_values,float).copy();d=np.asarray(log10_rate_scale,float)
    if not np.isfinite(d).all():raise ValueError('Finite log-rate scale required')
    v[...,1]-=d;v[...,2]-=d
    return v
