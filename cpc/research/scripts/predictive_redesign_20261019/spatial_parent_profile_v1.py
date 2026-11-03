"""Positive capped periodic parent profiles with exact mean-stock derivatives."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit,logsumexp


@dataclass
class ParentProfile:
    concentrations:np.ndarray
    reference_partial:np.ndarray
    amplitude_partial:np.ndarray
    positions:np.ndarray
    diagnostics:dict


def parent_profile(reference,caps,amplitude,*,grid_size=32,tolerance=8e-13,max_iterations=100):
    ref=np.asarray(reference,float);upper=np.asarray(caps,float);a=float(amplitude);N=int(grid_size)
    if ref.ndim!=1 or upper.shape!=ref.shape or len(ref)==0:
        raise ValueError('Matching one-dimensional parent inventories and caps required')
    if not all(np.isfinite(v).all() for v in [ref,upper]) or not np.isfinite(a):raise ValueError('Finite profile inputs required')
    if np.any(ref<=0) or np.any(ref>=upper):raise ValueError('Strictly interior mean stocks required;no floor is inserted')
    if N!=grid_size or N<8 or N%2 or abs(a)>40:raise ValueError('Even grid>=8 and numerical amplitude domain[-40,40] required')
    position=(np.arange(N)+.5)/N;cosine=np.cos(2*np.pi*position)
    g=np.log(2.)-np.logaddexp(0.,-a*cosine)
    ga=cosine*expit(-a*cosine)
    base=np.log(ref)-np.log(upper-ref)
    logupper=np.log(upper)[:,None];logref=np.log(ref)
    low=np.full(len(ref),min(g));high=np.full(len(ref),max(g));lam=np.zeros(len(ref))
    history=[]
    def evaluate(value):
        z=base[:,None]+g[None,:]-value[:,None]
        logc=logupper-np.logaddexp(0.,-z)
        logmean=logsumexp(logc,axis=1)-np.log(N)
        weights=np.exp(logc-logsumexp(logc,axis=1)[:,None])
        softness=np.sum(weights*expit(-z),axis=1)
        return z,logc,logmean-logref,softness
    for iteration in range(max_iterations+1):
        z,logc,residual,softness=evaluate(lam)
        error=float(max(abs(residual)));history.append(error)
        active=abs(residual)>tolerance
        if not active.any():break
        if iteration==max_iterations:raise ValueError('Parent multiplier iteration cap reached')
        if np.any(softness<=0) or not np.isfinite(softness).all():raise ValueError('Nonregular parent mean-stock multiplier')
        low=np.where(active&(residual>0),lam,low);high=np.where(active&(residual<0),lam,high)
        trial=lam+residual/softness
        outside=(trial<=low)|(trial>=high)|~np.isfinite(trial)
        trial=np.where(outside,(low+high)/2,trial)
        lam=np.where(active,trial,lam)
    fraction=expit(z);c=upper[:,None]*fraction
    tiny=z<-500
    c[tiny]=np.exp(logc[tiny])
    if a==0.:c=np.broadcast_to(ref[:,None],c.shape).copy()
    W=c*expit(-z);meanW=W.mean(axis=1)
    if np.any(meanW<=0):raise ValueError('Nonrepresentable mean susceptibility')
    dr=W/meanW[:,None]
    center=(W@ga)/W.sum(axis=1)
    da=W*(ga[None,:]-center[:,None])
    mass_error=float(max(abs(c.mean(axis=1)/ref-1)))
    ref_jac_error=float(max(abs(dr.mean(axis=1)-1)))
    amp_jac_error=float(max(abs(da.mean(axis=1))/(ref+np.mean(abs(W*ga),axis=1))))
    if np.any(c<=0) or np.any(c>upper[:,None]) or max(mass_error,ref_jac_error,amp_jac_error)>2e-10:
        raise ValueError('Local inventory/mean/derivative contract failed')
    return ParentProfile(c,dr,da,position,dict(grid_size=N,iterations=iteration,
        log_mean_residual=history[-1],residual_history=history,
        parent_mass_relative_error=mass_error,reference_partial_mean_error=ref_jac_error,
        amplitude_partial_mean_error=amp_jac_error,
        min_cap_fraction=float(min((c/upper[:,None]).ravel())),max_cap_fraction=float(max((c/upper[:,None]).ravel())),
        multipliers=lam.tolist(),multipliers_not_free_fitted_parameters=True,
        per_parent_caps_not_full_shared_site_exclusion=True))
