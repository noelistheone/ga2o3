"""Normalized finite-state source inference, conditional on an explicit input design.

The measure is a declared weighted grid in the raw source coordinate, optionally
with an atom at the reference state. It is not a continuum posterior, a posterior
over atomic mechanisms, or a calibrated material uncertainty distribution.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp,ndtr
from scipy.optimize import brentq


def source_grid(size=129,radius=6.):
    if size<3 or size%2!=1 or not np.isfinite(radius) or radius<=0:raise ValueError('Odd finite grid and positive radius required')
    t=np.linspace(-1.,1.,size);u=radius*t**3
    edges=np.r_[-radius,(u[:-1]+u[1:])/2.,radius]
    return u,np.diff(edges)/(2*radius)


def weighted_index(values,weights,probability=.5):
    x=np.asarray(values,float);w=np.asarray(weights,float)
    if x.ndim!=1 or w.shape!=x.shape or not np.isfinite(np.r_[x,w]).all() or np.any(w<0) or w.sum()<=0 or not 0<probability<1:
        raise ValueError('Finite values and nonnegative probability weights required')
    order=np.argsort(x,kind='stable');cdf=np.cumsum(w[order]);return int(order[min(np.searchsorted(cdf,probability*w.sum(),side='left'),len(x)-1)])


@dataclass
class Posterior:
    weights:np.ndarray
    log_evidence:float
    prior:np.ndarray
    spike_posterior_probability:float
    sigma:float


class SourceEnsemble:
    def __init__(self,coordinate,base_weights,values,valid,*,zero_index=None,response_limit=36.):
        self.coordinate=np.asarray(coordinate,float);self.base=np.asarray(base_weights,float);self.values=np.asarray(values,float)
        mask=np.asarray(valid,bool)
        if self.values.ndim!=2 or self.values.shape[0]!=len(self.coordinate) or self.base.shape!=self.coordinate.shape or mask.shape!=self.coordinate.shape:
            raise ValueError('Consistent grid/state/node arrays required')
        if not np.isfinite(np.r_[self.coordinate,self.base]).all() or np.any(self.base<=0):raise ValueError('Finite grid and strictly positive base weights required')
        zeros=np.flatnonzero(self.coordinate==0.)
        self.zero=int(zeros[0] if zero_index is None and len(zeros)==1 else zero_index) if zero_index is not None or len(zeros)==1 else -1
        if self.zero<0 or self.zero>=len(self.coordinate) or self.coordinate[self.zero]!=0.:raise ValueError('One declared zero/reference index required')
        if not mask[self.zero] or not np.isfinite(self.values[self.zero]).all():raise ValueError('Reference state must be valid')
        self.Q=np.mean((self.values-self.values[self.zero])**2,axis=1)
        self.valid=mask&np.isfinite(self.values).all(axis=1)&np.isfinite(self.Q)&(self.Q<=response_limit)
        if not self.valid[self.zero]:raise ValueError('Reference outside declared response support')

    def prior(self,tau,rho):
        if not np.isfinite([tau,rho]).all() or tau<=0 or not 0<=rho<=1:raise ValueError('Positive response scale and spike probability in[0,1] required')
        logp=np.full(len(self.base),-np.inf);idx=np.flatnonzero(self.valid)
        logp[idx]=np.log(self.base[idx])-self.Q[idx]/(2*tau*tau);logp-=logsumexp(logp)
        p=(1-rho)*np.exp(logp);p[self.zero]+=rho
        return p

    def posterior(self,anchor_indices,observed,*,tau,sigma,rho=0.):
        ai=np.asarray(anchor_indices,int);y=np.asarray(observed,float)
        if ai.ndim!=1 or y.shape!=ai.shape or not np.isfinite(y).all() or np.any(ai<0) or np.any(ai>=self.values.shape[1]) or len(set(ai))!=len(ai):
            raise ValueError('Distinct in-range anchor indices and finite labels required')
        if not np.isfinite(sigma) or sigma<=0:raise ValueError('Positive finite working-likelihood scale required')
        prior=self.prior(tau,rho);ok=prior>0;loglike=np.full(len(prior),-np.inf)
        residual=self.values[ok][:,ai]-y
        loglike[ok]=-len(ai)*np.log(np.sqrt(2*np.pi)*sigma)-np.sum(residual**2,axis=1)/(2*sigma*sigma)
        lp=np.full(len(prior),-np.inf);lp[ok]=np.log(prior[ok])+loglike[ok];z=float(logsumexp(lp));w=np.exp(lp-z)
        if not np.isfinite(z) or not np.isclose(w.sum(),1.,rtol=0.,atol=1e-10):raise ValueError('Posterior normalization failed')
        spike=float(np.exp(np.log(rho)+loglike[self.zero]-z)) if rho else 0.
        return Posterior(w,z,prior,spike,float(sigma))

    def summarize(self,posterior,*,uncertainty=False):
        w=posterior.weights;ok=w>0;g=self.values[ok];mass=w[ok];indices=np.flatnonzero(ok)
        med=np.array([indices[weighted_index(g[:,j],mass)] for j in range(g.shape[1])],int)
        mean=mass@g;variance=mass@((g-mean)**2)
        out=dict(median_indices=med,latent_median=self.values[med,np.arange(g.shape[1])],mean=mean,variance=variance,
            posterior_spike=posterior.spike_posterior_probability,posterior_zero_state=float(w[self.zero]),
            log_evidence=posterior.log_evidence,effective_states=float(1./(w@w)))
        if uncertainty:
            latent=np.array([[g[weighted_index(g[:,j],mass,q),j] for j in range(g.shape[1])] for q in (.025,.975)])
            predictive=[]
            for q in (.025,.5,.975):
                predictive.append([brentq(lambda a:float(mass@ndtr((a-g[:,j])/posterior.sigma))-q,
                    float(g[:,j].min()-12*posterior.sigma),float(g[:,j].max()+12*posterior.sigma),xtol=1e-11) for j in range(g.shape[1])])
            out.update(latent_interval=latent,predictive_quantiles=np.array(predictive))
        return out

    def predictive_logpdf(self,posterior,node_indices,observed):
        idx=np.asarray(node_indices,int);y=np.asarray(observed,float)
        if y.shape!=idx.shape or not np.isfinite(y).all():raise ValueError('Consistent finite query labels required for scoring only')
        ok=posterior.weights>0;g=self.values[ok][:,idx];s=posterior.sigma
        terms=np.log(posterior.weights[ok,None])-np.log(np.sqrt(2*np.pi)*s)-(g-y)**2/(2*s*s)
        return logsumexp(terms,axis=0)
