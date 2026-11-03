"""Vector frozen-parent charge at fixed temperature/material, no fitting.

Vector background extension of the existing fixed_fraction=0 cold branch. Derivatives are with
respect to local Fermi position and parent stocks; caller must include all
upstream stock/background derivatives. This is not a complete spatial model.
"""
from dataclasses import dataclass
import numpy as np
from sandbox import thermo
from sandbox.implicit_defects import _fd_log
from sandbox.coherent_fermi import reference,CoherentMaterialProfile


@dataclass
class LocalCharge:
    charge_cm3:np.ndarray
    charge_EF_partial:np.ndarray
    carriers_cm3:np.ndarray
    carrier_EF_partial:np.ndarray
    Nq2_cm3:np.ndarray
    Nq2_EF_partial:np.ndarray
    Nabs_cm3:np.ndarray
    Nabs_EF_partial:np.ndarray
    mean_q:np.ndarray
    mean_q2:np.ndarray
    mean_absq:np.ndarray
    probabilities:np.ndarray
    diagnostics:dict


class FrozenChargeKernel:
    def __init__(self,layer,parameters):
        if layer.fixed_fraction!=0.:raise ValueError('This version requires the audited fully ionized primary-donor branch')
        self.layer=layer;self.pars=np.asarray(parameters,float).copy()
        p,_,_,_=layer._unpack(self.pars);self.T=10.**p[layer.iobs];self.beta=1/(thermo.KB_EV*self.T)
        H,_,_,self.profile=layer._reference(self.T,p)
        if not isinstance(self.profile,CoherentMaterialProfile):raise ValueError('Same coherent scalar band required')
        entropy,_=layer._entropy(self.T)
        A=H-3*thermo.KB_EV*self.T*entropy*layer.db.sum_dm
        self.parents=layer.db.nD;self.count=np.bincount(layer.parent,minlength=self.parents)
        size=int(max(self.count));self.slot=np.zeros(len(layer.parent),int);used=np.zeros(self.parents,int)
        self.q=np.zeros((self.parents,size));self.energy=np.zeros_like(self.q)
        self.logg=np.full_like(self.q,-np.inf)
        for j,i in enumerate(layer.parent):
            k=used[i];used[i]+=1;self.slot[j]=k
            self.q[i,k]=layer.q[j];self.energy[i,k]=A[j];self.logg[i,k]=layer.logg[j]
        self.pairs=[]
        for plus,neutral in layer.pairs:
            i=int(layer.parent[plus])
            if self.count[i]!=2 or layer.parent[neutral]!=i:raise ValueError('Exactly two charge rows required for an extended isolated donor')
            self.pairs.append((i,int(self.slot[plus]),int(self.slot[neutral])))

    def carriers(self,fermi_eV):
        u=np.asarray(fermi_eV,float);profile=self.profile
        Ev,Ec,*_=profile.bands(self.T)
        eta=np.array([(u-Ec)*self.beta,(Ev-u)*self.beta,
            (Ev+profile.STH_energies[0]-u)*self.beta,(Ev+profile.STH_energies[1]-u)*self.beta])
        logs,g=_fd_log(eta);logs[:2],g[:2],_=reference().log(eta[:2])
        pref=np.array([profile.Nc_ref*(self.T/profile.T_ref)**1.5,
            profile.Nv_ref*(self.T/profile.T_ref)**1.5,profile.N_site,profile.N_site])
        value=pref[:,None]*np.exp(logs)
        derivative=value*g*self.beta*np.array([1.,-1.,-1.,-1.])[:,None]
        return value.T,derivative.T

    def evaluate(self,fermi_eV,stocks,*,background_cm3):
        u=np.asarray(fermi_eV,float);c=np.asarray(stocks,float);Nd=np.asarray(background_cm3,float)
        if Nd.ndim==0:Nd=np.full(len(u),float(Nd))
        if Nd.shape!=u.shape:raise ValueError("Matching scalar or local background[N] required")
        if u.ndim!=1 or not len(u) or c.shape!=(self.parents,len(u)):
            raise ValueError('Local positions[N] and frozen parent stocks[parents,N] required')
        if not np.isfinite(u).all() or not np.isfinite(c).all() or np.any(c<0) or not np.isfinite(Nd).all() or np.any(Nd<=0):
            raise ValueError('Finite nonnegative local stocks and positive background required')
        z=self.logg[:,:,None]-self.beta*(self.energy[:,:,None]+self.q[:,:,None]*u[None,None,:])
        z-=np.max(z,axis=1,keepdims=True);pi=np.exp(z);pi/=np.sum(pi,axis=1,keepdims=True)
        for i,plus,neutral in self.pairs:pi[i,plus,:]=1.;pi[i,neutral,:]=0.
        q=self.q[:,:,None]
        mq=np.sum(pi*q,axis=1);mq2=np.sum(pi*q*q,axis=1);ma=np.sum(pi*abs(q),axis=1)
        centered=q-mq[:,None,:]
        variance=np.sum(pi*centered**2,axis=1)
        covariance2=np.sum(pi*q*q*centered,axis=1)
        covariance_abs=np.sum(pi*abs(q)*centered,axis=1)
        car,caru=self.carriers(u)
        Q=Nd+np.sum(c*mq,axis=0)+np.sum(car[:,1:],axis=1)-car[:,0]
        Qu=-self.beta*np.sum(c*variance,axis=0)+np.sum(caru[:,1:],axis=1)-caru[:,0]
        NI=Nd+np.sum(c*mq2,axis=0);NIu=-self.beta*np.sum(c*covariance2,axis=0)
        NA=Nd+np.sum(c*ma,axis=0);NAu=-self.beta*np.sum(c*covariance_abs,axis=0)
        probability_error=float(max(abs(np.sum(pi,axis=1)-1).ravel()))
        if not all(np.isfinite(v).all() for v in [car,caru,Q,Qu,NI,NIu,NA,NAu]) or np.any(Qu>=0):
            raise ValueError('Nonregular vector charge state')
        return LocalCharge(Q,Qu,car,caru,NI,NIu,NA,NAu,mq,mq2,ma,pi,
            dict(max_parent_probability_error=probability_error,temperature_K=self.T,
                 vector_background_supported=True,background_mean_cm3=float(Nd.mean()),
                 minimum_negative_charge_slope=float(min(-Qu)),
                 fixed_temperature_and_material_partials_only=True,
                 primary_donor_branch='fixed_fraction=0;not a learned microscopic localization fraction',
                 coherent_profile_sha256=self.profile.fingerprint))
