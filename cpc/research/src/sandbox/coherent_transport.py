"""Differentiable positive-rate Hall moments from one Fermi gas state.

Elastic screened independent ions, isotropic parabolic band and RTA are
assumed. The optional constant-tau channel is explicitly phenomenological;
it is not a computed polar optical phonon, grain or impurity-band model.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.special import expit
from .coherent_fermi import reference
from .ionized_transport import (E_CHARGE,EPSILON0,K_B_J,HBAR,M_E,coulomb_log)

LN10=float(np.log(10.))


@lru_cache(maxsize=8)
def quadrature(order):return np.polynomial.legendre.leggauss(order)


@dataclass
class CoherentTransport:
    values:np.ndarray
    jacobian:np.ndarray
    log_drift_mobility:float
    log_hall_factor:float
    drift_jacobian:np.ndarray
    hall_jacobian:np.ndarray
    diagnostics:dict


def fermi_transport(log_n_cm3,log_Nq2_cm3,T_K,*,Nc_ref_cm3,T_ref_K,mass_ratio,
                    epsilon_r=10.2,extra_mobility_cm2Vs=np.inf,hall=True,order=256,tail_width=50.):
    """Return log10(nH,muH,sigma), partials vs log10(n,Nq2,T).

    DOS normalization and transport mass must describe the same scalar band.
    Main-model callers supply the mass implied by their immutable Nc profile.
    """
    args=np.asarray([log_n_cm3,log_Nq2_cm3,T_K,Nc_ref_cm3,T_ref_K,mass_ratio,epsilon_r,tail_width],float)
    if not np.isfinite(args).all() or min(args[2:])<=0:raise ValueError('Finite physical state and positive constants required')
    if not (np.isinf(extra_mobility_cm2Vs) and extra_mobility_cm2Vs>0) and not (np.isfinite(extra_mobility_cm2Vs) and extra_mobility_cm2Vs>0):
        raise ValueError('Positive constant-tau mobility or positive infinity required')
    mass=mass_ratio*M_E;eps=epsilon_r*EPSILON0;kT=K_B_J*T_K
    Nc=Nc_ref_cm3*(T_K/T_ref_K)**1.5
    # Verify the declared single scalar band before calculating any readout.
    expected_Nc=2*(mass*K_B_J*T_ref_K/(2*np.pi*HBAR**2))**1.5/1e6
    if abs(np.log(expected_Nc/Nc_ref_cm3))>2e-12:raise ValueError('Inconsistent scalar DOS and transport mass')
    eta=reference().eta_from_logF(log_n_cm3*LN10-np.log(Nc));_,g,h=reference().log(eta)
    if not -650<=eta<=1e4:raise ValueError('Outside declared numerical Hall eta[-650,1e4] domain')
    n=10.**log_n_cm3;NI=10.**log_Nq2_cm3;ns=n*g
    if not np.isfinite([n,NI,ns]).all() or min(n,NI,ns)<=0:raise ValueError('Positive representable state densities required')
    eta_jac=np.array([1/g,0.,-1.5/g]);ns_jac=np.array([1+h/g**2,0.,-1.5*h/g**2])
    nodes,weights=quadrature(int(order));lo=np.sqrt(max(0.,eta-tail_width));hi=np.sqrt(max(60.,eta+tail_width))
    t=(nodes+1)*(hi-lo)/2+lo;x=t*t;weights=weights*(hi-lo)/2
    f=expit(eta-x);vacancy=expit(x-eta);amplitude=float(expit(eta))
    measure=weights*2*t**4*f*vacancy/amplitude
    total=float(measure.sum())
    if not np.isfinite(total) or total<=0:raise ValueError('Positive finite transport measure required')
    probability=measure/total;q=1-2*f;qbar=float(probability@q)
    qs2=E_CHARGE**2*(ns*1e6)/(eps*kT)
    pref=NI*1e6*E_CHARGE**4/(16*np.sqrt(2)*np.pi*eps**2*np.sqrt(mass)*kT**1.5)
    def rate(xx):
        bb=8*mass*kT*xx/(HBAR**2*qs2)
        return pref*xx**(-1.5)*coulomb_log(bb),bb
    R,b=rate(x);Rref=float(rate(max(1.5,eta))[0])
    R0=0. if np.isinf(extra_mobility_cm2Vs) else E_CHARGE/(mass*extra_mobility_cm2Vs*1e-4)
    denominator=R0+R;tau_reference=1/(R0+Rref)
    # Compute the deviation directly: do not subtract two nearly unit ratios.
    delta=(Rref-R)/denominator;mean_delta=float(probability@delta);mean=1+mean_delta
    centered=delta-mean_delta;variance=float(probability@(centered*centered))
    if mean<=0 or variance<0:raise ValueError('Invalid positive-time moments')
    numerical_rH=1+variance/mean**2
    D=(b/(1+b))**2/coulomb_log(b)
    log_rate_jac=np.array([0.,1.,-1.5])[None,:]+D[:,None]*(np.array([0.,0.,2.])-ns_jac)[None,:]
    log_tau_jac=-(R/denominator)[:,None]*log_rate_jac
    # tau_reference is treated as a fixed local scaling in this derivative;
    # common scaling terms cancel in rH and are retained in drift mobility.
    ratio=1+delta
    dmean=(probability[:,None]*ratio[:,None]*log_tau_jac).sum(axis=0)
    dmean+=(probability*centered*q).sum()*eta_jac
    dvariance=(probability*centered**2*(q-qbar)).sum()*eta_jac
    dvariance+=2*(probability[:,None]*centered[:,None]*ratio[:,None]*log_tau_jac).sum(axis=0)
    log_mu_jac=dmean/mean
    log_rH_jac=(dvariance/mean**2-2*variance/mean**2*log_mu_jac)/numerical_rH
    log_mu=float(np.log10(E_CHARGE/mass*tau_reference*mean*1e4))
    log_rH=float(np.log10(numerical_rH)) if hall else 0.
    if not hall:log_rH_jac=np.zeros(3)
    values=np.array([log_n_cm3-log_rH,log_mu+log_rH,np.log10(E_CHARGE)+log_n_cm3+log_mu])
    jac=np.vstack([np.array([1.,0.,0.])-log_rH_jac,log_mu_jac+log_rH_jac,np.array([1.,0.,0.])+log_mu_jac])
    if not np.isfinite(values).all() or not np.isfinite(jac).all():raise ValueError('Nonfinite coherent transport output')
    return CoherentTransport(values,jac,log_mu,log_rH,log_mu_jac,log_rH_jac,dict(
        eta=float(eta),free_n_cm3=n,n_screen_cm3=ns,Nq2_cm3=NI,temperature_K=T_K,mass_ratio=mass_ratio,
        epsilon_r=epsilon_r,computed_moment_hall_factor=float(numerical_rH),readout_hall_factor=float(10**log_rH),
        unit_Hall_control=not hall,extra_mobility_cm2Vs=None if np.isinf(extra_mobility_cm2Vs) else extra_mobility_cm2Vs,
        extra_channel='none' if R0==0 else 'fixed_effective_constant_tau_not_computed_phonons',
        quadrature_order=int(order),x_domain=[lo*lo,hi*hi],tail_width=tail_width,
        Hall_conductivity_identity_error_dex=float(values[2]-(np.log10(E_CHARGE)+values[0]+values[1])),
        physical_scope='one scalar band,freeelectron-onlystatic_screening,independentelasticpointcharges,Born/RTA;no materialvalidation'))
