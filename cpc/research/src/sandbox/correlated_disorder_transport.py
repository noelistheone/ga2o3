"""Positive 3D screened-disorder Hall transport with a learned spectrum.

S_ZZ(q)=Nq2*(1+A*exp[-q^2*ell^2]) is a conditional continuum ansatz.
It is not a measured pair correlation, atomistic cluster identification, or a
certificate of the weak-scattering/Born, single-band or bulk approximations.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from .coherent_fermi import reference
from .coherent_transport import CoherentTransport,quadrature
from .ionized_transport import E_CHARGE,EPSILON0,K_B_J,HBAR,M_E,coulomb_log

LN10=float(np.log(10.))


def correlation_kernel(b,zeta,*,order=48):
    """C and dC/dln(zeta) for integral_0^b t/(1+t)^2 exp(-zeta*t) dt.

    y=ln(1+t) makes positive quadrature stable at large b. The exp(-50)
    tail is discarded only when zeta>0; its absolute contribution is bounded
    by exp(-50)*L(b). The upper-b derivative is evaluated exactly elsewhere.
    """
    b=np.asarray(b,float)
    if not np.isfinite(b).all() or np.any(b<=0) or not np.isfinite(zeta) or zeta<0:
        raise ValueError('Positive finite b and nonnegative finite zeta required')
    if zeta==0:return coulomb_log(b),np.zeros_like(b)
    cutoff=np.minimum(b,50./zeta)
    hi=np.log1p(cutoff)
    nodes,weights=quadrature(int(order))
    y=hi[...,None]*(nodes+1)/2
    t=np.expm1(y);zt=zeta*t
    measure=hi[...,None]/2*weights*(-np.expm1(-y))*np.exp(-zt)
    return measure.sum(axis=-1),-(measure*zt).sum(axis=-1)


def correlated_fermi_transport(log_n_cm3,log_Nq2_cm3,T_K,*,log10_A,
        log10_length_nm,Nc_ref_cm3,T_ref_K,mass_ratio,epsilon_r=10.2,
        extra_mobility_cm2Vs=205.,hall=True,order=256,kernel_order=48,tail_width=50.):
    """Log(nH,muH,sigma) and partials vs log10(n,Nq2,T,A,ell_nm).

    log10_A=None is the exact uncorrelated control. Its log-A derivative is
    interpreted as the zero-amplitude limit, not a finite parameter value.
    The declared scalar DOS mass must match the transport mass.
    """
    args=np.asarray([log_n_cm3,log_Nq2_cm3,T_K,log10_length_nm,
        Nc_ref_cm3,T_ref_K,mass_ratio,epsilon_r,extra_mobility_cm2Vs],float)
    if not np.isfinite(args).all() or min(args[2],*args[4:])<=0:
        raise ValueError('Finite state, length and positive material constants required')
    A=0. if log10_A is None else 10.**float(log10_A)
    length=10.**float(log10_length_nm)*1e-9
    if not np.isfinite([A,length]).all() or A<0 or length<=0:
        raise ValueError('Finite nonnegative amplitude and positive length required')
    mass=mass_ratio*M_E;eps=epsilon_r*EPSILON0;kT=K_B_J*T_K
    Nc=Nc_ref_cm3*(T_K/T_ref_K)**1.5
    expected_Nc=2*(mass*K_B_J*T_ref_K/(2*np.pi*HBAR**2))**1.5/1e6
    if abs(np.log(expected_Nc/Nc_ref_cm3))>2e-12:
        raise ValueError('Inconsistent scalar DOS and transport mass')
    eta=reference().eta_from_logF(log_n_cm3*LN10-np.log(Nc))
    if not -650<=eta<=1e4:raise ValueError('Outside declared numerical eta domain')
    _,g,h=reference().log(eta)
    n=10.**log_n_cm3;NI=10.**log_Nq2_cm3;ns=n*g
    if not np.isfinite([n,NI,ns]).all() or min(n,NI,ns)<=0:
        raise ValueError('Positive representable state densities required')
    eta_jac=np.array([1/g,0.,-1.5/g,0.,0.])
    ns_jac=np.array([1+h/g**2,0.,-1.5*h/g**2,0.,0.])
    qs2=E_CHARGE**2*(ns*1e6)/(eps*kT)
    zeta=qs2*length**2
    b_jac=np.array([0.,0.,2.,0.,0.])-ns_jac
    zeta_jac=ns_jac+np.array([0.,0.,-1.,0.,2.])
    pref=NI*1e6*E_CHARGE**4/(16*np.sqrt(2)*np.pi*eps**2*np.sqrt(mass)*kT**1.5)

    def rate(xx,*,derivatives=False):
        bb=8*mass*kT*np.asarray(xx)/(HBAR**2*qs2)
        L=coulomb_log(bb)
        C,Cz=correlation_kernel(bb,zeta,order=kernel_order)
        spectrum=L+A*C
        R=pref*np.asarray(xx)**(-1.5)*spectrum
        if not derivatives:return R
        db=(bb/(1+bb))**2
        Db=db*(1+A*np.exp(-zeta*bb))/spectrum
        Da=A*C/spectrum;Dz=A*Cz/spectrum
        J=np.array([0.,1.,-1.5,0.,0.])[None,:]+Db[:,None]*b_jac[None,:]
        J+=Dz[:,None]*zeta_jac[None,:];J[:,3]+=Da
        return R,J,bb,C,L

    nodes,weights=quadrature(int(order))
    lo=np.sqrt(max(0.,eta-tail_width));hi=np.sqrt(max(60.,eta+tail_width))
    t=(nodes+1)*(hi-lo)/2+lo;x=t*t
    f=expit(eta-x);measure=weights*(hi-lo)/2*2*t**4*f*expit(x-eta)/float(expit(eta))
    total=float(measure.sum())
    if not np.isfinite(total) or total<=0:raise ValueError('Positive finite Fermi measure required')
    probability=measure/total;q=1-2*f;qbar=float(probability@q)
    R,J,b,C,L=rate(x,derivatives=True)
    Rref=float(rate(np.array([max(1.5,eta)]))[0])
    R0=E_CHARGE/(mass*extra_mobility_cm2Vs*1e-4)
    denominator=R0+R;tau_reference=1/(R0+Rref)
    delta=(Rref-R)/denominator;mean_delta=float(probability@delta);mean=1+mean_delta
    centered=delta-mean_delta;variance=float(probability@(centered*centered))
    if mean<=0 or variance<0:raise ValueError('Invalid positive-time moments')
    rH=1+variance/mean**2
    log_tau_jac=-(R/denominator)[:,None]*J
    ratio=1+delta
    dmean=(probability[:,None]*ratio[:,None]*log_tau_jac).sum(axis=0)
    dmean+=(probability*centered*q).sum()*eta_jac
    dvariance=(probability*centered**2*(q-qbar)).sum()*eta_jac
    dvariance+=2*(probability[:,None]*centered[:,None]*ratio[:,None]*log_tau_jac).sum(axis=0)
    dmu=dmean/mean
    drH=(dvariance/mean**2-2*variance/mean**2*dmu)/rH
    log_mu=float(np.log10(E_CHARGE/mass*tau_reference*mean*1e4))
    log_rH=float(np.log10(rH)) if hall else 0.
    if not hall:drH=np.zeros(5)
    n_jac=np.array([1.,0.,0.,0.,0.])
    values=np.array([log_n_cm3-log_rH,log_mu+log_rH,np.log10(E_CHARGE)+log_n_cm3+log_mu])
    jac=np.vstack([n_jac-drH,dmu+drH,n_jac+dmu])
    if not np.isfinite(values).all() or not np.isfinite(jac).all():
        raise ValueError('Nonfinite correlated transport readout')
    return CoherentTransport(values,jac,log_mu,log_rH,dmu,drH,dict(
        eta=float(eta),free_n_cm3=n,n_screen_cm3=ns,Nq2_cm3=NI,temperature_K=T_K,
        mass_ratio=mass_ratio,epsilon_r=epsilon_r,
        computed_moment_hall_factor=float(rH),readout_hall_factor=float(10**log_rH),
        correlation_amplitude=A,correlation_length_nm=length*1e9,
        correlation_kernel_zeta=float(zeta),kernel_order=int(kernel_order),
        quadrature_order=int(order),x_domain=[lo*lo,hi*hi],
        kernel_C_over_L_min=float(np.min(C/L)),kernel_C_over_L_max=float(np.max(C/L)),
        positive_disorder_spectrum=True,unchanged_mean_charge_squared_budget=True,
        independent_scatterer_control=bool(A==0),
        extra_mobility_cm2Vs=float(extra_mobility_cm2Vs),
        Hall_conductivity_identity_error_dex=float(values[2]-(np.log10(E_CHARGE)+values[0]+values[1])),
        partial_input_names=['log10_free_n_cm3','log10_Nq2_cm3','log10_T_K',
            'log10_correlation_amplitude','log10_correlation_length_nm'],
        material_pair_correlation_measured=False,material_Born_validity_established=False,
        physical_scope='conditional continuum correlated screened disorder;isotropic parabolic band,elastic Born/RTA;not atomistic cluster identification'))
