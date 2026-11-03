"""Batched fixed-T local Fermi transport and laminated weak-field readout.

The local formula is the unchanged coherent_transport positive-time moment
model, evaluated at the Fermi positions already solved by Poisson. This file
does not learn a rate, fit labels or claim validity of Born/local transport.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit,logsumexp
from sandbox.coherent_fermi import reference
from sandbox.coherent_transport import quadrature
from sandbox.ionized_transport import E_CHARGE,EPSILON0,K_B_J,HBAR,M_E,coulomb_log

LN10=float(np.log(10.))


@dataclass
class LocalTransport:
    values:np.ndarray
    jacobian:np.ndarray
    diagnostics:dict


@dataclass
class EffectiveHall:
    values:np.ndarray
    jacobian:np.ndarray
    diagnostics:dict


def local_transport(eta,log_n_cm3,log_Nq2_cm3,T_K,*,Nc_ref_cm3,T_ref_K,
                    mass_ratio,epsilon_r=10.2,extra_mobility_cm2Vs=205.,
                    order=256,tail_width=50.):
    """Arrays[N,3] and partials[N,3,2] vs log10(n,Nq2), at fixed T.

    eta is redundant checked state information, not an independent parameter.
    The inverse Fermi call is avoided, but consistency with n is mandatory.
    """
    eta=np.asarray(eta,float);ln=np.asarray(log_n_cm3,float);lNI=np.asarray(log_Nq2_cm3,float)
    constants=np.array([T_K,Nc_ref_cm3,T_ref_K,mass_ratio,epsilon_r,tail_width],float)
    if eta.ndim!=1 or not len(eta) or ln.shape!=eta.shape or lNI.shape!=eta.shape:
        raise ValueError('Matching nonempty one-dimensional local states required')
    if not all(np.isfinite(v).all() for v in [eta,ln,lNI,constants]) or np.any(constants<=0):
        raise ValueError('Finite states and positive material constants required')
    if np.any(eta< -650.) or np.any(eta>1e4):raise ValueError('Outside declared local eta[-650,1e4] numerical domain')
    if not ((np.isinf(extra_mobility_cm2Vs) and extra_mobility_cm2Vs>0) or
            (np.isfinite(extra_mobility_cm2Vs) and extra_mobility_cm2Vs>0)):
        raise ValueError('Positive fixed extra mobility or positive infinity required')
    mass=mass_ratio*M_E;eps=epsilon_r*EPSILON0;kT=K_B_J*T_K
    expected=2*(mass*K_B_J*T_ref_K/(2*np.pi*HBAR**2))**1.5/1e6
    if abs(np.log(expected/Nc_ref_cm3))>2e-12:raise ValueError('Inconsistent DOS and transport mass')
    logF,g,h=reference().log(eta);Nc=Nc_ref_cm3*(T_K/T_ref_K)**1.5
    consistency=float(max(abs(ln*LN10-(np.log(Nc)+logF))))
    if consistency>2e-9:raise ValueError('Local eta and free-electron density are inconsistent')
    n=10.**ln;NI=10.**lNI;ns=n*g
    if not all(np.isfinite(v).all() for v in [n,NI,ns]) or np.any(np.minimum.reduce([n,NI,ns])<=0):
        raise ValueError('Representable positive local density required')
    etaJ=np.column_stack([1/g,np.zeros_like(g)])
    nsJ=np.column_stack([1+h/g**2,np.zeros_like(g)])
    nodes,weights=quadrature(int(order));lo=np.sqrt(np.maximum(0.,eta-tail_width));hi=np.sqrt(np.maximum(60.,eta+tail_width))
    t=(nodes[None,:]+1)*(hi-lo)[:,None]/2+lo[:,None];x=t*t
    weights=weights[None,:]*(hi-lo)[:,None]/2
    f=expit(eta[:,None]-x);vacancy=expit(x-eta[:,None])
    measure=weights*2*t**4*f*vacancy/expit(eta)[:,None]
    total=measure.sum(axis=1)
    if np.any(total<=0) or not np.isfinite(total).all():raise ValueError('Invalid transport measure')
    probability=measure/total[:,None];q=1-2*f;qbar=np.sum(probability*q,axis=1)
    qs2=E_CHARGE**2*(ns*1e6)/(eps*kT)
    pref=NI*1e6*E_CHARGE**4/(16*np.sqrt(2)*np.pi*eps**2*np.sqrt(mass)*kT**1.5)
    b=8*mass*kT*x/(HBAR**2*qs2[:,None]);R=pref[:,None]*x**(-1.5)*coulomb_log(b)
    xr=np.maximum(1.5,eta);br=8*mass*kT*xr/(HBAR**2*qs2)
    Rref=pref*xr**(-1.5)*coulomb_log(br)
    R0=0. if np.isinf(extra_mobility_cm2Vs) else E_CHARGE/(mass*extra_mobility_cm2Vs*1e-4)
    denominator=R0+R;tref=1/(R0+Rref)
    delta=(Rref[:,None]-R)/denominator;md=np.sum(probability*delta,axis=1);mean=1+md
    centered=delta-md[:,None];variance=np.sum(probability*centered**2,axis=1)
    if np.any(mean<=0) or np.any(variance<0):raise ValueError('Invalid local positive-time moments')
    rH=1+variance/mean**2
    D=(b/(1+b))**2/coulomb_log(b)
    rateJ=np.array([0.,1.])[None,None,:]-D[:,:,None]*nsJ[:,None,:]
    tauJ=-(R/denominator)[:,:,None]*rateJ
    ratio=1+delta
    dmean=np.sum(probability[:,:,None]*ratio[:,:,None]*tauJ,axis=1)
    dmean+=np.sum(probability*centered*q,axis=1)[:,None]*etaJ
    dvariance=np.sum(probability*centered**2*(q-qbar[:,None]),axis=1)[:,None]*etaJ
    dvariance+=2*np.sum(probability[:,:,None]*centered[:,:,None]*ratio[:,:,None]*tauJ,axis=1)
    muJ=dmean/mean[:,None]
    rhJ=(dvariance/mean[:,None]**2-2*variance[:,None]/mean[:,None]**2*muJ)/rH[:,None]
    logmu=np.log10(E_CHARGE/mass*tref*mean*1e4);logrh=np.log10(rH)
    values=np.column_stack([ln-logrh,logmu+logrh,np.log10(E_CHARGE)+ln+logmu])
    jac=np.stack([np.array([1.,0.])[None,:]-rhJ,muJ+rhJ,np.array([1.,0.])[None,:]+muJ],axis=1)
    tau=1/denominator;kell=2*kT*x*tau/HBAR
    first=probability*ratio;first/=first.sum(axis=1)[:,None]
    second=probability*ratio**2;second/=second.sum(axis=1)[:,None]
    mean_free_path=np.sqrt(2*kT*x/mass)*tau
    if not np.isfinite(values).all() or not np.isfinite(jac).all():raise ValueError('Nonfinite local transport')
    return LocalTransport(values,jac,dict(quadrature_order=int(order),fixed_T_partials_only=True,
        eta_consistency_log_error=consistency,eta=eta.tolist(),hall_factor=rH.tolist(),
        local_moment1_fraction_kell_below1=np.sum(first*(kell<1),axis=1).tolist(),
        local_moment2_fraction_kell_below1=np.sum(second*(kell<1),axis=1).tolist(),
        moment1_mean_free_path_nm=(np.sum(first*mean_free_path,axis=1)*1e9).tolist(),
        moment2_mean_free_path_nm=(np.sum(second*mean_free_path,axis=1)*1e9).tolist(),
        physical_scope='same scalar DOS/transport mass;freeelectron static screening;independent point charges;Born/RTA;fixed effective constant tau',
        material_Born_and_local_transport_validity_established=False))


def laminate_readout(local_values,local_jacobian,*,weights=None):
    """Homogenized small-contact vdP Hall logs and complete chain Jacobian.

    Each local row is log10(nH,muH,sigma); J is[N,3,P]. The normalized cell
    weights are fixed geometry in this version. Electron Hall magnitudes used.
    """
    v=np.asarray(local_values,float);J=np.asarray(local_jacobian,float)
    if v.ndim!=2 or v.shape[1]!=3 or J.ndim!=3 or J.shape[:2]!=v.shape:
        raise ValueError('Local logs[N,3] and Jacobian[N,3,P] required')
    if not np.isfinite(v).all() or not np.isfinite(J).all():raise ValueError('Finite local values/derivatives required')
    N=len(v);w=np.ones(N)/N if weights is None else np.asarray(weights,float)
    if w.shape!=(N,) or not np.isfinite(w).all() or np.any(w<=0) or abs(w.sum()-1)>1e-12:
        raise ValueError('Fixed positive normalized cell weights required')
    identity=float(max(abs(v[:,2]-np.log10(E_CHARGE)-v[:,0]-v[:,1])))
    if identity>2e-9:raise ValueError('Incompatible local Hall/conductivity state')
    lw=np.log(w)
    LA=float(logsumexp(lw+LN10*v[:,2]));LB=float(logsumexp(lw-LN10*v[:,2]));LC=float(logsumexp(lw+LN10*v[:,1]))
    wa=np.exp(lw+LN10*v[:,2]-LA);wb=np.exp(lw-LN10*v[:,2]-LB);wc=np.exp(lw+LN10*v[:,1]-LC)
    dA=wa@J[:,2,:];dB=-wb@J[:,2,:];dC=wc@J[:,1,:]
    values=np.array([(LA-LC)/LN10-np.log10(E_CHARGE),(LC-.5*(LA+LB))/LN10,.5*(LA-LB)/LN10])
    derivative=np.stack([dA-dC,dC-.5*(dA+dB),.5*(dA-dB)])
    return EffectiveHall(values,derivative,dict(local_Hall_identity_error_dex=identity,
        Hall_identity_error_dex=float(values[2]-np.log10(E_CHARGE)-values[0]-values[1]),
        log10_sigma_parallel=LA/LN10,log10_sigma_perpendicular=-LB/LN10,
        log10_conductivity_anisotropy=(LA+LB)/LN10,
        conductivity_weights=wa.tolist(),resistivity_weights=wb.tolist(),Hall_numerator_weights=wc.tolist(),
        geometry='one-dimensional periodic laminate homogenized in plane;weak field;small-contact homogeneous anisotropic vdP;not arbitrary3Dmicrostructure',
        normalized_fixed_geometry_weights=w.tolist()))
