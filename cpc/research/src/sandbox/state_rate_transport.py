"""Positive state-rate control for attributing a spectral closure's gain.

This is a phenomenological constant-time channel, not a phonon calculation.
Rextra=R205*(1+10**psi). No observed carrier/mobility is an input.
"""
import numpy as np
from scipy.special import expit
from .coherent_transport import CoherentTransport,fermi_transport,quadrature
from .ionized_transport import E_CHARGE,EPSILON0,K_B_J,HBAR,M_E,coulomb_log

LN10=float(np.log(10.))
BASE_LOG_MU=float(np.log10(205.))


def positive_state_rate_transport(log_n_cm3,log_Nq2_cm3,T_K,*,psi,
        Nc_ref_cm3,T_ref_K,mass_ratio,epsilon_r=10.2,hall=True,order=256,tail_width=50.):
    """Log outputs and partials vs log10(n,Nq2,T),psi.

    psi is a dimensionless base-10 logarithmic rate ratio. psi=None is exactly
    the no-added-rate control. Finite psi<=200 is a numerical domain only.
    Extremely negative psi is handled by stable logaddexp/expit, not clipping.
    """
    if psi is not None and (not np.isfinite(psi) or psi>200.):
        raise ValueError('Representable finite state-rate psi<=200 required;not a physical bound')
    logfactor=0. if psi is None else float(np.logaddexp(0.,LN10*psi)/LN10)
    fraction=0. if psi is None else float(expit(LN10*psi))
    mu0=10.**(BASE_LOG_MU-logfactor)
    base=fermi_transport(log_n_cm3,log_Nq2_cm3,T_K,Nc_ref_cm3=Nc_ref_cm3,
        T_ref_K=T_ref_K,mass_ratio=mass_ratio,epsilon_r=epsilon_r,
        extra_mobility_cm2Vs=mu0,hall=hall,order=order,tail_width=tail_width)
    eta=base.diagnostics['eta'];ns=base.diagnostics['n_screen_cm3']
    mass=mass_ratio*M_E;eps=epsilon_r*EPSILON0;kT=K_B_J*T_K
    nodes,weights=quadrature(int(order))
    lo=np.sqrt(max(0.,eta-tail_width));hi=np.sqrt(max(60.,eta+tail_width))
    t=(nodes+1)*(hi-lo)/2+lo;x=t*t
    measure=weights*(hi-lo)/2*2*t**4*expit(eta-x)*expit(x-eta)/float(expit(eta))
    probability=measure/measure.sum()
    qs2=E_CHARGE**2*(ns*1e6)/(eps*kT)
    b=8*mass*kT*x/(HBAR**2*qs2)
    pref=10.**log_Nq2_cm3*1e6*E_CHARGE**4/(16*np.sqrt(2)*np.pi*eps**2*np.sqrt(mass)*kT**1.5)
    ion_rate=pref*x**(-1.5)*coulomb_log(b)
    extra_rate=E_CHARGE/(mass*mu0*1e-4)
    r=extra_rate/(extra_rate+ion_rate)
    weighted=probability*r;weighted/=weighted.sum()
    a1=float(weighted@r);difference=float(weighted@((r-a1)**2))/a1
    dh=2*difference if hall else 0.
    du=np.array([-dh,a1+dh,a1])
    dpsi=-fraction*du
    dg=dict(base.diagnostics,extra_channel='state_conditioned_positive_constant_tau_not_computed_phonons',
        state_rate_psi=psi,added_rate_fraction=fraction,effective_log10_mu0=BASE_LOG_MU-logfactor,
        state_rate_partial=dpsi.tolist(),state_rate_is_not_identified_microscopic_scattering=True,
        numerical_psi_upper_domain_not_physical_bound=True)
    return CoherentTransport(base.values.copy(),np.c_[base.jacobian,dpsi],
        base.log_drift_mobility,base.log_hall_factor,
        np.r_[base.drift_jacobian,-fraction*a1],
        np.r_[base.hall_jacobian,-fraction*dh],dg)
