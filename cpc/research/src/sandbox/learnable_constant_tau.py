"""A positive, differentiable phenomenological scattering channel.

The existing screened-ion/Fermi transport map is unchanged. A fourth partial
is added for u=log10(mu0/[cm^2/(V*s)]), with tau0=m*mu0*1e-4/e.
This is an effective constant-tau closure, not a microscopic phonon theory.
"""
import numpy as np
from scipy.special import expit
from .coherent_transport import CoherentTransport, fermi_transport, quadrature
from .ionized_transport import E_CHARGE, EPSILON0, K_B_J, HBAR, M_E, coulomb_log


def learnable_fermi_transport(log_n_cm3,log_Nq2_cm3,T_K,*,log10_extra_mobility,
        Nc_ref_cm3,T_ref_K,mass_ratio,epsilon_r=10.2,hall=True,order=256,tail_width=50.):
    """Return log10(nH,muH,sigma), partials vs log10(n,Nq2,T,mu0).

    The interval [-6,6] is a declared numerical parameter range, not a physical
    bound on mobility. The model's Born/RTA material limitations still apply.
    """
    u=float(log10_extra_mobility)
    if not np.isfinite(u) or not -6.<=u<=6.:
        raise ValueError('Finite log10 extra mobility in numerical interval[-6,6] required')
    mu0=10.**u
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
    r=extra_rate/(extra_rate+ion_rate)  # tau/tau0, strictly positive and <=1
    weighted=probability*r;weighted/=weighted.sum()
    a1=float(weighted@r)
    variance=float(weighted@((r-a1)**2))
    # a2-a1 = Var_{p*tau/<tau>}(tau/tau0)/a1. This avoids
    # cancellation of two almost equal moments in the constant-tau limit.
    difference=variance/a1
    dlog_rH=2*difference if hall else 0.
    du=np.array([-dlog_rH,a1+dlog_rH,a1])
    if not np.isfinite(du).all() or not 0<a1<=1+1e-14:
        raise ValueError('Finite positive scattering-moment derivative required')
    dg=dict(base.diagnostics,extra_channel='learnable_effective_constant_tau_not_computed_phonons',
        log10_extra_mobility=u,extra_rate_s=extra_rate,constant_tau_moment_a1=a1,
        constant_tau_moment_a2=a1+difference,constant_tau_tilted_variance=variance,
        log_mu0_partial=du.tolist(),
        partial_input_names=['log10_free_n_cm3','log10_Nq2_cm3','log10_T_K','log10_mu0_cm2Vs'],
        numerical_mu0_interval_is_not_a_physical_bound=True,
        derivative_scope='fixed free-carrier and squared-charge state for the fourth partial;full state chains are applied by the joint model')
    return CoherentTransport(base.values.copy(),np.c_[base.jacobian,du],
        base.log_drift_mobility,base.log_hall_factor,
        np.r_[base.drift_jacobian,a1],np.r_[base.hall_jacobian,dlog_rH],dg)
