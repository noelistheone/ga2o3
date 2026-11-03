"""Independent positive-moment reconstruction and conditional k*l diagnostic.

This reconstructs the declared scalar-band independent-ion/constant-tau model.
It is a numerical/model-scope check, not a material Born or quasiparticle
certificate. Integration is split at k*l=1 instead of applying a discontinuous
indicator to an unsplit quadrature grid.
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from .coherent_fermi import reference
from .coherent_transport import quadrature
from .ionized_transport import E_CHARGE,EPSILON0,K_B_J,HBAR,M_E,coulomb_log


def reconstruct_transport(n_cm3,Nq2_cm3,T_K,mu0_cm2Vs,*,Nc_ref_cm3,T_ref_K,mass_ratio,
        epsilon_r=10.2,order=512,tail_width=50.):
    numbers=np.asarray([n_cm3,Nq2_cm3,T_K,mu0_cm2Vs,Nc_ref_cm3,T_ref_K,mass_ratio,epsilon_r],float)
    if not np.isfinite(numbers).all() or np.any(numbers<=0):raise ValueError('Finite positive declared transport state required')
    mass=mass_ratio*M_E;kT=K_B_J*T_K;epsilon=epsilon_r*EPSILON0
    expected_nc=2*(mass*K_B_J*T_ref_K/(2*np.pi*HBAR**2))**1.5/1e6
    if abs(np.log(expected_nc/Nc_ref_cm3))>2e-12:raise ValueError('Consistent scalar DOS and transport mass required')
    nc=Nc_ref_cm3*(T_K/T_ref_K)**1.5
    eta=reference().eta_from_logF(np.log(n_cm3/nc));_,g,_=reference().log(eta)
    ns=n_cm3*g;qs2=E_CHARGE**2*ns*1e6/(epsilon*kT)
    pref=Nq2_cm3*1e6*E_CHARGE**4/(16*np.sqrt(2)*np.pi*epsilon**2*np.sqrt(mass)*kT**1.5)
    extra=E_CHARGE/(mass*mu0_cm2Vs*1e-4)
    def times(x):
        x=np.asarray(x,float)
        b=8*mass*kT*x/(HBAR**2*qs2)
        return 1/(pref*x**(-1.5)*coulomb_log(b)+extra)
    def kl(x):return 2*x*kT*times(x)/HBAR
    nodes,weights=quadrature(int(order))
    lo=np.sqrt(max(0.,eta-tail_width));hi=np.sqrt(max(60.,eta+tail_width))
    def moments(a,b):
        if b<=a:return np.zeros(4)
        t=(nodes+1)*(b-a)/2+a;x=t*t
        measure=weights*(b-a)/2*2*t**4*expit(eta-x)*expit(x-eta)/expit(eta)
        tau=times(x)
        return np.array([measure.sum(),measure@tau,measure@(tau*tau),measure@(tau*kl(x))])
    total=moments(lo,hi)
    if not np.isfinite(total).all() or np.any(total[:3]<=0):raise ValueError('Positive transport moments required')
    # k*l increases strictly with E for this declared rate law (proof in docs).
    xlo=max(lo*lo,1e-20);xhi=hi*hi
    if kl(xlo)>=1.:
        below=np.zeros(4);cut=None;cut_scope='threshold_below_truncated_measure'
    elif kl(xhi)<=1.:
        below=total.copy();cut=None;cut_scope='threshold_above_truncated_measure'
    else:
        cut=float(brentq(lambda x:float(kl(x)-1.),xlo,xhi,xtol=1e-13,rtol=1e-13))
        below=moments(lo,np.sqrt(cut));cut_scope='resolved_unique_threshold_in_truncated_measure'
    mean=total[1]/total[0];rh=total[2]*total[0]/total[1]**2
    mud=E_CHARGE/mass*mean*1e4
    values=np.array([np.log10(n_cm3/rh),np.log10(mud*rh),
        np.log10(E_CHARGE*n_cm3*mud)])
    return dict(values=values.tolist(),eta=float(eta),n_screen_cm3=float(ns),
        hall_factor=float(rh),drift_mobility_cm2Vs=float(mud),
        mean_tau_s=float(mean),fraction_sigma_kl_lt1=float(below[1]/total[1]),
        fraction_Hall_kl_lt1=float(below[2]/total[2]),sigma_weighted_kl=float(total[3]/total[1]),
        threshold_x=cut,threshold_scope=cut_scope,quadrature_order=int(order),
        truncated_x_domain=[float(lo*lo),float(hi*hi)],
        diagnostic_is_a_material_Born_certificate=False)
