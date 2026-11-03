"""Explicit-SI screened-impurity Born/RTA reference, with Fermi moments.

An isotropic single parabolic band and independent static point charges are
assumed. This is not a fit to Ga2O3 data or a replacement for its full transport
closure. Density, screening and velocity moments use the same stated mass.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
from scipy.integrate import quad
from scipy.special import expit,gamma

E_CHARGE=1.602176634e-19
EPSILON0=8.8541878128e-12
K_B_J=1.380649e-23
HBAR=1.054571817e-34
M_E=9.1093837015e-31


def coulomb_log(b):
    b=np.asarray(b,float)
    if not np.isfinite(b).all() or np.any(b<=0):raise ValueError("Positive finite screening arguments required")
    small=b<.01;out=np.empty_like(b)
    x=b[small];series=np.zeros_like(x)
    for k in range(2,13):series+=((-1.)**k)*(k-1)/k*x**k
    out[small]=series
    out[~small]=np.log1p(b[~small])-b[~small]/(1+b[~small])
    if np.any(out<=0):raise ValueError("Coulomb logarithm underflow;outside numerical range")
    return float(out) if out.ndim==0 else out


def ionized_rate_s(energy_J,temperature_K,Nq2_cm3,n_screen_cm3,*,mass_ratio=.28,epsilon_r=10.2):
    args=np.asarray([temperature_K,Nq2_cm3,n_screen_cm3,mass_ratio,epsilon_r],float)
    ee=np.asarray(energy_J,float)
    if not np.isfinite(args).all() or np.any(args<=0) or not np.isfinite(ee).all() or np.any(ee<=0):
        raise ValueError("Explicit positive SI energy/temperature and densities required")
    mass=mass_ratio*M_E;eps=epsilon_r*EPSILON0;kT=K_B_J*temperature_K
    qs2=E_CHARGE**2*(n_screen_cm3*1e6)/(eps*kT)
    b=8*mass*ee/(HBAR**2*qs2)
    rate=(Nq2_cm3*1e6)*E_CHARGE**4/(16*np.sqrt(2)*np.pi*eps**2*np.sqrt(mass)*ee**1.5)*coulomb_log(b)
    return float(rate) if rate.ndim==0 else rate


def mb_constant_log_mobility_cm2Vs(T_K,Nq2_cm3,n_screen_cm3,*,mass_ratio=.28,epsilon_r=10.2,thermal_energy_factor=1.):
    if not np.isfinite([T_K,Nq2_cm3,n_screen_cm3,mass_ratio,epsilon_r,thermal_energy_factor]).all() or min(T_K,Nq2_cm3,n_screen_cm3,mass_ratio,epsilon_r,thermal_energy_factor)<=0:
        raise ValueError("Positive finite inputs required")
    mass=mass_ratio*M_E;eps=epsilon_r*EPSILON0;kT=K_B_J*T_K
    b=8*mass*eps*kT*kT*thermal_energy_factor/(HBAR**2*E_CHARGE**2*n_screen_cm3*1e6)
    mu=128*np.sqrt(2*np.pi)*eps**2*kT**1.5/(Nq2_cm3*1e6*E_CHARGE**3*np.sqrt(mass)*coulomb_log(b))
    return float(mu*1e4)


def _integral(function,eta,cutoff):
    points=sorted(set([0.,cutoff]+[x for x in (1.,10.,eta-20.,eta,eta+20.) if 0<x<cutoff]))
    value=0.;bound=0.
    for lo,hi in zip(points[:-1],points[1:]):
        y,error=quad(function,lo,hi,epsabs=2e-10,epsrel=2e-10,limit=180)
        value+=y;bound+=error
    return value,bound


def fermi_density_screen(eta,T_K,*,mass_ratio=.28,tail_width=50.):
    if not np.isfinite([eta,T_K,mass_ratio,tail_width]).all() or not(-80<=eta<=200) or min(T_K,mass_ratio,tail_width)<=0:
        raise ValueError("Declared eta[-80,200],positive T,mass and tail width required")
    amplitude=float(expit(eta));cutoff=max(60.,eta+tail_width)
    def f(x):return float(expit(eta-x))
    n,error_n=_integral(lambda x:np.sqrt(x)*f(x)/amplitude,eta,cutoff)
    s,error_s=_integral(lambda x:np.sqrt(x)*f(x)*(1-f(x))/amplitude,eta,cutoff)
    Nc_m3=2*(mass_ratio*M_E*K_B_J*T_K/(2*np.pi*HBAR**2))**1.5
    pref=Nc_m3*2/np.sqrt(np.pi)*amplitude/1e6
    return dict(n_cm3=pref*n,n_screen_cm3=pref*s,Nc_cm3=Nc_m3/1e6,
        density_integral=n,screen_integral=s,cutoff=cutoff,amplitude=amplitude,
        estimated_scaled_integration_error=max(error_n,error_s))


def normalized_moments(eta,tau_ratio,*,tail_width=50.):
    if not(-80<=eta<=200):raise ValueError("Declared eta interval required")
    amplitude=float(expit(eta));cutoff=max(60.,eta+tail_width);moments=[];errors=[]
    for power in (0,1,2):
        def integrand(x):
            f=float(expit(eta-x));r=1. if power==0 else float(tau_ratio(x))**power
            return x**1.5*f*(1-f)/amplitude*r
        value,error=_integral(integrand,eta,cutoff);moments.append(value);errors.append(error)
    if not np.isfinite(moments).all() or min(moments)<=0:raise ValueError("Finite positive Fermi transport moments required")
    return np.array(moments),dict(cutoff=cutoff,amplitude=amplitude,estimated_scaled_integration_error=max(errors))


def density_mass_ratio_from_Nc(Nc_ref_cm3,Tref_K):
    if not np.isfinite([Nc_ref_cm3,Tref_K]).all() or min(Nc_ref_cm3,Tref_K)<=0:raise ValueError("Positive DOS reference required")
    return float(2*np.pi*HBAR**2/(K_B_J*Tref_K)*(Nc_ref_cm3*1e6/2)**(2/3)/M_E)


@dataclass
class IonizedTransport:
    n_cm3:float
    n_screen_cm3:float
    screening_wavevector_squared_m2:float
    drift_mobility_cm2Vs:float
    hall_factor:float
    Hall_density_cm3:float
    Hall_mobility_cm2Vs:float
    conductivity_Scm:float
    eta:float
    temperature_K:float
    Nq2_cm3:float
    mass_ratio:float
    epsilon_r:float
    diagnostics:dict

    def to_dict(self):return asdict(self)


def ionized_fermi_transport(eta,T_K,Nq2_cm3,*,mass_ratio=.28,epsilon_r=10.2,tail_width=50.):
    if not np.isfinite(Nq2_cm3) or Nq2_cm3<=0:raise ValueError("Positive squared-charge density required")
    density=fermi_density_screen(eta,T_K,mass_ratio=mass_ratio,tail_width=tail_width)
    ns=density["n_screen_cm3"];kT=K_B_J*T_K;mass=mass_ratio*M_E;eps=epsilon_r*EPSILON0
    def tau(x):return 1/ionized_rate_s(kT*x,T_K,Nq2_cm3,ns,mass_ratio=mass_ratio,epsilon_r=epsilon_r)
    reference=tau(max(1.5,eta));J,info=normalized_moments(eta,lambda x:tau(x)/reference,tail_width=tail_width)
    mu_d=E_CHARGE/mass*reference*J[1]/J[0]*1e4;rH=float(J[2]*J[0]/J[1]**2)
    n=density["n_cm3"];nH=n/rH;muH=rH*mu_d;sigma=E_CHARGE*n*mu_d
    # Independent density-of-states conductivity integral, SI to S/cm.
    D0=1/(2*np.pi**2)*(2*mass/HBAR**2)**1.5
    sigma_integral=(2*E_CHARGE**2/(3*mass))*D0*kT**1.5*reference*J[1]*info["amplitude"]/100
    return IonizedTransport(n,ns,E_CHARGE**2*ns*1e6/(eps*kT),float(mu_d),rH,float(nH),float(muH),float(sigma),
        float(eta),float(T_K),float(Nq2_cm3),float(mass_ratio),float(epsilon_r),dict(
            density=density,moments=J.tolist(),moment_integration=info,
            independent_conductivity_integral_Scm=float(sigma_integral),
            conductivity_relative_identity_error=float(abs(sigma_integral-sigma)/sigma),
            statistical_scope="exact-FD numerical quadrature,one isotropic parabolic band,one mass,finite-tail control",
            physical_scope="elastic independent screened pointcharges,Born,RTA;no phonons/grains/impurityband or material validation"))
