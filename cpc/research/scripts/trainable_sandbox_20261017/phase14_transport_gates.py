"""Independent angular/unit/Fermi-moment checks; no material-label fitting."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
from sandbox.ionized_transport import (coulomb_log,ionized_rate_s,fermi_density_screen,normalized_moments,
    ionized_fermi_transport,mb_constant_log_mobility_cm2Vs,density_mass_ratio_from_Nc,
    E_CHARGE,EPSILON0,K_B_J,HBAR,M_E)
from sandbox import transport as legacy
from sandbox.material_profile import PUBLISHED_PROFILE
import phase2_energy as energy

OUT=energy.BASE/"phase14_joint_statistics/transport_audit"


def main():
    checks=[];angular=[];states=[];ratios=[]
    def add(name,good,**kw):checks.append(dict(name=name,passed=bool(good),**kw))
    for b in np.logspace(-10,10,21):
        def fn(theta):
            u=2*np.sin(theta/2)**2
            value=np.sin(theta)*u/(1+b*u/2)**2
            return value*b*b/4 if b>10 else value
        points=sorted(set([0.,np.pi,min(1/np.sqrt(b),np.pi),min(10/np.sqrt(b),np.pi)]))
        value=sum(quad(fn,lo,hi,epsabs=1e-11,epsrel=1e-11,limit=180)[0] for lo,hi in zip(points[:-1],points[1:]))
        from_angle=value if b>10 else value*b*b/4
        error=abs(from_angle-coulomb_log(b))/coulomb_log(b)
        angular.append(dict(b=b,angular=from_angle,closed=coulomb_log(b),relative_error=error))
        add("independent_angle_integral",error<2e-8,b=b,error=error)
    for T in (40.,300.,600.):
        for ns in (1e15,1e18,1e20):
            old=legacy.mu_brooks_herring(T,1e18,ns);new=mb_constant_log_mobility_cm2Vs(T,1e18,ns)
            ratio=new/old;error=abs(ratio/(16*np.pi**2)-1)
            ratios.append(dict(T=T,n_screen_cm3=ns,legacy=old,consistent_SI_MB=new,ratio=ratio))
            add("same_log_prefactor16pi2",error<2e-8,T=T,n_screen=ns,error=error)
    # Complete FGR prefactor from unintegrated angular expression.
    T=300.;ns=1e18;NI=3e18;mass=.28*M_E;eps=10.2*EPSILON0;energy_J=1.7*K_B_J*T
    qs2=E_CHARGE**2*ns*1e6/(eps*K_B_J*T);k=np.sqrt(2*mass*energy_J)/HBAR
    integral=quad(lambda theta:np.sin(theta)*(2*np.sin(theta/2)**2)/
        (2*k*k*(2*np.sin(theta/2)**2)+qs2)**2,0,np.pi,epsabs=1e-60,epsrel=1e-10)[0]
    rate=(NI*1e6)*mass*k/(2*np.pi*HBAR**3)*E_CHARGE**4/eps**2*integral
    closed=ionized_rate_s(energy_J,T,NI,ns);error=abs(rate/closed-1)
    add("full_FGR_SI_prefactor",error<2e-9,error=error,rate=rate,closed=closed)
    # Analytic constant-Coulomb-log Maxwell moments, including the Hall factor.
    J,_=normalized_moments(-35.,lambda x:x**1.5)
    expected_mu=gamma(4)/gamma(2.5);expected_rH=gamma(5.5)*gamma(2.5)/gamma(4)**2
    add("Maxwell_constantlog_moment_limit",abs(J[1]/J[0]/expected_mu-1)<1e-8,
        moment_ratio=J[1]/J[0],analytic=expected_mu)
    add("Maxwell_powerlaw_Hall_factor",abs((J[2]*J[0]/J[1]**2)/expected_rH-1)<1e-8,
        Hall_factor=J[2]*J[0]/J[1]**2,analytic=expected_rH)
    for eta in (-20.,-3.,0.,4.,40.,100.):
        a=ionized_fermi_transport(eta,300.,1e18);d=a.diagnostics
        states.append(a.to_dict())
        add("Hall_Cauchy_and_conductivity",a.hall_factor>=1-1e-9 and d["conductivity_relative_identity_error"]<2e-8,
            eta=eta,Hall_factor=a.hall_factor,error=d["conductivity_relative_identity_error"])
        h=1e-4;x=fermi_density_screen(eta+h,300.);y=fermi_density_screen(eta-h,300.)
        derivative=(x["n_cm3"]-y["n_cm3"])/(2*h);error=abs(derivative/a.n_screen_cm3-1)
        add("screening_is_free_density_derivative",error<2e-7,eta=eta,error=error)
        b=ionized_fermi_transport(eta,300.,2e18)
        error=max(abs(2*b.drift_mobility_cm2Vs/a.drift_mobility_cm2Vs-1),abs(b.hall_factor/a.hall_factor-1))
        add("fixed_carrier_inverse_Nq2",error<2e-9,eta=eta,error=error)
        c=ionized_fermi_transport(eta,300.,1e18,tail_width=70.)
        error=max(abs(c.drift_mobility_cm2Vs/a.drift_mobility_cm2Vs-1),abs(c.hall_factor/a.hall_factor-1))
        add("finite_tail_convergence",error<2e-8,eta=eta,error=error)
        if eta==-20.:
            add("nondegenerate_screening_limit",abs(a.n_screen_cm3/a.n_cm3-1)<1e-7,ratio=a.n_screen_cm3/a.n_cm3)
        if eta in (40.,100.):
            add("degenerate_screening_limit",abs((a.n_screen_cm3/a.n_cm3)/(1.5/eta)-1)<.002,eta=eta,ratio=a.n_screen_cm3/a.n_cm3)
            add("degenerate_Hall_limit",abs(a.hall_factor-1)<.01,eta=eta,Hall_factor=a.hall_factor)
    dos_mass=density_mass_ratio_from_Nc(PUBLISHED_PROFILE.Nc_ref,PUBLISHED_PROFILE.T_ref)
    density=fermi_density_screen(-20.,PUBLISHED_PROFILE.T_ref,mass_ratio=dos_mass)
    add("DOS_mass_reference_roundtrip",abs(density["Nc_cm3"]/PUBLISHED_PROFILE.Nc_ref-1)<1e-12,mass_ratio=dos_mass)
    protocol=dict(created_utc=energy.now(),code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (Path(__file__),
        ROOT/"src/sandbox/ionized_transport.py",ROOT/"src/sandbox/transport.py",ROOT/"docs/ML_SANDBOX_TRANSPORT_UNIT_DERIVATION_20261017.md")},
        scope="standalone isotropic Born/RTA numerical/unit validation;not fed into original nH task or new45fits",
        primary_source="https://www.rsc.org/suppdata/d0/nr/d0nr03432k/d0nr03432k1.pdf S1;differentmaterial,b24vslegacyb8;onlycoefficientconventionreference",
        data_labels_used=0,new_physical_experiments=0,goal_complete=False)
    pp=OUT/"protocol.json"
    if pp.exists():assert json.loads(pp.read_text())["code_hashes"]==protocol["code_hashes"]
    else:energy.write(pp,protocol)
    energy.write(OUT/"gates.json",dict(passed=all(c["passed"] for c in checks),checks=checks,angular=angular,
        coefficient_ratios=ratios,FD_states=states,published_profile_DOS_mass_ratio=dos_mass,
        legacy_mass_ratio=.28,legacy_mass_Nc_cm3=fermi_density_screen(-20,300.,mass_ratio=.28)["Nc_cm3"],
        material_transport_or_Hall_accuracy_validated=False,nH_pipeline_modified=False))
    print(json.dumps(dict(passed=all(c["passed"] for c in checks),checks=len(checks),failed=[c for c in checks if not c["passed"]]),indent=2))
    if not all(c["passed"] for c in checks):raise RuntimeError("Transport reference gates failed")


if __name__=="__main__":main()
