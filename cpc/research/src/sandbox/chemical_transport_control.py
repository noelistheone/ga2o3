"""Simple compensation-aware transport controls without a defect catalogue.

These are fixed heuristic comparison laws, not a microscopic or atom-resolved
state model. All models can use the same predicted nH and known chemical dose.
"""
import numpy as np
from scipy.optimize import brentq
from .coherent_transport import fermi_transport


def chemical_hall_transport(log_nH_cm3,chemical_cm3,H_cm3,T_K,*,acceptor_charge,profile,background_cm3=1e14):
    if acceptor_charge not in (1,3):raise ValueError('Predetermined qA1 orqA3 comparison required')
    values=np.array([log_nH_cm3,chemical_cm3,H_cm3,T_K,background_cm3],float)
    if not np.isfinite(values).all() or min(chemical_cm3,T_K,background_cm3)<=0 or H_cm3<0:
        raise ValueError('Finite positive carrier/chemical/temperature inputs required')
    D=chemical_cm3+H_cm3+background_cm3;cache={}
    def point(logn):
        if logn not in cache:
            n=10.**logn;comp=max(D-n,0.);effective_D=max(D,n)
            # Neutrality n=ND-qA*NA -> q-squared density ND+qA*(ND-n).
            # If n>D, an explicit unobserved donor excess closes this toy law.
            Nq2=effective_D+acceptor_charge*comp
            p=fermi_transport(logn,np.log10(Nq2),T_K,Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,
                mass_ratio=profile.electron_mass_ratio,extra_mobility_cm2Vs=205.,hall=True)
            cache[logn]=(p,Nq2,max(n-D,0.))
        return cache[logn]
    lo=float(log_nH_cm3);hi=lo+1.
    a=point(lo)[0].values[0]-log_nH_cm3;b=point(hi)[0].values[0]-log_nH_cm3
    if a>1e-9 or b<0:raise ValueError('Simple Hall inversion could not bracket declared carrier state')
    root=lo if abs(a)<1e-12 else brentq(lambda x:point(x)[0].values[0]-log_nH_cm3,lo,hi,xtol=1e-10,rtol=1e-12)
    p,Nq2,excess=point(root);error=abs(p.values[0]-log_nH_cm3)
    if error>2e-7:raise ValueError('Simple Hall inversion residual failed')
    return dict(log_muH=float(p.values[1]),log_sigma=float(p.values[2]),log_free_n=float(root),Nq2_cm3=float(Nq2),
        chemical_donor_supply_cm3=float(D),implied_extra_donor_cm3=float(excess),acceptor_charge=acceptor_charge,
        inversion_error_dex=float(error),transport_evaluations=len(cache),
        scope='simple chemical+compensation closure,no defectenergies or learned inventory;extra donors are an explicit heuristic when n>D')
