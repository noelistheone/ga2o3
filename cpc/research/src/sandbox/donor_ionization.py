"""Reduced FD donor-ionization sandbox for temperature-prediction controls.

This baseline has prescribed chemical Si stock plus effective background charge;
it does not replace the full KROGER energetic/inventory model. The observation
assumption is unit Hall factor and uniform one-channel band transport.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from . import thermo
from .implicit_defects import _fd_log, LN10, ImplicitDomainError

@dataclass
class IonizationPrediction:
    log_n: np.ndarray
    jacobian: np.ndarray
    diagnostics: dict

def solve_ionization(C, T, *, background, compensation, energies_eV):
    """Return FD charge root eta=(EF-EC)/kBT and partial derivatives.

    Neutral donor degeneracy/ionized degeneracy=2. All background positive
    charge and compensating negative charge are frozen effective inputs.
    """
    C,T,E=np.broadcast_arrays(np.asarray(C,float),np.asarray(T,float),np.asarray(energies_eV,float))
    if np.any(C<=0) or np.any(T<10) or np.any(T>500) or np.any(E<0):
        raise ValueError("Invalid declared donor/temperature domain")
    B=float(background);rho=float(compensation)
    if B<0 or not 0<=rho<1:raise ValueError("Invalid background or compensation fraction")
    Nc=np.asarray(thermo.Nc(T),float);thermal=thermo.KB_EV*T
    lo=np.full(C.shape,-300.);hi=np.full(C.shape,100.)
    eta=np.clip(np.log((B+(1-rho)*C)/Nc),lo,hi)
    def quantities(eta):
        logfd,g=_fd_log(eta);n=Nc*np.exp(logfd);dn=n*g
        f=expit(-np.log(2)-eta-E/thermal)
        P=B+C*f;M=rho*C+n
        residual=np.log(P)-np.log(M)
        slope=-C*f*(1-f)/P-dn/M
        return residual,slope,n,dn,f,P,M
    for step in range(120):
        residual,slope,n,dn,f,P,M=quantities(eta)
        if np.max(abs(residual))<3e-12:break
        active=abs(residual)>=3e-12
        lo=np.where(active&(residual>0),eta,lo);hi=np.where(active&(residual<=0),eta,hi)
        trial=eta-residual/slope
        admissible=np.isfinite(trial)&(trial>lo)&(trial<hi)
        eta=np.where(active,np.where(admissible,trial,(lo+hi)/2),eta)
    else:raise ImplicitDomainError("Donor FD root failed its residual criterion")
    Qeta=-C*f*(1-f)-dn
    if np.any(Qeta>=0):raise ImplicitDomainError("Nondecreasing donor neutrality")
    return dict(eta=eta,n=n,dn_deta=dn,ionized_fraction=f,Q_eta=Qeta,
                residual=residual,steps=step+1,thermal=thermal,E=E,T=T,C=C)

class DonorIonizationSandbox:
    """Shared effective background/compensation/binding law, first-order IFT.

    Parameters log10B, logit(rho), log10E0[,positive screening coefficient].
    E(C)=E0 exp[-s(C/1e18)^(1/3)] is an effective positive relation, not a
    derived screening theory or a treatment of a Mott transition.
    """
    def __init__(self,screened=False):self.screened=bool(screened)

    def predict(self,C,T,parameters):
        w=np.asarray(parameters,float)
        if len(w)!=(4 if self.screened else 3):raise ValueError("Parameter count")
        C,T=np.broadcast_arrays(np.asarray(C,float),np.asarray(T,float))
        B=10**w[0];rho=float(expit(w[1]));E0=10**w[2]
        x=(C/1e18)**(1/3);s=w[3] if self.screened else 0.
        if s<0:raise ValueError("Positive screening coefficient required")
        E=E0*np.exp(-s*x)
        q=solve_ionization(C,T,background=B,compensation=rho,energies_eV=E)
        f=q["ionized_fraction"];factor=C*f*(1-f)/q["thermal"]
        dQ=np.stack([np.full(C.shape,B*LN10),-C*rho*(1-rho),-factor*E*LN10],axis=-1)
        if self.screened:dQ=np.concatenate([dQ,(factor*x*E)[...,None]],axis=-1)
        deta=-dQ/q["Q_eta"][...,None]
        jac=q["dn_deta"][...,None]/(q["n"][...,None]*LN10)*deta
        diag=dict(max_log_charge_residual=float(abs(q["residual"]).max()),
            charge_derivative_strictly_negative=bool(np.all(q["Q_eta"]<0)),root_steps=q["steps"],
            background_cm3=B,compensation_fraction=rho,E0_eV=E0,screening=s,
            binding_min_eV=float(E.min()),binding_max_eV=float(E.max()),
            minimum_n_cm3=float(q["n"].min()),maximum_C_over_Mott_reference=float(C.max()/2.4e18),
            above_roomT_detection_reference=bool(np.all(q["n"]>=1e13)),
            interpretation="effective reduced prediction baseline;no microscopic identification",
            Hall_assumption="unit Hall factor,uniform single band channel",
            full_energetic_KROGER=False)
        return IonizationPrediction(np.log10(q["n"]),jac,diag)
