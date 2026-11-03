"""C2 numerical Fermi gas profile for coherent single-band Hall readout.

Only free-electron/free-hole integrals replace the legacy approximation.
The two legacy effective STH populations remain explicitly unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.special import zeta
from .material_profile import MaterialProfile,PUBLISHED_PROFILE
from .thermo import KB_EV
from .ionized_transport import density_mass_ratio_from_Nc

TABLE=Path(__file__).resolve().parents[2]/'data/physics_reference/fermi_half_v1.npz'
STATISTICS_VERSION='normalized_Fhalf_log_C2_v1_freebands_legacySTH'
LN10=float(np.log(10.))


def low_tail(eta):
    eta=np.asarray(eta,float);s=np.ones_like(eta);d=np.zeros_like(eta);dd=np.zeros_like(eta)
    for j in range(2,9):
        value=(-1.)**(j+1)*np.exp((j-1)*eta)/j**1.5
        s+=value;d+=(j-1)*value;dd+=(j-1)**2*value
    return eta+np.log(s),1+d/s,dd/s-(d/s)**2


def high_tail(eta):
    eta=np.asarray(eta,float);s=np.ones_like(eta);d=np.zeros_like(eta);dd=np.zeros_like(eta)
    for j in range(1,6):
        fall=float(np.prod([.5-k for k in range(2*j-1)]))
        coefficient=3*(1-2.**(1-2*j))*float(zeta(2*j))*fall
        value=coefficient*eta**(-2*j)
        s+=value;d+=(-2*j)*value/eta;dd+=2*j*(2*j+1)*value/eta**2
    return np.log(4/(3*np.sqrt(np.pi)))+1.5*np.log(eta)+np.log(s),1.5/eta+d/s,-1.5/eta**2+dd/s-(d/s)**2


class FermiHalf:
    def __init__(self,path=TABLE):
        self.path=Path(path);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        table=np.load(self.path,allow_pickle=False);self.eta=table['eta'];self.log_values=table['logF']
        self.spline=CubicSpline(self.eta,self.log_values,bc_type=((1,float(table['slope'][0])),(1,float(table['slope'][-1]))))
        self.inverse=PchipInterpolator(self.log_values,self.eta,extrapolate=False)

    def log(self,eta):
        x=np.asarray(eta,float)
        if not np.isfinite(x).all():raise ValueError('Finite reduced chemical potential required')
        out=[np.empty_like(x) for _ in range(3)]
        low=x<-20;high=x>60;middle=~(low|high)
        for a,v in zip(out,low_tail(x[low])):a[low]=v
        for a,v in zip(out,high_tail(x[high])):a[high]=v
        q=x[middle];v=[self.spline(q,nu=j) for j in range(3)]
        # A C2 blend to the high-temperature expansion prevents a derivative
        # seam at eta=60. Both representations are independently gated.
        z=np.clip((q-50)/10,0,1);w=6*z**5-15*z**4+10*z**3
        wp=(30*z**4-60*z**3+30*z**2)/10
        wpp=(120*z**3-180*z**2+60*z)/100
        if np.any(q>50):
            ids=q>50;tail=high_tail(q[ids]);base=[u[ids].copy() for u in v]
            v[0][ids]=(1-w[ids])*base[0]+w[ids]*tail[0]
            v[1][ids]=(1-w[ids])*base[1]+w[ids]*tail[1]+wp[ids]*(tail[0]-base[0])
            v[2][ids]=(1-w[ids])*base[2]+w[ids]*tail[2]+2*wp[ids]*(tail[1]-base[1])+wpp[ids]*(tail[0]-base[0])
        for a,b in zip(out,v):a[middle]=b
        if np.any(out[1]<=0):raise ValueError('Nonpositive free-carrier compressibility')
        return tuple(float(a) if a.ndim==0 else a for a in out)

    def eta_from_logF(self,value):
        target=float(value)
        if not np.isfinite(target):raise ValueError('Finite log Fermi density required')
        if target<self.log_values[0]:eta=target
        elif target>self.log_values[-1]:eta=float(np.exp((target-np.log(4/(3*np.sqrt(np.pi))))/1.5))
        else:eta=float(self.inverse(target))
        for _ in range(4):
            f,g,_=self.log(eta);step=(f-target)/g;eta-=step
            if abs(step)<1e-12*(1+abs(eta)):break
        return eta


@lru_cache(maxsize=1)
def reference():return FermiHalf()


@dataclass(frozen=True)
class CoherentMaterialProfile(MaterialProfile):
    statistics_kind:str=STATISTICS_VERSION
    fd_reference_sha256:str=''

    def __post_init__(self):
        super().__post_init__()
        if self.statistics_kind!=STATISTICS_VERSION:raise ValueError('Unknown coherent statistics version')
        if self.fd_reference_sha256!=reference().sha256:raise ValueError('Fermi mathematical reference hash mismatch')

    @property
    def electron_mass_ratio(self):return density_mass_ratio_from_Nc(self.Nc_ref,self.T_ref)

    def carriers(self,EF,T):
        values,ef,dt,dm=super().carriers(EF,T)
        Ev,Ec,dEv,dEc,mEv,mEc=self.bands(T);beta=1/(KB_EV*T)
        eta=np.array([(EF-Ec)*beta,(Ev-EF)*beta]);logs,g,_=reference().log(eta)
        pref=np.array([self.Nc_ref,self.Nv_ref])*(T/self.T_ref)**1.5
        values[:2]=pref*np.exp(logs);ef[:2]=values[:2]*g*beta*np.array([1.,-1.])
        eta_T=np.array([-dEc,dEv])*beta-eta/T
        dt[:2]=values[:2]*(1.5/T+g*eta_T)
        dm[:2]=values[:2,None]*g[:,None]*beta*np.vstack([-mEc,mEv])
        dm[0,0]+=values[0]*LN10;dm[1,1]+=values[1]*LN10
        return values,ef,dt,dm


def published_coherent_profile():
    values=PUBLISHED_PROFILE.to_dict();values['name']='published_DOS_gap_coherent_freebands_v1'
    values['provenance']+=';same scalar electron DOS/transport mass;freebandsC2FD;legacySTH unchanged'
    return CoherentMaterialProfile(**values,fd_reference_sha256=reference().sha256)
