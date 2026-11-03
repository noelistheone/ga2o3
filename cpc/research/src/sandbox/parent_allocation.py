"""Gauge-projected parent allocation bias inside the complete defect Sandbox.

The dimensionless process bias changes relative parent inventories, not the
charge-transition levels. It is an effective statistical closure, not new DFT.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .kroger_db import DefectDB
from .temperature_defects import TemperatureDefectLayer
from .material_profile import PUBLISHED_PROFILE
from .thermo import KB_EV


@dataclass(frozen=True)
class ParentContrast:
    charge_basis: np.ndarray
    raw_charge_basis: np.ndarray
    parent_ids: np.ndarray
    parent_composition: np.ndarray
    raw_parent: np.ndarray
    projected_parent: np.ndarray
    removed_potential_coordinates: np.ndarray
    isolated_parent_ids: tuple
    gauge_rank: int
    orthogonality_error: float


def isolated_parent_contrast(db:DefectDB,elements,donor_element=None):
    """Project against only freely solved incorporated-element potentials.

    Each active parent receives fixed unit weight; host Ga/O are not projected.
    Projection removes a gauge without changing the physical bias contrast.
    """
    elements=tuple(elements);donor_element=donor_element or elements[0]
    cols=[db.col(e) for e in elements]
    absent=[j for j in range(2,len(db.elements)) if db.elements[j] not in elements]
    keep=np.flatnonzero(np.all(db.dm[:,absent]==0,axis=1))
    parents=np.unique(db.cs_ID[keep]);first=np.array([np.flatnonzero(db.cs_ID==i)[0] for i in parents])
    A=db.dm[first][:,cols].astype(float);raw=np.zeros(len(parents))
    expected=np.zeros(len(db.elements),int);expected[0]=-1;expected[db.col(donor_element)]=1
    isolated=[]
    for j,parent in enumerate(parents):
        rows=np.flatnonzero(db.cs_ID==parent)
        if len(rows)==1 and db.charge[rows[0]]==1 and np.array_equal(db.dm[rows[0]],expected):
            raw[j]=1.;isolated.append(int(parent))
    if donor_element not in ("Si","Sn") or tuple(isolated)!=((169,170) if donor_element=="Si" else (151,152)):
        raise ValueError("Audited isolated donor parents changed")
    v=np.linalg.pinv(A,rcond=1e-12)@raw
    u=raw-A@v
    if np.linalg.norm(u)<1e-10:raise ValueError("Catalogue contrast is entirely a chemical-potential gauge")
    basis=np.zeros(db.nCS);original=np.zeros(db.nCS)
    for parent,p,q in zip(parents,u,raw):
        basis[db.cs_ID==parent]=p;original[db.cs_ID==parent]=q
    for a in (basis,original,parents,A,raw,u,v):a.setflags(write=False)
    return ParentContrast(basis,original,parents,A,raw,u,v,tuple(isolated),
        int(np.linalg.matrix_rank(A)),float(np.max(abs(A.T@u))))


class ParentAllocationLayer:
    """Full solver with a dimensionless common-parent process-weight input."""
    def __init__(self,db,elements=("Si",),*,profile=PUBLISHED_PROFILE,extension=True):
        self.elements=tuple(elements);self.contrast=isolated_parent_contrast(db,self.elements)
        self.core=TemperatureDefectLayer(db,self.elements,profile=profile,
            basis=self.contrast.charge_basis[:,None],donor_extension=extension,
            donor_element=self.elements[0],hall="rH1")
        self.extension=extension
        self.parameter_names=("log10_Tf_K","log10_background_cm3","log10_E0_eV","concentration_s","parent_bias")

    def point(self,*,Tf_K,background_cm3,doses,Tobs_K=300.,E0_eV=.0366,s=0.,bias=0.,pO2_atm=1e-5):
        energy_weight=KB_EV*Tf_K*bias
        p=self.core.parameters(T=Tf_K,Nd=background_cm3,doses=doses,Tobs=Tobs_K,
            binding_eV=E0_eV,screen_s=s,pO2=pO2_atm,weights=[energy_weight])
        prediction=self.core.predict(p);J=np.zeros((3,5))
        # w_bias=kB*Tf*b; its Tf derivative is required even when b is fixed.
        J[:,0]=prediction.jacobian[:,0]+prediction.jacobian[:,2]*energy_weight*np.log(10.)
        J[:,1]=prediction.jacobian[:,1]
        if self.extension:
            J[:,2]=prediction.jacobian[:,self.core.ie];J[:,3]=prediction.jacobian[:,self.core.ie+1]
        J[:,4]=prediction.jacobian[:,2]*KB_EV*Tf_K
        return prediction,J,p
