"""Reference-covariant native basis with a supplied dopant-parent anchor.

Unlike per-element projection, the host/charge basis is determined only by
native energies already supplied to every benchmark method. It is an energy
coordinate convention, not an inferred physical reservoir chemical potential.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from .energy_learning import features as base_features
from .reference_features import reference_catalogue,ReferenceEnergyLearner

def anchor_priority(row):
    composition=(row["dm_Ga"],row["dm_O"])
    priority=0 if composition==(-1,0) else 1 if composition==(0,0) else 2 if composition==(0,-1) else 3
    return priority,abs(row["dm_Ga"])+abs(row["dm_O"]),row["parent"]

class NativeReferenceBank:
    def __init__(self,db):
        self.catalogue=reference_catalogue(db)
        ids=np.flatnonzero(np.abs(db.dm[:,2:]).sum(axis=1)==0)
        D=np.column_stack([db.dm[ids,0],db.dm[ids,1],db.charge[ids]]).astype(float)
        A=np.asarray(db.dHo[ids],float)
        if not np.isfinite(A).all():raise ValueError("A supplied native energy is missing")
        counts=Counter(map(int,db.cs_ID[ids]))
        w=np.array([1/counts[int(p)] for p in db.cs_ID[ids]],float);w/=w.sum()
        design=D*np.sqrt(w[:,None]);target=A*np.sqrt(w)
        beta,residuals,rank,s=np.linalg.lstsq(design,target,rcond=1e-12)
        if rank!=3:raise ValueError("Native states do not span host/charge reference coordinates")
        self.native_coefficients=beta
        self.by_parent={};self.anchors={}
        for e in sorted({r["element"] for r in self.catalogue}):
            rows=[r for r in self.catalogue if r["element"]==e]
            anchor=min(rows,key=anchor_priority)
            b0=np.array([anchor["dm_Ga"],anchor["dm_O"],anchor["q_reference"]],float)
            self.anchors[e]=dict(anchor)
            for row in rows:
                b=np.array([row["dm_Ga"],row["dm_O"],row["q_reference"]],float)
                value=row["reference_eV"]-anchor["reference_eV"]-(b-b0)@beta
                self.by_parent[row["parent"]]=dict(element=e,reference_index=row["reference_index"],
                                                   native=float(value))
        self.diagnostics=dict(native_indices=ids.tolist(),native_states=len(ids),
            native_parents=len(counts),rank=int(rank),weighted_condition=float(s.max()/s.min()),
            weighted_fit_residual_eV=float(np.sqrt(w@((D@beta-A)**2))),
            coefficients_eV=beta.tolist(),
            native_basis_interpretation="reference coordinates only, not physical muGa/muO/EF")

    def descriptor(self,records,kind="native"):
        X=base_features(records)
        if kind=="base":return X
        if kind!="native":raise ValueError(kind)
        values=[]
        for r in records:
            p=self.by_parent[r["parent"]]
            if p["element"]!=r["element"] or p["reference_index"]!=r["reference_index"]:
                raise ValueError("Supplied reference parent mismatch")
            values.append(p["native"])
        return np.column_stack([X,np.asarray(values,float)])

class NativeReferenceLearner(ReferenceEnergyLearner):
    def __init__(self,learner="rbf",kind="native",alpha=.01,length=1.):
        if kind!="native":raise ValueError("Use the native-anchored descriptor")
        # Reuse the exact fitting/scaling code; descriptor validation resides
        # in this constructor and in NativeReferenceBank.descriptor.
        super().__init__(learner=learner,kind="closed",alpha=alpha,length=length)
        self.kind="native"
