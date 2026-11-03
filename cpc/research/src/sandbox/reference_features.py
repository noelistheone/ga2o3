"""Conditional defect descriptors from known parent reference energies only.

P_B A removes the span of reference shifts B. This is a balanced linear
combination of known energetic inputs, not a new geometric descriptor, an
elementary reaction, or an additional DFT calculation.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from .energy_learning import features as base_features, parent_reference, group_weights

KINDS=("base","atomic","closed","raw")

def reference_catalogue(db):
    """Read metadata and one supplied A_ref per mono-single-impurity parent.

    Nonreference dHo values are never indexed by this function and may be NaN.
    The parent/charge catalogue and reference-selection rule are supplied.
    """
    rows=[]
    for pid in sorted(np.unique(db.cs_ID)):
        ids=np.flatnonzero(db.cs_ID==pid)
        if not np.all(db.dm[ids]==db.dm[ids[0]]):
            raise ValueError("Parent atom composition changes with charge")
        dm=db.dm[ids[0]]
        if (np.count_nonzero(dm[2:])!=1 or np.abs(dm[2:]).sum()!=1 or dm[2:].max()!=1):
            continue
        ref=parent_reference(db,ids)
        value=float(db.dHo[ref])
        if not np.isfinite(value):
            raise ValueError("A supplied parent reference energy is missing")
        rows.append(dict(parent=int(pid),element=db.elements[2+int(np.argmax(dm[2:]))],
            reference_index=int(ref),q_reference=int(db.charge[ref]),
            dm_Ga=int(dm[0]),dm_O=int(dm[1]),reference_eV=value))
    return rows

def balanced_residual(A,B):
    """Orthogonal projection onto ker(B.T), retaining rank-deficient cases."""
    A=np.asarray(A,float);B=np.asarray(B,float)
    U,s,_=np.linalg.svd(B,full_matrices=False)
    cutoff=max(1.,float(s.max()))*1e-12
    use=s>cutoff;rank=int(use.sum())
    Q=U[:,use]
    residual=np.zeros_like(A) if rank==len(A) else A-Q@(Q.T@A)
    projector=np.eye(len(A))-Q@Q.T
    return residual,dict(rank=rank,nullity=len(A)-rank,cutoff=cutoff,
        annihilation_max=float(abs(projector@B).max()),
        idempotence_max=float(abs(projector@projector-projector).max()),
        balanced_energy_orthogonality_max=float(abs(B.T@residual).max()))

class ReferenceFeatureBank:
    """Per-element deterministic encoding of supplied reference inputs."""
    def __init__(self,catalogue):
        self.catalogue=[dict(r) for r in catalogue]
        self.by_parent={};self.diagnostics={}
        for e in sorted({r["element"] for r in catalogue}):
            rows=sorted((r for r in catalogue if r["element"]==e),key=lambda r:r["parent"])
            if len({r["parent"] for r in rows})!=len(rows):
                raise ValueError("Duplicate reference parent")
            A=np.array([r["reference_eV"] for r in rows],float)
            B=np.array([[1.,r["dm_Ga"],r["dm_O"]] for r in rows])
            q=np.array([r["q_reference"] for r in rows],float)
            atomic,da=balanced_residual(A,B)
            closed,dc=balanced_residual(A,np.column_stack([B,q]))
            self.diagnostics[e]=dict(parents=len(rows),atomic=da,closed=dc)
            for j,r in enumerate(rows):
                self.by_parent[int(r["parent"])]=dict(element=e,reference_index=r["reference_index"],
                    q_reference=r["q_reference"],atomic=float(atomic[j]),
                    closed=float(closed[j]),raw=float(A[j]))

    @classmethod
    def from_database(cls,db):return cls(reference_catalogue(db))

    def descriptor(self,records,kind):
        if kind not in KINDS:raise ValueError(kind)
        X=base_features(records)
        if kind=="base":return X
        values=[]
        for r in records:
            p=self.by_parent[int(r["parent"])]
            if p["element"]!=r["element"] or p["reference_index"]!=r["reference_index"]:
                raise ValueError("Known-reference catalogue mismatch")
            values.append(p[kind])
        return np.column_stack([X,np.asarray(values,float)])

class ReferenceEnergyLearner:
    """Same-budget feature ablation with training-only scaling.

    The bank is passed to fit/predict explicitly and is not stored in the
    trained model. Target-element reference covariates are inference inputs,
    never training labels or feature-standardization statistics.
    """
    def __init__(self,learner="rbf",kind="base",alpha=.01,length=1.):
        if learner not in ("ridge","rbf") or kind not in KINDS:
            raise ValueError("Unknown learner or reference descriptor")
        if alpha<=0 or length<=0:raise ValueError("Positive penalties required")
        self.learner=learner;self.kind=kind
        self.alpha=float(alpha);self.length=float(length)

    def fit(self,records,bank):
        X=bank.descriptor(records,self.kind)
        y=np.array([r["crossing_eV"] for r in records],float)
        w=group_weights(records);w/=w.sum()
        self.mean=float(np.average(y,weights=w))
        self.center=np.average(X,axis=0,weights=w)
        self.scale=np.sqrt(np.average((X-self.center)**2,axis=0,weights=w))
        self.scale[self.scale<1e-12]=1.
        self.training_elements=sorted({r["element"] for r in records})
        self.training_label_ids=[r["label_id"] for r in records]
        if self.learner=="ridge":
            self.model=Ridge(alpha=self.alpha)
        else:
            # Keep the original 18-dimensional bandwidth normalization for all
            # representations; adding a descriptor must not rescale old axes.
            self.gamma=1/(2*18*self.length**2)
            self.model=KernelRidge(alpha=self.alpha,kernel="rbf",gamma=self.gamma)
        self.model.fit((X-self.center)/self.scale,y-self.mean,sample_weight=w)
        return self

    def predict(self,records,bank):
        X=bank.descriptor(records,self.kind)
        return self.mean+self.model.predict((X-self.center)/self.scale)
