"""Shared charge-energy kernels trained in a physical property metric.

Q = D + lambda J.T W J is a local training metric, not a global error bound.
All feature scaling, kernel centers and metric blocks belong to training
elements. A supplied held-element energy/reference is never a feature.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics.pairwise import rbf_kernel
from .energy_learning import features, group_weights, FEATURE_NAMES


def element_database(db,element):
    """Exact absence of all other impurities; return original state indices.

    Parent labels are compacted only internally for the equilibrium solver.
    Keeping absent elements at a large negative chemical potential is avoided
    here, so their unknown energies cannot enter a training-element metric.
    """
    col=db.col(element)
    other=[j for j in range(2,len(db.elements)) if j!=col]
    keep=np.flatnonzero(np.all(db.dm[:,other]==0,axis=1))
    parents=np.unique(db.cs_ID[keep])
    parent_map={int(p):j+1 for j,p in enumerate(parents)}
    return replace(db,dHo=db.dHo[keep].copy(),charge=db.charge[keep].copy(),
        dm=db.dm[keep].copy(),cs_ID=np.array([parent_map[int(p)] for p in db.cs_ID[keep]]),
        prefactor=db.prefactor[keep].copy(),sum_dm=db.sum_dm[keep].copy(),
        nCS=len(keep),nD=len(parents)),keep


class PropertyMetricKernel:
    """RBF regression with an RKHS penalty and a positive physical loss."""
    def __init__(self,length=1.,alpha=1e-3,metric="energy",strength=0.,
                 property_scale=.3):
        if length<=0 or alpha<=0 or strength<0 or property_scale<=0:
            raise ValueError("Invalid positive regularization/scale")
        if metric not in ("energy","diagonal","full"):
            raise ValueError(metric)
        self.length=float(length);self.alpha=float(alpha)
        self.metric=metric;self.strength=float(strength)
        self.property_scale=float(property_scale)

    def fit(self,records,metric_blocks=None):
        X=features(records);y=np.array([r["crossing_eV"] for r in records])
        w=group_weights(records);w/=w.sum()
        self.center=np.average(X,axis=0,weights=w)
        self.scale=np.sqrt(np.average((X-self.center)**2,axis=0,weights=w))
        self.scale[self.scale<1e-12]=1.
        self.mean=float(np.average(y,weights=w))
        self.training_elements=sorted({r["element"] for r in records})
        self.training_label_ids=[r["label_id"] for r in records]
        self.X=(X-self.center)/self.scale
        self.gamma=1/(2*len(FEATURE_NAMES)*self.length**2)
        K=rbf_kernel(self.X,self.X,gamma=self.gamma)
        eigen,U=np.linalg.eigh(K)
        cutoff=max(1.,float(eigen.max()))*1e-12
        use=eigen>cutoff
        if not use.any():raise ValueError("Empty numerical kernel space")
        self.projector=U[:,use]/np.sqrt(eigen[use])
        Phi=U[:,use]*np.sqrt(eigen[use])
        M=np.zeros((len(y),len(y)))
        metric_labels=[]
        if self.metric!="energy" and self.strength>0:
            if metric_blocks is None:raise ValueError("Physical metric blocks missing")
            for e in self.training_elements:
                block=metric_blocks.get(e)
                if block is None:raise ValueError("Missing training element metric: "+e)
                ids=np.flatnonzero([r["element"]==e for r in records])
                label_ids=[records[i]["label_id"] for i in ids]
                if label_ids!=list(block["label_ids"]):
                    raise ValueError("Metric-label order mismatch for "+e)
                J=np.asarray(block["jacobian"],float)
                if J.size:
                    J=J.reshape((-1,len(ids)))
                    A=(J.T@J)/(len(J)*len(self.training_elements)*self.property_scale**2)
                    if self.metric=="diagonal":A=np.diag(np.diag(A))
                    M[np.ix_(ids,ids)]=A
                metric_labels.extend(label_ids)
        Q=np.diag(w)+self.strength*M
        F=Phi.T@Q@Phi+self.alpha*np.eye(Phi.shape[1])
        rhs=Phi.T@Q@(y-self.mean)
        self.beta=cho_solve(cho_factor(F,lower=True,check_finite=True),rhs)
        self.dual=self.projector@self.beta
        self.diagnostics=dict(kernel_rank=int(use.sum()),labels=len(y),
            kernel_cutoff=cutoff,kernel_reconstruction_error=float(abs(Phi@Phi.T-K).max()),
            normal_equation_residual=float(np.linalg.norm(F@self.beta-rhs)/max(np.linalg.norm(rhs),1.)),
            normal_equation_condition=float(np.linalg.cond(F)),
            metric_trace=float(np.trace(M)),
            metric_training_elements=self.training_elements if metric_labels else [],
            metric_label_ids=metric_labels)
        return self

    def predict(self,records):
        X=(features(records)-self.center)/self.scale
        return self.mean+rbf_kernel(X,self.X,gamma=self.gamma)@self.dual
