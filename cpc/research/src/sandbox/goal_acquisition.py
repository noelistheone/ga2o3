"""Gaussian moment conditioning and local prediction-directed label selection.

The algebra is exact for the stated Gaussian surrogate and linear goal. A
nonlinear Sandbox is linearized only for selection, then evaluated in full.
Neither Gaussian calibration nor global optimality follows from these formulas.
"""
from __future__ import annotations
import numpy as np


def validate_covariance(cov):
    C=np.asarray(cov,float)
    if C.ndim!=2 or C.shape[0]!=C.shape[1] or not np.isfinite(C).all():
        raise ValueError("Expected a finite square covariance")
    scale=max(float(np.linalg.norm(C,2)),1.)
    if not np.allclose(C,C.T,atol=1e-11*scale,rtol=1e-11):
        raise ValueError("Nonsymmetric covariance")
    if np.linalg.eigvalsh(C).min() < -1e-10*scale:
        raise ValueError("Nonpositive covariance")
    return .5*(C+C.T)


def condition_coordinate(mean,cov,index,value,noise_variance=0.):
    """Condition x_j+noise=value; exact published-label replay uses noise=0."""
    m=np.asarray(mean,float);C=validate_covariance(cov)
    if noise_variance<0:
        raise ValueError("Negative observation variance")
    den=float(C[index,index]+noise_variance)
    if den<=1e-14:
        raise ValueError("Coordinate has no remaining variance")
    column=C[:,index].copy()
    out_m=m+column*(float(value)-m[index])/den
    out_C=C-np.outer(column,column)/den
    out_C=.5*(out_C+out_C.T)
    if noise_variance==0:
        # Exact observed coordinate; remove roundoff in its zero covariance row.
        out_m[index]=float(value);out_C[index,:]=0.;out_C[:,index]=0.
    validate_covariance(out_C)
    return out_m,out_C


def prediction_variance_reduction(jacobian,cov,noise_variance=0.,weights=None):
    """Reduction in tr(W J C J^T) for each coordinate observation."""
    J=np.asarray(jacobian,float);C=validate_covariance(cov)
    if J.ndim!=2 or J.shape[1]!=len(C) or not np.isfinite(J).all():
        raise ValueError("Invalid goal Jacobian")
    if weights is None:weights=np.full(len(J),1/len(J))
    w=np.asarray(weights,float)
    if w.shape!=(len(J),) or np.any(w<0) or w.sum()<=0:
        raise ValueError("Invalid prediction weights")
    den=np.diag(C)+noise_variance
    proj=J@C
    numerator=(w[:,None]*proj**2).sum(axis=0)
    return np.divide(numerator,den,out=np.zeros_like(numerator),where=den>1e-14)
