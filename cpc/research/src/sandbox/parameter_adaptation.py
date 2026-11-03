"""Bounded, regularized local parameter adaptation with exact supplied Jacobians.

The callback solves a physical forward model. Its parameters remain effective
calibration coordinates: good predictions do not certify microscopic truth.
No data files, global material settings or training state are accessed here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import least_squares


@dataclass
class AdaptationResult:
    parameters: np.ndarray
    standardized_shift: np.ndarray
    status: str
    trials: list
    diagnostics: dict


def local_information(jacobian, scales, lam):
    """Local data resolution, not a calibrated posterior confidence interval."""
    J=np.asarray(jacobian,float)*np.asarray(scales,float)[None,:]
    active=np.flatnonzero(np.asarray(scales)>0)
    J=J[:,active]
    s=np.linalg.svd(J,compute_uv=False) if J.size else np.zeros(0)
    # Include all parameter-space null directions for underdetermined anchors.
    spectrum=np.r_[s,np.zeros(max(0,len(active)-len(s)))]
    tol=1e-8*max(float(s.max()) if len(s) else 0.,1.)
    rank=int((spectrum>tol).sum())
    resolution=(spectrum*spectrum/(spectrum*spectrum+lam)
                if np.isfinite(lam) and lam>0 else np.zeros(len(active)))
    return dict(standardized_jacobian=J, singular_values=spectrum,
        rank=rank,coordinates=len(active),
        condition=float(spectrum.max()/spectrum.min()) if rank==len(active) and rank else np.inf,
        data_resolution_eigenvalues=resolution,
        effective_data_dimensions=float(resolution.sum()),
        interpretation="local GaussNewton data/prior resolution;not physical identification or posterior CI")


def regularized_adapt(
    forward: Callable[[np.ndarray],tuple[np.ndarray,np.ndarray]],
    observed, initial, scales, lower, upper, lam, *, max_nfev=30,
):
    """Minimize ||forward(theta0+D z)-y||² + lam||z||² with bounds.

    Positive scales define active coordinates. Zero scales keep coordinates at
    their initial values. Nonlinear fitting uses zero and linear-MAP starts,
    selecting the best finite fit; failures and convergence status are saved.
    An all-start failure returns the declared base-model fallback explicitly.
    """
    y=np.asarray(observed,float);theta0=np.asarray(initial,float)
    D=np.asarray(scales,float);lo=np.asarray(lower,float);hi=np.asarray(upper,float)
    if not (theta0.ndim==1 and D.shape==lo.shape==hi.shape==theta0.shape):
        raise ValueError("Coordinate vector shapes must match")
    if not (np.isfinite(np.r_[theta0,D,lo,hi,y]).all() and np.all(D>=0)
            and np.all(lo<=theta0) and np.all(theta0<=hi) and np.all(lo<hi)):
        raise ValueError("Finite ordered bounds/scales and in-bounds initial coordinates required")
    if not (lam>0):raise ValueError("Positive regularization or infinity required")
    active=np.flatnonzero(D>0)
    if not len(y) or not len(active) or not np.isfinite(lam):
        return AdaptationResult(theta0.copy(),np.zeros_like(theta0),"base_no_update",[],
            dict(anchor_count=len(y),regularization=lam,reason="zero anchors, no active coordinate or infinite shrinkage"))
    zl=(lo[active]-theta0[active])/D[active]
    zh=(hi[active]-theta0[active])/D[active]
    failures=[];calls=0;cache={}
    def evaluate(z):
        nonlocal calls
        key=tuple(z)
        if key not in cache:
            theta=theta0.copy();theta[active]+=D[active]*z
            try:
                pred,J=forward(theta)
                pred=np.asarray(pred,float);J=np.asarray(J,float)
                if pred.shape!=y.shape or J.shape!=(len(y),len(theta0)):
                    raise ValueError("Forward value/Jacobian shape mismatch")
                if not np.isfinite(np.r_[pred,J.ravel()]).all():
                    raise ValueError("Nonfinite forward value/Jacobian")
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failures.append(dict(standardized_parameters=z.copy(),error=repr(exc)))
                raise
            calls+=1;cache.clear();cache[key]=(pred,J)
        return cache[key]
    trials=[]
    try:
        p0,J0=evaluate(np.zeros(len(active)))
        Jz=J0[:,active]*D[active]
        linear=-np.linalg.solve(Jz.T@Jz+lam*np.eye(len(active)),Jz.T@(p0-y))
        linear=np.clip(linear,zl,zh)
        starts=[np.zeros(len(active)),linear]
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        return AdaptationResult(theta0.copy(),np.zeros_like(theta0),"base_fallback",[],
            dict(anchor_count=len(y),regularization=lam,state_failures=failures,error=repr(exc)))
    for start in starts:
        try:
            result=least_squares(lambda z:np.r_[evaluate(z)[0]-y,np.sqrt(lam)*z],start,
                jac=lambda z:np.vstack([evaluate(z)[1][:,active]*D[active],np.sqrt(lam)*np.eye(len(active))]),
                bounds=(zl,zh),max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            trials.append(dict(standardized_parameters=result.x.copy(),
                loss=float(result.fun@result.fun),success=bool(result.success),
                active_mask=result.active_mask.copy(),nfev=int(result.nfev),message=result.message))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            trials.append(dict(standardized_parameters=start.copy(),loss=None,success=False,error=repr(exc)))
    finite=[t for t in trials if t.get("loss") is not None and np.isfinite(t["loss"])]
    if not finite:
        return AdaptationResult(theta0.copy(),np.zeros_like(theta0),"base_fallback",trials,
            dict(anchor_count=len(y),regularization=lam,state_failures=failures,forward_calls=calls))
    best=min(finite,key=lambda t:t["loss"])
    z=best["standardized_parameters"];theta=theta0.copy();theta[active]+=D[active]*z
    p,J=evaluate(z);shift=np.zeros_like(theta0);shift[active]=z
    info=local_information(J,D,lam)
    info.update(anchor_count=len(y),regularization=lam,forward_calls=calls,
        state_failures=failures,selected_start_converged=best["success"],
        anchor_predictions=p,anchor_residuals=p-y,objective=best["loss"],
        standardized_prior_penalty=float(lam*(z@z)),active_bounds=best["active_mask"])
    return AdaptationResult(theta,shift,"adapted" if best["success"] else "adapted_nonconverged",trials,info)
