"""Finite one-dimensional source adaptation through a supplied physical map.

The map supplies values and exact derivatives, and raises for invalid states.
Observed anchors enter only the scalar objective. A finite grid with local
refinement is not a proof of the continuous global optimum or a posterior CI.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class ScalarProfileResult:
    shift: float
    status: str
    candidates: list
    trials: list
    diagnostics: dict


class ScalarProfile:
    def __init__(self, forward: Callable, *, lower=-6., upper=6., nodes=25, maxiter=40, xatol=1e-5):
        if not (np.isfinite([lower,upper]).all() and lower<0<upper and nodes>=3):
            raise ValueError("Finite ordered bounds containing zero and at least3nodes required")
        self.forward=forward;self.lower=float(lower);self.upper=float(upper)
        self.nodes=int(nodes);self.maxiter=int(maxiter);self.xatol=float(xatol)
        self.cache={};self.failures=[];self.forward_calls=0

    def evaluate(self, z):
        z=float(z)
        if not self.lower-1e-12<=z<=self.upper+1e-12:raise ValueError("Outside source-coordinate range")
        z=float(np.clip(z,self.lower,self.upper))
        if z not in self.cache:
            self.forward_calls+=1
            try:
                pred,jac=self.forward(z);pred=np.asarray(pred,float);jac=np.asarray(jac,float)
                if pred.ndim!=1 or pred.shape!=jac.shape or not np.isfinite(np.r_[pred,jac]).all():
                    raise ValueError("Finite one-dimensional prediction/derivative arrays required")
                self.cache[z]=(pred,jac)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                self.failures.append(dict(shift=z,error=repr(exc)));self.cache[z]=None
        return self.cache[z]

    def fit(self, observed, lam):
        y=np.asarray(observed,float)
        if y.ndim!=1 or not np.isfinite(y).all() or np.isnan(lam) or lam<0:
            raise ValueError("Finite anchor vector and nonnegative regularization required")
        before=self.forward_calls;failure_start=len(self.failures)
        if not len(y) or not np.isfinite(lam):
            return ScalarProfileResult(0.,"base_no_update",[],[],dict(anchor_count=len(y),regularization=lam,
                reason="zero anchors or infinite shrinkage",new_forward_calls=0,data_resolution=0.))
        candidates=[];trials=[]
        def measure(z,kind):
            value=self.evaluate(z)
            if value is None:return dict(shift=float(z),objective=np.inf,valid=False,kind=kind)
            prediction,jac=value
            if prediction.shape!=y.shape:raise ValueError("Anchor/value dimension mismatch")
            residual=prediction-y;data=float(residual@residual)
            return dict(shift=float(z),objective=data+lam*z*z,data_loss=data,prior_loss=lam*z*z,
                derivative=float(2*(residual@jac)+2*lam*z),prediction=prediction,jacobian=jac,valid=True,kind=kind)
        grid=np.unique(np.r_[np.linspace(self.lower,self.upper,self.nodes),0.])
        for z in grid:candidates.append(measure(z,"grid"))
        costs=[c["objective"] for c in candidates]
        # Include every grid-local minimum, including feasible-component edges.
        for i,c in enumerate(list(candidates)):
            if not c["valid"]:continue
            left=costs[i-1] if i else np.inf;right=costs[i+1] if i+1<len(grid) else np.inf
            if c["objective"]>left or c["objective"]>right:continue
            lo=grid[max(0,i-1)];hi=grid[min(len(grid)-1,i+1)]
            if lo==hi:continue
            try:
                op=minimize_scalar(lambda z:measure(z,"refinement_evaluation")["objective"],
                    method="bounded",bounds=(lo,hi),options=dict(maxiter=self.maxiter,xatol=self.xatol))
                trial=dict(shift=float(op.x),success=bool(op.success),nfev=int(op.nfev),
                    message=str(op.message),interval=[float(lo),float(hi)])
                trials.append(trial);candidates.append(measure(op.x,"refined"))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                trials.append(dict(success=False,interval=[float(lo),float(hi)],error=repr(exc)))
        finite=[c for c in candidates if c["valid"] and np.isfinite(c["objective"])]
        if not finite:
            return ScalarProfileResult(0.,"no_feasible_anchor_state",candidates,trials,
                dict(anchor_count=len(y),regularization=lam,state_failures=self.failures[failure_start:],
                    new_forward_calls=self.forward_calls-before))
        best=min(finite,key=lambda c:(c["objective"],abs(c["shift"])))
        info=float(best["jacobian"]@best["jacobian"])
        resolution=info/(info+lam) if info+lam>0 else 0.
        projected=best["shift"]-float(np.clip(best["shift"]-best["derivative"],self.lower,self.upper))
        dg=dict(anchor_count=len(y),regularization=lam,objective=best["objective"],
            anchor_predictions=best["prediction"],anchor_residuals=best["prediction"]-y,
            data_information=info,data_resolution=resolution,selected_kind=best["kind"],
            projected_gradient=projected,at_boundary=bool(min(best["shift"]-self.lower,self.upper-best["shift"])<1e-4),
            new_forward_calls=self.forward_calls-before,state_failures=self.failures[failure_start:],
            interpretation="finite-grid/local-refinement optimum and local GaussNewton resolution;not global proof or calibrated CI")
        return ScalarProfileResult(best["shift"],"profile_adapted",candidates,trials,dg)
