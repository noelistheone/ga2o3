"""Data-independent scalar support adaptation with bounded multibasin search.

This is a deterministic numerical search, not a proof of continuous global
optimality. Its callback receives source inputs; only provided anchor labels
enter the objective. It has no filesystem or experiment-table dependencies.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar


def learn_bounded_source(predict_anchors,observed,lambdas,*,bounds=(-6.,6.),grid_points=49,xatol=1e-5,maxiter=100):
    y=np.asarray(observed,float);lo,hi=map(float,bounds)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all():raise ValueError('Finite anchor labels required')
    if not lo<0<hi or grid_points<3:raise ValueError('Declared interval must contain zero and a finite grid')
    cache={};failures=[]
    def predictions(z):
        z=float(z)
        if z not in cache:
            try:
                p=np.asarray(predict_anchors(z),float)
                if p.shape!=y.shape or not np.isfinite(p).all():raise ValueError('Nonfinite anchor predictions')
                cache[z]=p
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failures.append(dict(z=z,error=repr(exc)));cache[z]=None
        return cache[z]
    def objective(z,lam):
        p=predictions(z)
        return np.inf if p is None else float((p-y)@(p-y)+lam*z*z)
    grid=np.unique(np.r_[np.linspace(lo,hi,grid_points),0.]);solutions=[]
    for lam in lambdas:
        lam=float(lam)
        if lam<0 or np.isnan(lam):raise ValueError('Nonnegative regularization required')
        if np.isinf(lam):
            p=predictions(0.)
            if p is None:raise ValueError('Unadapted source state fails')
            solutions.append(dict(lam=None,z=0.,objective=float((p-y)@(p-y)),anchor_predictions=p.tolist(),local_searches=[]))
            continue
        values=np.array([objective(z,lam) for z in grid]);candidates=[(v,float(z)) for v,z in zip(values,grid) if np.isfinite(v)]
        searches=[]
        for i in range(1,len(grid)-1):
            if not np.isfinite(values[i-1:i+2]).all():continue
            if values[i]<=values[i-1] and values[i]<=values[i+1] and (values[i]<values[i-1] or values[i]<values[i+1]):
                op=minimize_scalar(lambda z:objective(z,lam),method='bounded',bounds=(grid[i-1],grid[i+1]),
                    options=dict(xatol=xatol,maxiter=maxiter))
                searches.append(dict(bounds=[float(grid[i-1]),float(grid[i+1])],z=float(op.x),objective=float(op.fun),success=bool(op.success),nfev=int(op.nfev)))
                if np.isfinite(op.fun):candidates.append((float(op.fun),float(op.x)))
        if not candidates:raise ValueError('No finite support-adaptation candidate')
        value,z=min(candidates,key=lambda pair:(round(pair[0],12),abs(pair[1]),pair[1]))
        solutions.append(dict(lam=lam,z=z,objective=value,anchor_predictions=predictions(z).tolist(),local_searches=searches))
    return dict(solutions=solutions,grid=grid.tolist(),prediction_callbacks=len(cache),failed_probes=failures,
        boundary_rule='grid endpoints andzero retained;all sampled interior localbasins refined;ties roundobjective1e-12/minabsz',
        exact_global_continuous_optimum_claimed=False)
