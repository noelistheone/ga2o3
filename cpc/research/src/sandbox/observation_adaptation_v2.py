"""Anchor-response coordinates for comparable internal/output regularization.

The first anchor defines a coordinate in observable dex. It does not provide a
query label. Numerical branch checks and all failed proposals are retained.
"""
import numpy as np
from scipy.optimize import brentq,minimize_scalar


class ObservationCoordinate:
    def __init__(self,forward,*,linear_shift=False,bounds=(-6.,6.),max_calls=4096):
        self.forward=forward;self.linear=bool(linear_shift);self.raw_bounds=tuple(bounds)
        self.max_calls=int(max_calls);self.cache={};self.inverse_cache={};self.failures=[];self.calls=0
        base=self._raw(0.)
        if base is None or not base[2][0]:raise ValueError('First anchor reference observation is invalid')
        self.base=base[0];self.baseline_valid=bool(base[2].all())
        if self.linear:self.bounds=(max(-6.,float(np.max(13.-self.base))),6.)
        else:
            grid=np.unique(np.r_[np.linspace(bounds[0],bounds[1],9),0.])
            values=[]
            for z in grid:
                a=self._raw(z)
                if a is None:raise ValueError('Cannot bracket the declared first-anchor branch')
                values.append(a[0][0])
            if np.min(np.diff(values)) < -1e-6:
                raise ValueError('Sampled first-anchor branch is not increasing')
            self.bounds=(max(-6.,13.-self.base[0],values[0]-self.base[0]),min(6.,values[-1]-self.base[0]))
        if not self.bounds[0]<=0<=self.bounds[1] or self.bounds[1]-self.bounds[0]<1e-8:
            raise ValueError('No nontrivial valid first-anchor response interval')

    def _raw(self,z):
        z=float(z)
        if z not in self.cache:
            if self.calls>=self.max_calls:raise RuntimeError('Declared anchor-vector solve budget exhausted')
            self.calls+=1
            try:
                y,J,valid=self.forward(z);y=np.asarray(y,float);J=np.asarray(J,float);valid=np.asarray(valid,bool)
                if y.ndim!=1 or J.shape!=y.shape or valid.shape!=y.shape or not np.isfinite(np.r_[y,J]).all():raise ValueError('Invalid source callback')
                self.cache[z]=(y,J,valid)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                self.failures.append(dict(raw_shift=z,error=repr(exc)));self.cache[z]=None
        return self.cache[z]

    def evaluate(self,effect):
        effect=float(effect)
        if not self.bounds[0]-1e-10<=effect<=self.bounds[1]+1e-10:return None
        effect=float(np.clip(effect,*self.bounds))
        if effect in self.inverse_cache:return self.inverse_cache[effect]
        if self.linear:
            z=effect;a=self._raw(z)
            if a is None:return None
            y,J,valid=a
            if np.max(np.abs(y-self.base-effect))>2e-8:raise ValueError('Declared output-shift identity failed')
        else:
            target=self.base[0]+effect
            if abs(effect)<1e-13:z=0.
            else:
                points=sorted((z,a[0][0]) for z,a in self.cache.items() if a is not None)
                # A monotone floating-point response can have two distinct
                # raw coordinates with EXACTLY the target value. Both enter
                # <= and >= sets, reversing the old strict bracket. Return
                # the exact cached root first; do not relax any tolerance or
                # suppress genuine nonmonotonicity.
                exact=[p for p in points if p[1]==target]
                if exact:z=min(exact,key=lambda p:(abs(p[0]),p[0]))[0]
                else:
                    below=[(z,y) for z,y in points if y<target];above=[(z,y) for z,y in points if y>target]
                    if not below or not above:return None
                    lo=max(below,key=lambda p:p[0])[0];hi=min(above,key=lambda p:p[0])[0]
                    if lo>hi:raise ValueError('Cached inverse branch is not ordered')
                    def residual(t):
                        a=self._raw(t)
                        if a is None:raise ValueError('First-anchor inverse encountered an invalid numeric state')
                        return a[0][0]-target
                    z=brentq(residual,lo,hi,xtol=1e-14,rtol=1e-15,maxiter=100)
            a=self._raw(z)
            if a is None or a[1][0]<=1e-12:return None
            y,J,valid=a[0],a[1]/a[1][0],a[2]
            if abs(y[0]-target)>2e-6:raise ValueError('First-anchor inverse did not meet observable tolerance')
        result=dict(effect=effect,raw_shift=float(z),predictions=y,jacobian=J,valid=bool(valid.all()))
        self.inverse_cache[effect]=result;return result


def penalty(z,lam,kind):
    if kind=='quadratic':return lam*z*z
    if kind=='log_tail':return lam*np.log1p(z*z)
    raise ValueError('Declared quadratic or logarithmic-tail penalty required')


def solve_effect(coordinate,observed,lambdas,*,kind='quadratic',nodes=33,maxiter=80):
    y=np.asarray(observed,float)
    if y.shape!=coordinate.base.shape or not np.isfinite(y).all():raise ValueError('Finite anchor labels required')
    lo,hi=coordinate.bounds;solutions=[]
    def measure(z,lam):
        a=coordinate.evaluate(z)
        if a is None or not a['valid']:return np.inf,None
        residual=a['predictions']-y;return float(residual@residual+penalty(z,lam,kind)),a
    for lam in lambdas:
        lam=float(lam);trials=[]
        if np.isinf(lam):
            value,a=measure(0.,0.);solutions.append(dict(lam=None,effect=0.,raw_shift=0.,objective=value,
                predictions=None if a is None else a['predictions'],valid=a is not None,searches=trials));continue
        candidates=[]
        # A shift model has exactly y_i=base_i+effect. A single physical
        # anchor has that identity by coordinate construction. Solve those
        # cases analytically, including all real cubic stationary points.
        analytic=coordinate.linear or len(y)==1
        if analytic:
            m=len(y) if coordinate.linear else 1;R=float(np.sum(y-coordinate.base))
            roots=[R/(m+lam)] if kind=='quadratic' else np.roots([m,-R,m+lam,-R])
            zs=[lo,hi,0.]
            zs += [float(np.real(r)) for r in roots if abs(np.imag(r))<1e-9 and lo<=np.real(r)<=hi]
            zs += [float(np.clip(R/m,lo,hi))]
            for z in zs:candidates.append((measure(z,lam)[0],float(z)))
        else:
            grid=np.unique(np.r_[np.linspace(lo,hi,nodes),0.,np.clip(y[0]-coordinate.base[0],lo,hi)])
            values=[measure(z,lam)[0] for z in grid];candidates=[(v,float(z)) for v,z in zip(values,grid)]
            for i,v in enumerate(values):
                if not np.isfinite(v):continue
                left=values[i-1] if i else np.inf;right=values[i+1] if i+1<len(grid) else np.inf
                if v>left or v>right:continue
                a=grid[max(0,i-1)];b=grid[min(len(grid)-1,i+1)]
                if a==b:continue
                op=minimize_scalar(lambda z:measure(z,lam)[0],method='bounded',bounds=(a,b),options=dict(xatol=1e-6,maxiter=maxiter))
                trials.append(dict(effect=float(op.x),success=bool(op.success),nfev=int(op.nfev),objective=float(op.fun)))
                candidates.append((float(op.fun),float(op.x)))
        finite=[v for v in candidates if np.isfinite(v[0])]
        if not finite:
            solutions.append(dict(lam=lam,valid=False,error='no feasible anchor response',searches=trials));continue
        value,z=min(finite,key=lambda t:(round(t[0],12),abs(t[1]),t[1]));_,a=measure(z,lam)
        solutions.append(dict(lam=lam,kind=kind,valid=True,effect=float(z),raw_shift=a['raw_shift'],objective=value,
            predictions=a['predictions'],first_anchor_identity_error=float(abs(a['predictions'][0]-coordinate.base[0]-z)),
            analytic_shift_solution=analytic,searches=trials))
    return dict(solutions=solutions,anchor_vector_calls=coordinate.calls,failed_proposals=coordinate.failures,
        observable_bounds=list(coordinate.bounds),raw_bounds=list(coordinate.raw_bounds),
        exact_general_global_optimum=False,branch='sampled increasing branch connected to first-anchor zero reference')
