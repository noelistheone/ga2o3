"""Scalar source adaptation with an explicit finite-input response penalty.

Only supplied anchor labels enter the loss. Other supplied inputs define a
finite design, not a continuum validity certificate or a Bayesian posterior.
"""
import numpy as np
from scipy.optimize import brentq,minimize_scalar


class ResponseDesign:
    def __init__(self,forward,anchors,penalty_indices,valid_indices,*,linear=False,
                 raw_bounds=(-6.,6.),point_budget=32768):
        self.forward=forward;self.anchors=np.asarray(anchors,int);self.penalty_indices=np.asarray(penalty_indices,int)
        self.valid_indices=np.asarray(valid_indices,int);self.linear=bool(linear);self.raw_bounds=tuple(map(float,raw_bounds))
        self.cache={};self.failures=[];self.calls=0;self.points=0;self.point_budget=int(point_budget);self.bounds=None
        if not len(self.anchors) or not len(self.penalty_indices) or not len(self.valid_indices):raise ValueError('Nonempty explicit anchor/penalty/validity designs required')
        base=self.raw(0.)
        if base is None:raise ValueError('Reference source evaluation failed')
        self.n=len(base[0]);self.base_all=base[0].copy();self.base=self.base_all[self.anchors]
        for idx in (self.anchors,self.penalty_indices,self.valid_indices):
            if idx.min()<0 or idx.max()>=self.n:raise ValueError('Design index outside supplied inputs')
        if not self.feasible(base):raise ValueError('Reference fails the declared finite-input validity gate')
        self.baseline_sensitivity=base[1].copy()
        self.bounds=self.branch_bounds()

    def raw(self,u):
        u=float(u)
        if u not in self.cache:
            if self.calls and self.points+self.n>self.point_budget:raise RuntimeError('Declared scalar-source point budget exhausted')
            self.calls+=1
            try:
                y,j,valid=self.forward(u);y=np.asarray(y,float);j=np.asarray(j,float);valid=np.asarray(valid,bool)
                if y.ndim!=1 or j.shape!=y.shape or valid.shape!=y.shape:raise ValueError('Source callback shape mismatch')
                self.points+=len(y);self.cache[u]=(y,j,valid)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                self.failures.append(dict(raw_shift=u,error=repr(exc)));self.cache[u]=None
        return self.cache[u]

    def feasible(self,a):
        if a is None:return False
        idx=np.unique(np.r_[self.anchors,self.penalty_indices,self.valid_indices])
        return bool(np.isfinite(a[0][idx]).all() and np.isfinite(a[1][idx]).all() and a[2][self.valid_indices].all())

    def response(self,a):
        idx=self.penalty_indices;d=a[0][idx]-self.base_all[idx]
        return float(np.mean(d*d)),float(2*np.mean(d*a[1][idx]))

    def branch_bounds(self):
        # This identifies a sampled component adjacent to zero. Every later
        # candidate is checked again. No global connectedness theorem follows.
        result=[]
        for end in self.raw_bounds:
            good=0.;boundary=float(end)
            for u in np.linspace(0.,end,9)[1:]:
                a=self.raw(u)
                if self.feasible(a) and self.response(a)[0]<=36.+1e-12:good=float(u);continue
                bad=float(u)
                for _ in range(40):
                    mid=(good+bad)/2.;p=self.raw(mid)
                    if self.feasible(p) and self.response(p)[0]<=36.:good=mid
                    else:bad=mid
                    if abs(good-bad)<=1e-11:break
                boundary=good;break
            result.append(boundary)
        if result[0]>0 or result[1]<0 or result[1]-result[0]<=1e-12:raise ValueError('No usable sampled source branch around zero')
        return tuple(result)

    def evaluate(self,u,observed,lam,kind):
        a=self.raw(u)
        if not self.feasible(a):return np.inf,np.nan,None
        q,dq=self.response(a);r=a[0][self.anchors]-observed
        if q>36.+1e-9:return np.inf,np.nan,None
        if kind=='quadratic':p,dp=q,dq
        elif kind=='log_tail':p,dp=np.log1p(q),dq/(1+q)
        else:raise ValueError('Declared quadratic or log-tail penalty required')
        value=float(r@r+lam*p);grad=float(2*r@a[1][self.anchors]+lam*dp)
        return value,grad,dict(predictions=a[0][self.anchors],penalty=q,penalty_derivative=dq,
            penalty_effect=float(np.copysign(np.sqrt(q),u)),raw_shift=float(u),valid=True)


def solve_response(design,observed,lambdas,*,kind='quadratic',nodes=33,maxiter=80):
    y=np.asarray(observed,float)
    if y.shape!=design.base.shape or not np.isfinite(y).all():raise ValueError('Finite explicit anchor labels required')
    lo,hi=design.bounds;solutions=[]
    half=(nodes-1)//2;unit=np.linspace(0.,1.,half+1)**3
    grid=np.unique(np.r_[lo*unit,hi*unit,0.])
    for lam in lambdas:
        lam=float(lam);trials=[];failures=[]
        if np.isinf(lam):
            value,grad,a=design.evaluate(0.,y,0.,kind)
            solutions.append(dict(lam=None,raw_shift=0.,objective=value,valid=a is not None,penalty=0.,searches=trials));continue
        def objective(u):return design.evaluate(float(u),y,lam,kind)[0]
        def gradient(u):return design.evaluate(float(u),y,lam,kind)[1]
        candidates=[(objective(0.),0.)]
        # Exact output-shift solutions preserve the earlier control. This is
        # the only analytic case claimed here; general nonlinear minima use
        # sampled brackets and derivatives of the actual physical solver.
        if design.linear:
            m=len(y);R=float(np.sum(y-design.base));zs=[lo,hi,0.]
            roots=[R/(m+lam)] if kind=='quadratic' else np.roots([m,-R,m+lam,-R])
            zs += [float(np.real(v)) for v in roots if abs(np.imag(v))<1e-9 and lo<=np.real(v)<=hi]
            candidates += [(objective(u),u) for u in zs]
        else:
            try:
                base=design.raw(0.);ja=base[1][design.anchors];v=float(np.mean(base[1][design.penalty_indices]**2))
                den=float(ja@ja+lam*v);local=float(ja@(y-design.base)/den) if den>0 else 0.
                active=np.unique(np.r_[grid,np.clip(local,lo,hi)])
                value=[];derivative=[]
                for u in active:
                    f,g,_=design.evaluate(u,y,lam,kind);value.append(f);derivative.append(g);candidates.append((f,float(u)))
                for i in range(len(active)-1):
                    gl,gr=derivative[i:i+2]
                    if not np.isfinite([gl,gr]).all() or gl>0 or gr<0:continue
                    if gl==0:root=float(active[i])
                    elif gr==0:root=float(active[i+1])
                    else:root=float(brentq(gradient,float(active[i]),float(active[i+1]),xtol=1e-12,rtol=1e-12,maxiter=maxiter))
                    candidates.append((objective(root),root));trials.append(dict(kind='gradient_bracket',raw_shift=root,objective=objective(root),success=True))
                # Rescue sampled minima lacking a derivative sign bracket,
                # including numerical non-smoothness adjacent to a boundary.
                for i,vv in enumerate(value):
                    if not np.isfinite(vv):continue
                    if vv>(value[i-1] if i else np.inf) or vv>(value[i+1] if i+1<len(value) else np.inf):continue
                    l=max(0,i-1);r=min(len(active)-1,i+1)
                    if l==r:continue
                    if any(active[l]<=t['raw_shift']<=active[r] for t in trials):continue
                    op=minimize_scalar(objective,bounds=(float(active[l]),float(active[r])),method='bounded',options=dict(xatol=1e-10,maxiter=maxiter))
                    candidates.append((float(op.fun),float(op.x)));trials.append(dict(kind='bounded_rescue',raw_shift=float(op.x),objective=float(op.fun),success=bool(op.success),nfev=int(op.nfev)))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:failures.append(repr(exc))
        finite=[r for r in candidates if np.isfinite(r[0])]
        if not finite:solutions.append(dict(lam=lam,valid=False,error='no feasible source candidate',searches=trials,failures=failures));continue
        value,u=min(finite,key=lambda r:(round(r[0],12),abs(r[1]),r[1]));value,grad,a=design.evaluate(u,y,lam,kind)
        solutions.append(dict(lam=lam,kind=kind,valid=a is not None and not failures,raw_shift=float(u),objective=value,gradient=grad,
            penalty=a['penalty'],penalty_effect=a['penalty_effect'],predictions=a['predictions'],searches=trials,failures=failures))
    return dict(solutions=solutions,source_vector_calls=design.calls,source_point_calls=design.points,
        raw_bounds=design.bounds,failed_proposals=design.failures,exact_general_global_optimum=False,
        branch='sampledzero-adjacentfinite-inputbranch;all reportedcandidateschecked')
