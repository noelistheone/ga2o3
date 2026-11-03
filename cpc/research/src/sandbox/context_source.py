"""Finite-design source learning using only explicitly supplied Hall anchors."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from .functional_source import ResponseDesign


def solve_affine_context(base, log_doses, anchor_indices, observed, lambdas):
    """Strong two-coordinate output control on the same finite input design.

    At zero penalty with one anchor, choose zero slope explicitly. At positive
    penalty the unlabeled finite-design response supplies the regularization.
    No uniqueness of an experimental source slope is asserted.
    """
    base=np.asarray(base,float);x=np.asarray(log_doses,float);ai=np.asarray(anchor_indices,int);y=np.asarray(observed,float)
    if base.ndim!=1 or x.shape!=base.shape or y.shape!=ai.shape or not len(y) or not np.isfinite(np.r_[base,x,y]).all():
        raise ValueError('Finite declared base predictions, doses and anchors required')
    P=np.c_[np.ones(len(x)),x-x[ai[0]]];A=P[ai];r=y-base[ai];solutions=[]
    for lam in lambdas:
        if lam is None or np.isinf(lam):
            coef=np.zeros(2);rank=0;singular=[]
        elif lam==0. and len(y)==1:
            coef=np.array([r[0],0.]);rank=1;singular=[]
        else:
            X=np.r_[A/np.sqrt(len(y)),np.sqrt(lam)*P/np.sqrt(len(x))]
            b=np.r_[r/np.sqrt(len(y)),np.zeros(len(x))]
            coef,residual,rank,s=np.linalg.lstsq(X,b,rcond=None);singular=s.tolist()
        delta=P@coef;penalty=float(np.mean(delta*delta))
        objective=float(np.mean((base[ai]+delta[ai]-y)**2)+(0. if lam is None else lam*penalty))
        solutions.append(dict(lam=lam,source_parameters=coef.tolist(),response_delta=delta.tolist(),
            objective=objective,penalty=penalty,design_rank=int(rank),design_singular_values=singular,
            finite_response_bound_valid=bool(np.max(np.abs(delta))<=6.+1e-12),
            source_parameter_count=2,finite_input_design_only=True))
    return dict(solutions=solutions,basis='intercept and logdose centered onfirstanchor',
        one_anchor_zero_penalty_rule='explicit zero slope',global_convex_least_squares=True,
        constraint='accept only finite-design response |delta|<=6 and observationvalid;violations retained,not clipped')


def solve_context_source(forward, anchor_indices, observed, lambdas, *, linear,
                         nodes=33, local_candidates=3, maxiter=80):
    """Minimize mean anchor residual squared + lambda*mean design response².

    A sampled zero-adjacent branch and at most three local searches are used.
    No connectedness/global-optimum/continuous-domain certificate is claimed.
    """
    y=np.asarray(observed,float);ai=list(anchor_indices)
    if y.ndim!=1 or len(y)!=len(ai) or not len(y) or not np.isfinite(y).all():
        raise ValueError('Finite explicit anchor labels matching supplied indices required')
    baseline=forward(0.)
    all_indices=list(range(len(baseline[0])))
    design=ResponseDesign(forward,ai,all_indices,all_indices,linear=linear,point_budget=32768)
    lo,hi=design.bounds;m=len(y);solutions=[]
    half=(nodes-1)//2;u=np.linspace(0.,1.,half+1)**3
    fixed_grid=np.unique(np.r_[lo*u,hi*u,0.])
    for lam in lambdas:
        if lam is None or np.isinf(lam):
            solutions.append(dict(lam=None,raw_shift=0.,objective=float(np.mean((design.base-y)**2)),
                penalty=0.,valid=True,status='exact_no_update',searches=[],failures=[]))
            continue
        lam=float(lam);failures=[];searches=[]
        def evaluate(z):
            f,g,a=design.evaluate(float(z),y,m*lam,'quadratic')
            return f/m,g/m,a
        def objective(z):return evaluate(z)[0]
        def gradient(z):return evaluate(z)[1]
        candidates=[(objective(0.),0.)]
        if linear:
            analytic=float(np.clip(np.mean(y-design.base)/(1.+lam),lo,hi))
            candidates.extend((objective(z),float(z)) for z in (lo,hi,analytic))
            searches.append(dict(kind='exact_linear_offset',raw_shift=analytic,success=True))
        else:
            a=design.raw(0.);j=a[1][ai]
            den=float(np.mean(j*j)+lam*np.mean(a[1]**2))
            local=float(np.mean(j*(y-design.base))/den) if den>0 else 0.
            grid=np.unique(np.r_[fixed_grid,np.clip(local,lo,hi)])
            evaluated=[evaluate(z) for z in grid]
            vals=[v[0] for v in evaluated];grads=[v[1] for v in evaluated]
            candidates.extend((float(f),float(z)) for f,z in zip(vals,grid) if np.isfinite(f))
            brackets=[]
            for i in range(len(grid)-1):
                gl,gr=grads[i:i+2]
                if np.isfinite([gl,gr]).all() and gl<=0<=gr:
                    brackets.append(dict(left=i,right=i+1,kind='gradient_bracket',priority=min(vals[i:i+2])))
            for i,f in enumerate(vals):
                if not np.isfinite(f) or f>(vals[i-1] if i else np.inf) or f>(vals[i+1] if i+1<len(vals) else np.inf):continue
                l=max(0,i-1);r=min(len(grid)-1,i+1)
                if l==r or any(grid[l]<=grid[b['left']] and grid[b['right']]<=grid[r] for b in brackets):continue
                brackets.append(dict(left=l,right=r,kind='bounded_rescue',priority=f))
            brackets=sorted(brackets,key=lambda b:(b['priority'],b['left'],b['right']))[:local_candidates]
            for b in brackets:
                l=float(grid[b['left']]);r=float(grid[b['right']])
                try:
                    if b['kind']=='gradient_bracket':
                        z=float(brentq(gradient,l,r,xtol=1e-12,rtol=1e-12,maxiter=maxiter));success=True;nfev=None
                    else:
                        op=minimize_scalar(objective,bounds=(l,r),method='bounded',options=dict(xatol=1e-10,maxiter=maxiter))
                        z=float(op.x);success=bool(op.success);nfev=int(op.nfev)
                    f=objective(z);candidates.append((f,z))
                    searches.append(dict(kind=b['kind'],bounds=[l,r],raw_shift=z,objective=f,success=success,nfev=nfev))
                    if not success:failures.append('local search did not stop')
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    failures.append(repr(exc));searches.append(dict(kind=b['kind'],bounds=[l,r],success=False,error=repr(exc)))
        finite=[c for c in candidates if np.isfinite(c[0])]
        if not finite:
            solutions.append(dict(lam=lam,valid=False,error='no finite declared source candidate',searches=searches,failures=failures))
            continue
        value,z=min(finite,key=lambda a:(round(a[0],12),abs(a[1]),a[1]))
        value,grad,a=evaluate(z)
        solutions.append(dict(lam=lam,raw_shift=float(z),objective=value,gradient=grad,
            penalty=a['penalty'] if a else None,valid=bool(a is not None and not failures),
            searches=searches,failures=failures,at_sampled_boundary=bool(min(abs(z-lo),abs(z-hi))<1e-7)))
    return dict(solutions=solutions,source_vector_calls=design.calls,source_point_calls=design.points,
        failed_proposals=design.failures,raw_bounds=design.bounds,objective='mean anchor residual²+lambda mean finite-design response²',
        exact_general_global_optimum=False,finite_design_only=True)
