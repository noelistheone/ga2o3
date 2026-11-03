"""Research primitive: capped relative-entropy allocation with exact fixed B,u.

This is a statistical inventory closure, not a measured material free energy.
No model fitting, cold-state dependence, or outcome labels are handled here.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import qr
from scipy.special import expit,xlogy


class ProjectionDomainError(ValueError):pass


@dataclass
class ProjectedInventory:
    concentrations:np.ndarray
    jacobian:np.ndarray
    diagnostics:dict


def objective(c,reference,cap,field):
    r=np.asarray(c)/cap;r0=np.asarray(reference)/cap
    return float(np.sum(cap*(xlogy(r,r/r0)+xlogy(1-r,(1-r)/(1-r0))))-np.dot(field,c))


def project(reference,cap,B,field,*,reference_jacobian=None,field_jacobian=None,
            rank_rtol=1e-12,whitening_singular_ratio_min=1e-14,
            residual_tolerance=2e-11,max_iterations=100):
    """Return c and dc for fixed stoichiometry B and fixed per-parent caps.

    All reference parents must be strictly interior. Coordinate scalings and
    whiteners are held locally fixed for IFT; the primal is invariant to them.
    Boundaries, new supports, dB, or du require a separately derived version.
    """
    ref=np.asarray(reference,float);upper=np.asarray(cap,float)
    atoms=np.asarray(B,float);g=np.asarray(field,float)
    if ref.ndim!=1 or upper.shape!=ref.shape or g.shape!=ref.shape:
        raise ProjectionDomainError('Matching one-dimensional reference/cap/field required')
    if atoms.ndim!=2 or atoms.shape[1]!=len(ref) or len(ref)==0:
        raise ProjectionDomainError('Explicit finite stoichiometry matrix required')
    if not all(np.isfinite(v).all() for v in [ref,upper,atoms,g]):
        raise ProjectionDomainError('Nonfinite projection inputs')
    if np.any(ref<=0) or np.any(ref>=upper):
        raise ProjectionDomainError('Strict reference interior required;no artificial concentration floor')
    if (reference_jacobian is None)!=(field_jacobian is None):
        raise ProjectionDomainError('Supply both derivative matrices, or neither')
    dc0=np.zeros((len(ref),0)) if reference_jacobian is None else np.asarray(reference_jacobian,float)
    dg=np.zeros_like(dc0) if field_jacobian is None else np.asarray(field_jacobian,float)
    if dc0.ndim!=2 or dc0.shape[0]!=len(ref) or dg.shape!=dc0.shape or not all(np.isfinite(v).all() for v in [dc0,dg]):
        raise ProjectionDomainError('Finite matching reference and field derivative matrices required')
    # Locally fixed concentration unit; no learnable density scale is introduced.
    unit=float(np.max(ref));r0=ref/upper;cs=ref/unit;us=upper/unit
    W0=cs*(1-r0)
    if len(atoms):
        _,R,pivot=qr(atoms.T,mode='economic',pivoting=True)
        diag=abs(np.diag(R));rank=int(np.sum(diag>rank_rtol*diag[0])) if len(diag) and diag[0]>0 else 0
    else:rank=0;pivot=np.array([],int)
    independent=np.asarray(pivot[:rank],int)
    reduced=atoms[independent]
    if rank:
        U,s,V=np.linalg.svd(reduced*np.sqrt(W0)[None,:],full_matrices=False)
        ratio=float(s[-1]/s[0])
        if ratio<whitening_singular_ratio_min or s[-1]<=0:
            raise ProjectionDomainError('Reference weighted constraints outside whitening condition domain')
        A=(U.T@reduced)/s[:,None]
        K0=(A*W0)@A.T
        # This removes only an exact constraint-row gauge. Its derivative
        # contributes another row-space field, annihilated by primal IFT.
        gauge=np.linalg.solve(K0,A@(W0*g))
        geff=g-A.T@gauge
    else:
        A=np.empty((0,len(ref)));ratio=1.;geff=g.copy()
    z0=np.log(ref)-np.log(upper-ref)

    def evaluate(lam):
        h=geff-A.T@lam
        small=abs(h)<50
        stock=us*expit(z0+h);delta=stock-cs
        # Avoid losing conservation residuals by subtracting almost equal stocks.
        t=np.expm1(h[small])
        delta[small]=cs[small]*(1-r0[small])*t/(1+r0[small]*t)
        stock[small]=cs[small]+delta[small]
        W=stock*(1-stock/us)
        return stock,delta,W,A@delta

    lam=np.zeros(rank);history=[];converged=False
    for iteration in range(max_iterations+1):
        stock,delta,W,F=evaluate(lam)
        if np.any(W<=0) or not np.isfinite(stock).all():
            raise ProjectionDomainError('Finite-precision saturation;no clipped derivative served')
        residual=float(np.max(abs(F))) if rank else 0.
        history.append(residual)
        if residual<=residual_tolerance:
            converged=True;break
        if iteration==max_iterations:break
        K=(A*W)@A.T
        if np.linalg.cond(K)>1e12:
            raise ProjectionDomainError('Current dual Hessian outside declared condition domain')
        step=np.linalg.solve(K,F);norm=float(F@F)
        accepted=False
        for backtrack in range(40):
            factor=2.**(-backtrack)
            candidate=lam+factor*step
            trial,_,Wt,Ft=evaluate(candidate)
            trialnorm=float(Ft@Ft)
            if np.all(Wt>0) and np.isfinite(trial).all() and (trialnorm<=(1-1e-4*factor)*norm or
                    (max(abs(Ft)) if rank else 0.)<=residual_tolerance):
                lam=candidate;accepted=True;break
        if not accepted:raise ProjectionDomainError('Dual backtracking failed;preserve candidate')
    if not converged:raise ProjectionDomainError('Dual iteration cap reached')
    stock=stock*unit;W=W*unit
    # Use original g differential; all omitted gauge/whitener differentials
    # lie in row(B) and cancel from the primal derivative.
    V=dc0/(ref*(1-r0))[:,None]+dg
    weighted=W[:,None]*V
    K=(A*W)@A.T
    if rank:
        dl=np.linalg.solve(K,A@weighted-A@dc0)
        jac=weighted-W[:,None]*(A.T@dl)
    else:jac=weighted
    budget=atoms@(stock-ref)
    denominator=1.+abs(atoms)@(abs(stock)+abs(ref))
    budget_error=float(max(abs(budget)/denominator)) if len(atoms) else 0.
    derivative_budget=atoms@(jac-dc0)
    ddenom=1.+abs(atoms)@(abs(jac)+abs(dc0))
    derivative_error=float(np.max(abs(derivative_budget)/ddenom)) if derivative_budget.size else 0.
    if max(budget_error,derivative_error)>1e-8:
        raise ProjectionDomainError('Original-scale chemistry or derivative check failed')
    return ProjectedInventory(stock,jac,dict(iterations=iteration,independent_rows=independent.tolist(),
        original_rows=len(atoms),rank=rank,reference_whitening_singular_ratio=ratio,
        residual_history=history,chemistry_relative_error=budget_error,
        derivative_chemistry_relative_error=derivative_error,
        original_objective=objective(stock,ref,upper,g),reference_objective=objective(ref,ref,upper,g),
        cap_fraction_min=float(min(stock/upper)),cap_fraction_max=float(max(stock/upper)),
        fixed_stoichiometry_and_caps_only=True,material_free_energy_claim=False))
