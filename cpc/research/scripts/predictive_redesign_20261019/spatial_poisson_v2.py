"""Periodic nonlinear Poisson--Fermi solve and fixed-T/material implicit Jacobian.

u is in eV (numerically the corresponding potential in V); densities in cm^-3.
The atomic profile is fixed before solving the field. No fitting occurs here.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from sandbox.ionized_transport import EPSILON0,E_CHARGE
from spatial_local_charge_v2 import FrozenChargeKernel,LocalCharge

LN10=float(np.log(10.))


@dataclass
class SpatialChargeSolution:
    fermi_eV:np.ndarray
    fermi_jacobian:np.ndarray
    local:LocalCharge
    log_state_jacobian:np.ndarray
    diagnostics:dict


def negative_laplacian_matrix(N):
    D=2*np.eye(N)-np.roll(np.eye(N),1,axis=1)-np.roll(np.eye(N),-1,axis=1)
    return N*N*D


def laplacian(value):
    v=np.asarray(value,float)
    return len(v)**2*((v-np.roll(v,1))+(v-np.roll(v,-1)))


def solve_poisson(kernel:FrozenChargeKernel,stocks,*,background_cm3,period_nm,initial_eV,
                  stock_jacobian=None,background_jacobian=None,log_period_jacobian=None,
                  epsilon_r=10.2,max_iterations=120,residual_tolerance=2e-10,
                  step_tolerance_eV=3e-12,energy_bounds=(-5.,12.)):
    c=np.asarray(stocks,float);Nd=np.asarray(background_cm3,float);length=float(period_nm)
    if c.ndim!=2 or c.shape[0]!=kernel.parents or c.shape[1]<8 or c.shape[1]%2:
        raise ValueError('Even periodic grid>=8 with the explicit parent inventory required')
    if Nd.ndim==0:Nd=np.full(c.shape[1],float(Nd))
    if Nd.shape!=(c.shape[1],):raise ValueError('Scalar or local background[N] required')
    if not np.isfinite(c).all() or np.any(c<0) or not np.isfinite(Nd).all() or not np.isfinite([length,epsilon_r]).all() or min(float(Nd.min()),length,epsilon_r)<=0:
        raise ValueError('Finite stocks, positive background/period/permittivity required')
    N=c.shape[1];u0=np.asarray(initial_eV,float)
    if u0.ndim==0:u0=np.full(N,float(u0))
    if u0.shape!=(N,) or not np.isfinite(u0).all():raise ValueError('Finite scalar or N-vector initial field required')
    supplied=[v is not None for v in [stock_jacobian,background_jacobian,log_period_jacobian]]
    if any(supplied) and not all(supplied):raise ValueError('Provide all three derivative inputs or none')
    dc=np.zeros((*c.shape,0)) if not any(supplied) else np.asarray(stock_jacobian,float)
    if dc.ndim!=3 or dc.shape[:2]!=c.shape:raise ValueError('Stock Jacobian must have shape[parents,N,parameters]')
    npar=dc.shape[2]
    dNd=np.zeros(npar) if background_jacobian is None else np.asarray(background_jacobian,float)
    dL=np.zeros(npar) if log_period_jacobian is None else np.asarray(log_period_jacobian,float)
    if dNd.shape==(npar,):dNd=np.broadcast_to(dNd,(N,npar)).copy()
    if dNd.shape!=(N,npar) or dL.shape!=(npar,) or not all(np.isfinite(v).all() for v in [dc,dNd,dL]):
        raise ValueError('Matching finite stock/background/log10-period derivatives required')
    D=negative_laplacian_matrix(N)
    C=epsilon_r*EPSILON0/(E_CHARGE*1e6*(length*1e-9)**2)
    inventory_scale=Nd+np.max(abs(kernel.q),axis=1)@c
    # A depleted cell can have tiny ionic stock but finite Poisson curvature
    # and electron tail. Avoid demanding an absolute residual below floating
    # point resolution relative to that vanishing stock. This fixed global
    # normalization changes only Newton merit/diagnostics; the unscaled root,
    # exact Jacobian and separate field-correction stopping criterion remain.
    scale=np.maximum(inventory_scale,max(float(inventory_scale.mean()),1.))
    low,high=map(float,energy_bounds)
    if not low<high or np.any(u0<=low) or np.any(u0>=high):raise ValueError('Initial field must be inside numerical energy bounds')
    left=kernel.evaluate(np.full(N,low),c,background_cm3=Nd)
    right=kernel.evaluate(np.full(N,high),c,background_cm3=Nd)
    if not (np.all(left.charge_cm3>0) and np.all(right.charge_cm3<0)):
        raise ValueError('No constant sub/supersolutions inside declared numerical field bounds')
    origin=float(np.mean(u0));v=u0-origin;calls=2;history=[];converged=False

    def evaluate(center,offset):
        nonlocal calls
        local=kernel.evaluate(center+offset,c,background_cm3=Nd);calls+=1
        Lv=laplacian(offset);R=C*Lv-local.charge_cm3
        J=C*D-np.diag(local.charge_EF_partial)
        normal=float(np.max(np.diag(J)))
        factor=cho_factor(J/normal,lower=True,check_finite=False)
        correction=cho_solve(factor,R/normal,check_finite=False)
        merit=float(np.mean((R/scale)**2))
        return local,Lv,R,J,normal,factor,correction,merit

    state=evaluate(origin,v)
    for iteration in range(max_iterations+1):
        local,Lv,R,J,normal,factor,correction,merit=state
        relative=float(np.max(abs(R)/scale));step=float(max(abs(correction)))
        history.append(dict(relative_residual=relative,estimated_correction_eV=step))
        if relative<=residual_tolerance and step<=step_tolerance_eV:
            converged=True;break
        if iteration==max_iterations:break
        initial_factor=min(1.,.5/max(step,1e-300))
        accepted=False
        for backtrack in range(45):
            alpha=initial_factor*2.**(-backtrack)
            trial=v-alpha*correction;shift=float(trial.mean());neworigin=origin+shift;trial-=shift
            if np.any(neworigin+trial<=low) or np.any(neworigin+trial>=high):continue
            candidate=evaluate(neworigin,trial)
            if candidate[-1]<=(1-1e-4*alpha)*merit:
                origin=neworigin;v=trial;state=candidate;accepted=True;break
        if not accepted:raise ValueError('Poisson residual backtracking stalled before declared convergence')
    if not converged:raise ValueError('Periodic Poisson iteration cap reached')
    local,Lv,R,J,normal,factor,correction,merit=state
    condition=float(np.linalg.cond(J/normal))
    if not np.isfinite(condition) or condition>1e12:raise ValueError('Poisson Jacobian outside declared numerical condition domain')
    direct=np.einsum('ij,ijk->jk',local.mean_q,dc)+dNd
    dC=-2*LN10*C*dL
    rhs=direct-Lv[:,None]*dC[None,:]
    du=cho_solve(factor,rhs/normal,check_finite=False)
    dn=local.carrier_EF_partial[:,0,None]*du
    dNI=np.einsum('ij,ijk->jk',local.mean_q2,dc)+dNd+local.Nq2_EF_partial[:,None]*du
    if np.any(local.carriers_cm3[:,0]<=0) or np.any(local.Nq2_cm3<=0):raise ValueError('Nonpositive transport state')
    logJ=np.stack([dn/(local.carriers_cm3[:,0,None]*LN10),dNI/(local.Nq2_cm3[:,None]*LN10)],axis=1)
    backward=float(np.max(abs(J@du-rhs)/(1+abs(J)@abs(du)+abs(rhs)))) if npar else 0.
    return SpatialChargeSolution(origin+v,du,local,logJ,dict(grid_size=N,period_nm=length,
        epsilon_r=epsilon_r,poisson_coefficient_cm3_per_eV=C,iterations=iteration,kernel_evaluations=calls,
        relative_residual=float(max(abs(R)/scale)),estimated_field_correction_eV=float(max(abs(correction))),
        global_charge_relative_error=float(abs(local.charge_cm3.mean())/inventory_scale.mean()),
        jacobian_condition=condition,implicit_linear_backward_error=backward,
        residual_history=history,field_span_eV=float(np.ptp(origin+v)),mean_fermi_eV=origin,
        fixed_T_material_and_cold_energy_partials_only=True,vector_background_supported=True,background_mean_cm3=float(Nd.mean()),numerical_energy_bounds=list(energy_bounds),
        atomic_profiles_not_solved_as_functions_of_cold_field=True))
