"""Independent constrained-primal, finite-difference and gauge checks."""
from pathlib import Path
import sys,time,numpy as np
from scipy.optimize import minimize,LinearConstraint,Bounds
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from capped_inventory_projection_v2 import project,objective,ProjectionDomainError

OUT=c.BASE/'capped_inventory_candidate/gates_v2'


def main():
    assert not (OUT/'protocol_before.json').exists()
    theory=c.ROOT/'docs/ML_SANDBOX_CAPPED_INVENTORY_CANDIDATE_20261020.md'
    core=Path(__file__).with_name('capped_inventory_projection_v2.py')
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),
        code_hashes=c.file_hashes([Path(__file__),core,Path(c.__file__)]),
        input_hashes=c.file_hashes([theory,c.BASE/'capped_inventory_candidate/gates_v1/first_failure.json']),data='synthetic constrained inventories only',
        no_real_training_or_prediction_gain_claim=True,finite_difference_step=2e-6,goal_complete=False))
    rng=np.random.default_rng(19371);start=time.perf_counter();checks=[];details=[]
    B=np.array([[-1,-1,-2,0,1,0],[0,0,0,-1,0,1],[1,1,1,0,0,0]],float)
    ref=np.array([.4,.2,.1,.04,.02,.07])*1e18
    cap=np.array([10,10,5,1,1,1])*1e18
    fields=np.array([[0,0,1,0,0,0],[0,0,0,1,1,1]],float).T
    for case in range(12):
        r=ref*np.exp(rng.normal(0,.3,len(ref)));u=cap.copy()
        g=fields@rng.uniform(-2.5,2.5,2)
        dr=r[:,None]*np.c_[np.ones(6),rng.normal(0,.2,(6,2)),np.zeros((6,3))]
        dg=np.c_[np.zeros((6,3)),fields,B.T@np.array([.2,-.3,.4])]
        dg[:,1]=.1*fields[:,0] # One coordinate affects both reference and field.
        result=project(r,u,B,g,reference_jacobian=dr,field_jacobian=dg)
        error=[]
        for k in range(6):
            h=2e-6
            plus=project(r*np.exp(h*dr[:,k]/r),u,B,g+h*dg[:,k]).concentrations
            minus=project(r*np.exp(-h*dr[:,k]/r),u,B,g-h*dg[:,k]).concentrations
            fd=(plus-minus)/(2*h)
            err=float(max(abs(fd-result.jacobian[:,k]))/(max(r)+max(abs(fd))+max(abs(result.jacobian[:,k]))))
            error.append(err);checks.append(err<2e-6)
        zero=project(r,u,B,np.zeros(6),reference_jacobian=dr,field_jacobian=np.zeros_like(dg))
        e0=float(max(abs(zero.concentrations-r))/max(r))
        j0=float(np.max(abs(zero.jacobian-dr))/max(r))
        gauge=project(r,u,B,g+B.T@rng.normal(size=3)).concentrations
        eg=float(max(abs(gauge-result.concentrations))/max(r))
        extra=project(r,u,np.vstack([B,2*B[0],np.zeros(6)]),g).concentrations
        er=float(max(abs(extra-result.concentrations))/max(r))
        checks.extend([e0<1e-12,j0<1e-10,eg<1e-9,er<1e-9,
            max(abs(result.jacobian[:,-1]))/max(r)<1e-10,
            bool(np.all(result.concentrations>0)),bool(np.all(result.concentrations<u)),
            result.diagnostics['chemistry_relative_error']<1e-9,
            result.diagnostics['derivative_chemistry_relative_error']<1e-9,
            result.diagnostics['original_objective']<=result.diagnostics['reference_objective']+max(r)*1e-10])
        # A separate primal optimizer uses neither dual logits nor implicit derivatives.
        unit=max(r);us=u/unit;z0=np.log(r)-np.log(u-r)
        fun=lambda v:objective(v*unit,r,u,g)/unit
        jac=lambda v:np.log(v/(us-v))-z0-g
        fit=minimize(fun,r/unit,jac=jac,method='SLSQP',
            constraints=[LinearConstraint(B,B@(r/unit),B@(r/unit))],
            bounds=Bounds(np.full(6,1e-12),us-1e-12),options=dict(ftol=1e-12,maxiter=500))
        ep=float(max(abs(fit.x-result.concentrations/unit)))
        eo=float(abs(fit.fun-fun(result.concentrations/unit)))
        checks.extend([fit.success,ep<1e-5,eo<1e-9])
        scales=[]
        for scale in [1e-12,1e12]:
            scaled=project(r*scale,u*scale,B,g).concentrations/scale
            err=float(max(abs(scaled-result.concentrations))/max(r));scales.append(err);checks.append(err<1e-9)
        details.append(dict(case=case,derivative_errors=error,zero_recovery_error=e0,zero_derivative_error=j0,
            gauge_error=eg,redundancy_error=er,primal_slsqp_success=bool(fit.success),
            primal_solution_error=ep,primal_objective_error=eo,scale_errors=scales,
            diagnostics=result.diagnostics))
    free=project(ref,cap,np.empty((0,6)),np.arange(6)/5.)
    from scipy.special import expit
    expected=cap*expit(np.log(ref)-np.log(cap-ref)+np.arange(6)/5.)
    checks.append(max(abs(free.concentrations-expected))/max(ref)<1e-12)
    for mode in ['zero','negative','at_cap']:
        bad=ref.copy();bad[0]=0. if mode=='zero' else -1. if mode=='negative' else cap[0]
        rejected=False
        try:project(bad,cap,B,np.zeros(6))
        except ProjectionDomainError:rejected=True
        checks.append(rejected)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'details.json',details)
    c.write(OUT/'gates.json',dict(passed=all(checks),checks=len(checks),failures=sum(not q for q in checks),
        max_derivative_error=max(max(d['derivative_errors']) for d in details),
        max_independent_primal_solution_error=max(d['primal_solution_error'] for d in details),
        max_gauge_error=max(d['gauge_error'] for d in details),seconds=time.perf_counter()-start,
        actual_material_states=0,new_fits=0,new_predictive_accuracy_claim=False,goal_complete=False))
    print('Capped-inventory mathematical gates',len(checks),'failed',sum(not q for q in checks),flush=True)
    assert all(checks),'Preserve first gate result and repair by version'


if __name__=='__main__':main()
