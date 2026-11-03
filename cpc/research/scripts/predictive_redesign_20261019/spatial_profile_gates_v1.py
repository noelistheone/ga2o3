"""Synthetic independent checks of the frozen spatial parent-profile primitive."""
from pathlib import Path
import sys,numpy as np
from scipy.optimize import minimize,Bounds,LinearConstraint
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from spatial_parent_profile_v1 import parent_profile

OUT=c.BASE/'spatial_predictive_candidate/profile_gates_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes([
        Path(__file__),Path(__file__).with_name('spatial_parent_profile_v1.py')]),
        input_hashes=c.file_hashes([c.ROOT/'docs/ML_SANDBOX_SPATIAL_PREDICTIVE_MODEL_V2_20261020.md']),
        synthetic_only=True,new_models_fitted=0,goal_complete=False))
    caps=np.array([1e18,2e18,5e17,1e20,7e18]);ref=caps*np.array([1e-8,.01,.2,.7,.95])
    checks=[];details=[]
    for N in [8,16,64]:
        for a in [0.,.5,3.,10.,30.]:
            result=parent_profile(ref,caps,a,grid_size=N)
            checks.extend([result.diagnostics['parent_mass_relative_error']<2e-10,
                bool(np.all(result.concentrations>0)),bool(np.all(result.concentrations<=caps[:,None]))])
            h=2e-5
            plus=parent_profile(ref,caps,a+h,grid_size=N).concentrations
            minus=parent_profile(ref,caps,a-h,grid_size=N).concentrations
            fd=(plus-minus)/(2*h)
            error=float(np.max(abs(fd-result.amplitude_partial)/ref[:,None]));checks.append(error<2e-6)
            re=[]
            for i in range(len(ref)):
                rp=ref.copy();rm=ref.copy();rp[i]*=np.exp(h);rm[i]*=np.exp(-h)
                fd=(parent_profile(rp,caps,a,grid_size=N).concentrations[i]-parent_profile(rm,caps,a,grid_size=N).concentrations[i])/(2*h*ref[i])
                err=float(max(abs(fd-result.reference_partial[i])));re.append(err);checks.append(err<2e-6)
            opposite=parent_profile(ref,caps,-a,grid_size=N).concentrations
            shift_error=float(np.max(abs(opposite-np.roll(result.concentrations,N//2,axis=1))/ref[:,None]));checks.append(shift_error<1e-10)
            if a==0:checks.append(np.array_equal(result.concentrations,np.broadcast_to(ref[:,None],result.concentrations.shape)))
            details.append(dict(N=N,amplitude=a,amplitude_derivative_error=error,reference_derivative_errors=re,
                half_period_shift_error=shift_error,diagnostics=result.diagnostics))
    independent=[]
    for fraction in [.01,.6]:
        N=8;a=3.;result=parent_profile(np.array([fraction]),np.ones(1),a,grid_size=N)
        cosine=np.cos(2*np.pi*result.positions);g=np.log(2)-np.logaddexp(0,-a*cosine)
        z0=np.log(fraction/(1-fraction))
        def fun(x):
            return float(np.mean(x*np.log(x/fraction)+(1-x)*np.log((1-x)/(1-fraction))-g*x))
        def jac(x):return (np.log(x/(1-x))-z0-g)/N
        opt=minimize(fun,np.full(N,fraction),jac=jac,method='SLSQP',
            bounds=Bounds(np.full(N,1e-12),np.full(N,1-1e-12)),
            constraints=[LinearConstraint(np.ones((1,N))/N,[fraction],[fraction])],options=dict(ftol=1e-13,maxiter=1000))
        error=float(max(abs(opt.x-result.concentrations[0])))
        checks.extend([opt.success,error<2e-6]);independent.append(dict(reference=fraction,success=bool(opt.success),max_error=error))
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'details.json',dict(cases=details,independent_primal=independent))
    c.write(OUT/'gates.json',dict(passed=all(checks),checks=len(checks),failures=sum(not x for x in checks),
        max_amplitude_derivative_error=max(d['amplitude_derivative_error'] for d in details),
        max_reference_derivative_error=max(max(d['reference_derivative_errors']) for d in details),
        max_independent_primal_error=max(d['max_error'] for d in independent),
        actual_material_states=0,new_models_fitted=0,goal_complete=False))
    print('Spatial parent-profile gates',len(checks),'failed',sum(not x for x in checks),flush=True)
    assert all(checks),'Preserve first profile gate before versioned repair'


if __name__=='__main__':main()
