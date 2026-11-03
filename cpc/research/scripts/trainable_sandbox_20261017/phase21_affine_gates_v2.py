"""Refine failed independent SLSQP references with a second convex optimizer.

V1 gate failures and code remain intact. The source algorithm, test matrix and
acceptance tolerances are unchanged; only failed reference solves are refined.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint
from sandbox.affine_source import basis,fit_correction
import phase2_energy as energy
OUT=energy.BASE/'phase21_curve_response/affine_controls'


def main():
    checks=[];C=np.array([16.,17.,18.,19.,20.]);base=np.array([13.3,14.,15.,16.,17.]);refinements=[]
    for kind in ('slope','affine'):
        V=basis(C,kind);G=V.T@V/len(V)
        for idx in ([1],[1,3]):
            for obs in (np.array([13.01,13.01])[:len(idx)],base[idx]+np.array([1.5,-.5])[:len(idx)]):
                for lam in (0.,.1,1.,100.):
                    for radius in (.2,6.):
                        r=fit_correction(base,C,idx,obs,lam,kind=kind,radius=radius);coef=r['coefficients']
                        def loss(c):return float(np.sum((base[idx]+V[idx]@c-obs)**2)+lam*(c@G@c))
                        def jac(c):return 2*V[idx].T@(base[idx]+V[idx]@c-obs)+2*lam*G@c
                        op=minimize(loss,np.zeros(V.shape[1]),jac=jac,
                            constraints=[dict(type='ineq',fun=lambda c:base+V@c-13.,jac=lambda c:V),
                                dict(type='ineq',fun=lambda c:radius*radius-c@G@c,jac=lambda c:-2*G@c)],
                            method='SLSQP',options=dict(ftol=1e-11,maxiter=300))
                        initial_success=bool(op.success)
                        if not initial_success:
                            original=dict(message=str(op.message),objective=float(op.fun),parameters=op.x)
                            op=minimize(loss,np.zeros(V.shape[1]),jac=jac,hess=lambda c:2*V[idx].T@V[idx]+2*lam*G,
                                constraints=[LinearConstraint(V,13.-base,np.inf),NonlinearConstraint(lambda c:radius*radius-c@G@c,0.,np.inf,
                                    jac=lambda c:-2*G@c,hess=lambda c,v:-2*float(v[0])*G)],method='trust-constr',
                                options=dict(gtol=1e-10,xtol=1e-12,barrier_tol=1e-12,maxiter=1000))
                            refinements.append(dict(kind=kind,anchors=idx,lam=lam,radius=radius,initial=original,
                                refined_success=bool(op.success),refined_message=str(op.message),refined_objective=float(op.fun)))
                        error=max(loss(coef)-op.fun,0.);feasible=bool(np.min(base+V@coef)>=13. and coef@G@coef<=radius*radius+1e-9)
                        checks.append(dict(kind=kind,anchors=idx,lam=lam,radius=radius,error=error,initial_reference_success=initial_success,
                            reference_success=bool(op.success),feasible=feasible,passed=bool(error<2e-6 and feasible and op.success)))
    r=fit_correction(np.array([17.,18.]),np.array([18.,18.]),[0],[17.2],1.,kind='affine')
    checks.append(dict(test='constant-dose rank',passed=r['rank']==1,error=0. if r['rank']==1 else 1.))
    result=dict(passed=all(r['passed'] for r in checks),checks=len(checks),rows=checks,independent_reference_refinements=refinements,
        original_gate_retained='numerical_gates.json;5SLSQPreferencefailures,sourcefeasibility/objectivesalreadywithin tolerance',
        source_algorithm_unchanged=True,acceptance_tolerances_unchanged=True,maximum_objective_excess=max(r['error'] for r in checks),
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (Path(__file__),ROOT/'src/sandbox/affine_source.py')},new_material_measurements=0)
    energy.write(OUT/'numerical_gates_v2.json',result);print('affine V2gates',result['passed'],len(checks),'refinements',len(refinements),'max',result['maximum_objective_excess'],flush=True)
    if not result['passed']:print([r for r in checks if not r['passed']],flush=True)
    assert result['passed']


if __name__=='__main__':main()
