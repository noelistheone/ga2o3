"""Independent constrained optimizer reference for affine source controls."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.optimize import minimize
from sandbox.affine_source import basis,fit_correction
import phase2_energy as energy
OUT=energy.BASE/'phase21_curve_response/affine_controls'


def main():
    checks=[];C=np.array([16.,17.,18.,19.,20.]);base=np.array([13.3,14.,15.,16.,17.])
    for kind in ('slope','affine'):
        V=basis(C,kind);G=V.T@V/len(V)
        for idx in ([1],[1,3]):
            for obs in (np.array([13.01,13.01])[:len(idx)],base[idx]+np.array([1.5,-.5])[:len(idx)]):
                for lam in (0.,.1,1.,100.):
                    for radius in (.2,6.):
                        r=fit_correction(base,C,idx,obs,lam,kind=kind,radius=radius);coef=r['coefficients']
                        def loss(c):return float(np.sum((base[idx]+V[idx]@c-obs)**2)+lam*(c@G@c))
                        op=minimize(loss,np.zeros(V.shape[1]),jac=lambda c:2*V[idx].T@(base[idx]+V[idx]@c-obs)+2*lam*G@c,
                            constraints=[dict(type='ineq',fun=lambda c:base+V@c-13.,jac=lambda c:V),
                                dict(type='ineq',fun=lambda c:radius*radius-c@G@c,jac=lambda c:-2*G@c)],
                            method='SLSQP',options=dict(ftol=1e-11,maxiter=300))
                        error=max(loss(coef)-op.fun,0.);feasible=bool(np.min(base+V@coef)>=13. and coef@G@coef<=radius*radius+1e-9)
                        checks.append(dict(kind=kind,anchors=idx,lam=lam,radius=radius,error=error,reference_success=bool(op.success),
                            feasible=feasible,passed=bool(error<2e-6 and feasible and op.success)))
    # A constant-dose design has only a rank-one identifiable affine response.
    r=fit_correction(np.array([17.,18.]),np.array([18.,18.]),[0],[17.2],1.,kind='affine')
    checks.append(dict(test='constant-dose rank',passed=r['rank']==1,error=0. if r['rank']==1 else 1.))
    result=dict(passed=all(r['passed'] for r in checks),checks=len(checks),rows=checks,
        maximum_objective_excess=max(r['error'] for r in checks),code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in
            (Path(__file__),ROOT/'src/sandbox/affine_source.py')},new_material_measurements=0)
    energy.write(OUT/'numerical_gates.json',result);print('affine gates',result['passed'],len(checks),'max',result['maximum_objective_excess'],flush=True)
    if not result['passed']:print([r for r in checks if not r['passed']],flush=True)
    assert result['passed']


if __name__=='__main__':main()
