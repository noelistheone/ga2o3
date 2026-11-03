"""Verify strong affine source controls before any outer result evaluation."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.context_source import solve_affine_context
import phase2_energy as e


def main():
    rows=[]
    def check(name,error,tol):rows.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
    x=np.array([17.,18.,19.,20.]);base=np.array([16.2,17.1,17.8,18.3]);P=np.c_[np.ones(4),x-x[0]]
    ai=np.array([0,3]);truth=np.array([.3,.2]);y=base[ai]+P[ai]@truth
    lams=[None,0.,.01,.1,1.,10.]
    out=solve_affine_context(base,x,ai,y,lams)
    scaled=solve_affine_context(base,2*x+3,ai,y,lams)
    for a,b in zip(out['solutions'],scaled['solutions']):
        lam=a['lam'];coef=np.array(a['source_parameters']);delta=np.array(a['response_delta'])
        check(str(lam)+'/representation_invariance',np.max(np.abs(delta-b['response_delta'])),1e-11)
        if lam is None:
            check('no_update',np.max(abs(coef)),0.)
        else:
            grad=P[ai].T@(base[ai]+delta[ai]-y)/len(ai)+lam*P.T@delta/len(base)
            check(str(lam)+'/normal_equations',np.max(np.abs(grad)),1e-11)
            check(str(lam)+'/objective',abs(a['objective']-(np.mean((base[ai]+delta[ai]-y)**2)+lam*np.mean(delta**2))),1e-12)
            if lam==0.:check('exact_two_anchor_recovery',np.max(abs(coef-truth)),1e-12)
    one=solve_affine_context(base,x,[0],[base[0]+.3],[0.])['solutions'][0]
    check('one_anchor_zero_slope',abs(one['source_parameters'][1]),0.)
    bad=solve_affine_context(base,x,ai,y+20,[0.])['solutions'][0]
    check('out_of_response_domain_retained',float(bad['finite_response_bound_valid']),0.)
    check('not_clipped',float(max(bad['response_delta'])<=6),0.)
    paths=[Path(__file__),ROOT/'src/sandbox/context_source.py']
    value=dict(passed=all(r['passed'] for r in rows),checks=len(rows),rows=rows,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths},synthetic_only=True,goal_complete=False)
    e.write(e.BASE/'phase25_shared_process/affine_gates.json',value)
    print('phase25 affine gates',value['passed'],len(rows),flush=True)
    if not value['passed']:raise RuntimeError('Preserve failed affine gate before repair')


if __name__=='__main__':main()
