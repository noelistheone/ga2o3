"""Independent analytic/derivative gates for finite-input source adaptation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.functional_source import ResponseDesign,solve_response
from sandbox.relative_predictor_v2 import SavedModel
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.kroger_db import load
from types import SimpleNamespace
import phase2_energy as energy
OUT=energy.BASE/'phase21_curve_response'


def main():
    checks=[]
    def check(name,error,tolerance,**extra):checks.append(dict(test=name,error=float(error),tolerance=tolerance,passed=bool(error<tolerance),**extra))
    for ja in ([1.,1.,1.],[1e-7,.4,2.],[.2,3.,.7]):
        j=np.array(ja);b=np.array([17.,18.,19.]);forward=lambda u:(b+j*u,j,b+j*u>=13.)
        for anchors in ([0],[0,1]):
            for pi in ([anchors[0]],anchors,[0,1,2]):
                for vi in (anchors,[0,1,2]):
                    design=ResponseDesign(forward,anchors,pi,vi,linear=bool(np.all(j==1.)))
                    y=b[anchors]+np.array([.21,-.13])[:len(anchors)]
                    solutions=solve_response(design,y,[0.,.1,1.,1e4,np.inf])['solutions']
                    for lam,r in zip([0.,.1,1.,1e4,np.inf],solutions):
                        optimum=0. if np.isinf(lam) else float(j[anchors]@(y-b[anchors])/(j[anchors]@j[anchors]+lam*np.mean(j[pi]**2)))
                        optimum=float(np.clip(optimum,*design.bounds));check('linear_quadratic_exact',abs(r['raw_shift']-optimum)/(1+abs(optimum)),2e-6,
                            anchors=anchors,penalty=pi,validity=vi,lam=None if np.isinf(lam) else lam)
                        check('declared_state_gate',float(not r['valid'] or not design.raw(r['raw_shift'])[2][vi].all()),.5)
    b=np.array([17.,18.,19.]);a=np.array([1.,2.,.5]);j0=np.array([.1,.3,.2])
    def nonlinear(u):
        y=b+a*np.arctan(u)+j0*u;j=a/(1+u*u)+j0;return y,j,y>=13.
    for pi in ([0],[0,1],[0,1,2]):
        design=ResponseDesign(nonlinear,[0,1],pi,[0,1,2]);obs=b[:2]+[.1,-.1]
        for u in (-.1,.1,.3):
            h=2e-5
            for prior in ('quadratic','log_tail'):
                f,g,p=design.evaluate(u,obs,.7,prior);fd=(design.evaluate(u+h,obs,.7,prior)[0]-design.evaluate(u-h,obs,.7,prior)[0])/(2*h)
                check('nonlinear_objective_derivative',abs(g-fd)/(1+abs(fd)),2e-7,penalty=pi,raw=u,prior=prior)
    # A fully independent dense scalar reference checks the nonlinear search;
    # it is not a physical material experiment or a global optimization proof.
    design=ResponseDesign(nonlinear,[0,1],[0,1,2],[0,1,2]);obs=b[:2]+[.4,.2]
    for prior in ('quadratic','log_tail'):
        for r in solve_response(design,obs,[0.,.1,1.,10.],kind=prior)['solutions']:
            grid=np.linspace(*design.bounds,4001);best=min(design.evaluate(u,obs,r['lam'],prior)[0] for u in grid)
            check('nonlinear_dense_reference',max(r['objective']-best,0.),2e-7,prior=prior,lam=r['lam'])
    # Actual physical IFT derivatives and a synthetic label generated from the
    # same retained trained model, with no experimental query labels supplied.
    record=next(json.loads(p.read_text()) for p in (energy.BASE/'phase20_source_inventory/delivery_v2/predictor_bundle_v7/models').glob('*.json')
        if json.loads(p.read_text())['method']=='inventory' and not json.loads(p.read_text()).get('reference_search_control'))
    model=SavedModel(record,profile=published_coherent_profile(),db=load());rows=[SimpleNamespace(element='Si',incorporated_cm3=C,Tobs_K=300.,H_known=False,H_cm3=0.,
        carbon_known=False,carbon_cm3=0.,growth_T_C=550.) for C in (1e18,1e19,1e20)]
    def physics(u):
        r=[model.point(x,source_kind='field',shift=u,require_observation=False) for x in rows]
        return np.array([a.values[0] for a in r]),np.array([a.source_jacobian[0] for a in r]),np.array([a.diagnostics['observation_domain_valid'] for a in r])
    design=ResponseDesign(physics,[0,1],[0,1,2],[0,1,2]);obs=physics(.02)[0][:2]
    for u in (.0,.01,.02):
        for prior in ('quadratic','log_tail'):
            h=2e-5;_,g,_=design.evaluate(u,obs,.1,prior);fd=(design.evaluate(u+h,obs,.1,prior)[0]-design.evaluate(u-h,obs,.1,prior)[0])/(2*h)
            check('physical_IFT_loss_derivative',abs(g-fd)/(1+abs(fd)),2e-5,raw=u,prior=prior)
    r=solve_response(design,obs,[0.,np.inf])['solutions'];check('physical_synthetic_source_recovery',abs(r[0]['raw_shift']-.02),2e-6)
    check('physical_zero_update_replay',abs(r[1]['raw_shift']),1e-14)
    payload=dict(passed=all(r['passed'] for r in checks),checks=len(checks),rows=checks,
        maximum_error=max(r['error'] for r in checks),physical_point_calls=design.points,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (Path(__file__),ROOT/'src/sandbox/functional_source.py')},
        new_material_measurements=0,experimental_query_labels_used=False,goal_complete=False)
    key=energy.sha(ROOT/'src/sandbox/functional_source.py')[:16];energy.write(OUT/'gates'/f'{key}.json',payload)
    energy.write(OUT/'numerical_gates.json',payload);print('gates',payload['passed'],len(checks),'max',payload['maximum_error'],flush=True)
    if not payload['passed']:print([r for r in checks if not r['passed']],flush=True)
    assert payload['passed']


if __name__=='__main__':main()
