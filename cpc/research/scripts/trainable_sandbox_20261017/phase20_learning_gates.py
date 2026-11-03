"""Check observable-coordinate inference and the actual shared learning loss."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.observation_adaptation import ObservationCoordinate,solve_effect
import phase20_prediction as s


def main():
    checks=[]
    for count in (1,2):
        base=np.arange(count,dtype=float)+17.
        for r in (.1,2.,4.):
            y=base+r
            def linear(z):return base+z,np.ones(count),np.ones(count,dtype=bool)
            for prior in ('quadratic','log_tail'):
                coordinate=ObservationCoordinate(linear,linear_shift=True)
                result=solve_effect(coordinate,y,[0.,.1,1.,10.],kind=prior)
                for op in result['solutions']:
                    if prior=='quadratic':expected=count*r/(count+op['lam'])
                    else:
                        candidates=[-4.,6.]+[float(np.real(x)) for x in np.roots([count,-count*r,count+op['lam'],-count*r]) if abs(np.imag(x))<1e-8 and -4.<=np.real(x)<=6.]
                        expected=min(candidates,key=lambda z:count*(z-r)**2+op['lam']*np.log1p(z*z))
                    error=abs(op['effect']-expected)
                    checks.append(dict(test='exact_shift_solution',count=count,r=r,prior=prior,lam=op['lam'],error=float(error),passed=bool(error<1e-7)))
    # The true shared-loss gradient includes both the changed and h=0 states.
    tr=s.RT[s.RT.element.eq('Si')].iloc[[0,-1]];weights=s.weights(tr);y=tr.target.to_numpy(float)
    for method in s.PHYSICS:
        w=np.array([1.3,16.,.1]);gamma=.1
        def loss(w):
            v,J,d,dJ,valid=s.forward(method,w,tr,penalty=True)
            return float(weights@((v-y)**2)+gamma*(weights@(d*d)))
        v,J,d,dJ,_=s.forward(method,w,tr,penalty=True)
        analytic=2*((weights*(v-y))@J+gamma*(weights*d)@dJ)
        for k in range(3):
            h=2e-4;a=w.copy();b=w.copy();a[k]+=h;b[k]-=h
            finite=(loss(a)-loss(b))/(2*h);error=float(abs(finite-analytic[k])/(1+abs(finite)))
            checks.append(dict(test='shared_loss_gradient',method=method,column=k,error=error,passed=error<3e-5))
    # Physical synthetic targets test inversion/recovery only, not material fit.
    raw=s.read(s.BASE/'phase19_delivery/leedy2018_v2/samples.csv').sort_values('incorporated_cm3').iloc[[0,-1]].copy()
    raw['Tobs_K']=300.;ref,_=s.reference(s.FAMILIES);pars=np.r_[ref['parameters'],0.]
    model=s.mapper('inventory')
    def physical(z):
        a=[model.point(row,pars,source_kind='field',source_shift=z,require_observation=False) for row in raw.itertuples()]
        return np.array([p.values[0] for p in a]),np.array([p.source_jacobian[0] for p in a]),np.array([p.diagnostics['observation_domain_valid'] for p in a])
    coordinate=ObservationCoordinate(physical);target=physical(.2)[0]
    result=solve_effect(coordinate,target,[0.,1.],kind='quadratic')
    zero=result['solutions'][0];error=abs(zero['raw_shift']-.2)
    checks.append(dict(test='physical_synthetic_recovery',error=float(error),passed=bool(error<1e-5)))
    for effect in (-.3,0.,.3,1.):
        q=coordinate.evaluate(effect);error=abs(q['predictions'][0]-coordinate.base[0]-effect)
        checks.append(dict(test='first_anchor_coordinate',effect=effect,error=float(error),passed=bool(error<2e-6)))
    payload=dict(passed=bool(all(c['passed'] for c in checks)),checks=len(checks),rows=checks,
        maximum_error=max(c['error'] for c in checks),source_inverse_calls=coordinate.calls,
        code_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in (Path(__file__),Path(s.__file__),ROOT/'src/sandbox/observation_adaptation.py')},
        purpose='analytic loss/coordinate gates and synthetic consistency;not real source accuracy')
    s.energy.write(s.OUT/'learning_gates.json',payload);print(json.dumps(s.energy.clean(payload|{'rows':None}),ensure_ascii=False),flush=True)
    assert payload['passed']


if __name__=='__main__':main()
