"""Exact Hall-moment rate-scaling identity; not a new material-accuracy test."""
from pathlib import Path
import sys,numpy as np
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

OUT=c.BASE/'proposal_followup_20261021_v1/hall_scaling_identity'


def main():
    assert not (OUT/'protocol_before.json').exists()
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes([Path(__file__)]),
        eta=[-8.,0.,8.],log10_rate_multipliers=[-3.,-1.,0.,1.,3.],quadrature=256,
        same_frozen_state=True,uniform_rate_multiplier_every_energy=True,new_model_fits=0,
        interpretation='exact moment identity;equivalent to common logmu/logsigma offset,not novel microscopic identification'))
    x,w=np.polynomial.legendre.leggauss(256);E=(x+1)*20;weight=w*20;records=[]
    for eta in [-8.,0.,8.]:
        f=expit(eta-E);prob=weight*E**1.5*f*(1-f);prob/=prob.sum()
        tau=1/(1+E+.01*E**2)
        t1=prob@tau;t2=prob@(tau*tau);rh=t2/t1**2;mu=t2/t1
        for d in [-3.,-1.,0.,1.,3.]:
            rate=10**d;tt=tau/rate;a=prob@tt;b=prob@(tt*tt)
            err=max(abs(np.log10(b/a**2)-np.log10(rh)),abs(np.log10(b/a)-np.log10(mu)+d))
            records.append(dict(eta=eta,log10_rate_multiplier=d,max_identity_error=err,passed=err<2e-14))
    # Uniform multiplication also cancels from effective laminate Hall density.
    sigma=np.array([.2,1.,7.]);rh_local=np.array([1.,2.,3.]);W=np.array([.2,.5,.3])
    A=W@sigma;B=W@(1/sigma);H=W@(sigma*rh_local);base=np.array([A/H,H/np.sqrt(A*B),np.sqrt(A/B)])
    for scale in [.1,10.]:
        ss=sigma/scale;aa=W@ss;bb=W@(1/ss);hh=W@(ss*rh_local)
        yy=np.array([aa/hh,hh/np.sqrt(aa*bb),np.sqrt(aa/bb)])
        err=float(max(abs(yy/base-np.array([1.,1/scale,1/scale]))))
        records.append(dict(laminate_uniform_multiplier=scale,max_identity_error=err,passed=err<2e-14))
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'checks.json',dict(created_utc=c.e.now(),passed=all(r['passed'] for r in records),checks=len(records),records=records,
        new_material_predictions=0,new_fits=0,no_claim_actual_rates_or_physical_regime_restored=True))
    print(dict(passed=all(r['passed'] for r in records),checks=len(records)))


if __name__=='__main__':main()
