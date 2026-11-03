"""Independent Bayesian odds and truncated-normal checks of finite-state inference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.stats import norm,truncnorm
from sandbox.source_ensemble import SourceEnsemble,source_grid
import phase2_energy as energy
OUT=energy.BASE/'phase22_predictive_inference'


def main():
    checks=[]
    def check(name,error,tolerance):checks.append(dict(test=name,error=float(error),tolerance=tolerance,passed=bool(error<tolerance)))
    u=np.array([-1.,0.,1.]);base=np.array([.25,.5,.25]);g=np.c_[18.+u,18.+2*u];ens=SourceEnsemble(u,base,g,np.ones(3,bool))
    for tau in (.1,.5,2.):
        for sigma in (.05,.2,1.):
            for rho in (0.,.5,.9,1.):
                prior=base*np.exp(-np.mean((g-g[1])**2,axis=1)/(2*tau*tau));prior/=prior.sum();prior*=1-rho;prior[1]+=rho
                likelihood=norm.pdf(18.4,loc=g[:,0],scale=sigma);manual=prior*likelihood;manual/=manual.sum()
                p=ens.posterior([0],[18.4],tau=tau,sigma=sigma,rho=rho)
                check('independent_probability_odds',np.max(abs(p.weights-manual)),2e-12)
                zero=ens.posterior([],[],tau=tau,sigma=sigma,rho=rho);check('no_anchor_recovers_prior',np.max(abs(zero.weights-prior)),2e-12)
    # A continuous truncated Gaussian provides an independent refinement target.
    uu=np.linspace(-2.,2.,4097);bw=np.ones(len(uu));bw[[0,-1]]=.5;gg=(18.+uu)[:,None]
    E=SourceEnsemble(uu,bw,gg,np.ones(len(uu),bool));tau=.35;sig=.2;y=18.55
    v=1/(1/tau**2+1/sig**2);mu=v*(y-18)/sig**2;sd=np.sqrt(v);a=(-2-mu)/sd;b=(2-mu)/sd
    target=truncnorm(a,b,loc=18+mu,scale=sd);p=E.posterior([0],[y],tau=tau,sigma=sig);z=E.summarize(p,uncertainty=True)
    check('independent_truncated_mean',abs(z['mean'][0]-target.mean()),1e-8)
    check('independent_truncated_variance',abs(z['variance'][0]-target.var()),1e-8)
    check('independent_latent_median',abs(z['latent_median'][0]-target.median()),.0006)
    for rho in (.25,.75):
        p=E.posterior([0],[y],tau=tau,sigma=sig,rho=rho)
        # Integral of a product of two normal densities, with prior truncation.
        slab=norm.pdf(y,loc=18,scale=np.sqrt(tau*tau+sig*sig))*(norm.cdf(b)-norm.cdf(a))/(norm.cdf(2/tau)-norm.cdf(-2/tau))
        atom=norm.pdf(y,loc=18,scale=sig);truth=rho*atom/(rho*atom+(1-rho)*slab)
        check('normalized_spike_evidence',abs(p.spike_posterior_probability-truth),1e-8)
    shift=SourceEnsemble(uu,bw,gg+4,np.ones(len(uu),bool));p1=E.posterior([0],[y],tau=tau,sigma=sig,rho=.5)
    p2=shift.posterior([0],[y+4],tau=tau,sigma=sig,rho=.5);check('common_log_translation',max(abs(p1.weights-p2.weights)),1e-12)
    perm=np.arange(len(uu))[::-1];per=SourceEnsemble(uu[perm],bw[perm]*13.,gg[perm],np.ones(len(uu),bool))
    p3=per.posterior([0],[y],tau=tau,sigma=sig,rho=.5);check('grid_permutation_and_base_normalization',max(abs(p1.weights-p3.weights[::-1])),1e-12)
    for n in (33,65,129):
        x,w=source_grid(n);check('grid_probability_measure',abs(w.sum()-1),1e-12)
        check('nested_exact_zero',abs(x[n//2]),1e-12)
        assert (w>0).all()
    r=dict(passed=all(c['passed'] for c in checks),checks=len(checks),maximum_error=max(c['error'] for c in checks),rows=checks,
        scope='finite-state normalized working model; not material accuracy or calibrated uncertainty',
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (Path(__file__),ROOT/'src/sandbox/source_ensemble.py')})
    energy.write(OUT/'ensemble_gates.json',r);print('ensemble gates',r['passed'],len(checks),flush=True);assert r['passed'],[c for c in checks if not c['passed']]


if __name__=='__main__':main()
