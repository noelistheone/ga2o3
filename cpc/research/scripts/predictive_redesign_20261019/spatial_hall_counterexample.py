"""Synthetic operator counterexample, not a trained material predictor."""
from pathlib import Path
import sys,numpy as np
from scipy.special import i0
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

OUT=c.BASE/'spatial_hall_counterexample_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_sha256=c.e.sha(__file__),
        theory_sha256=c.e.sha(c.ROOT/'docs/ML_SANDBOX_SPATIAL_CLOSURE_RESEARCH_20261020.md'),
        synthetic_only=True,no_KROGER_Poisson_integration=True,no_model_training=True,
        no_real_accuracy_or_microstructure_recovery=True,goal_complete=False))
    x=(np.arange(32768)+.5)/32768;mu0=205.;nbar=1e18;e=c.learning.E_CHARGE
    rows=[];checks=[]
    for a in [0.,2.,5.,8.,12.]:
        n=nbar*np.exp(a*np.cos(2*np.pi*x))/i0(a)
        sigma=e*n*mu0;rho=1/sigma;RH=1/(e*n)
        sx=1/rho.mean();sy=sigma.mean();reff=float((sigma*RH).mean()/sigma.mean())
        sgeo=np.sqrt(sx*sy);nh=1/(e*reff);mh=sgeo*reff
        expected=mu0/i0(a)
        checks.extend([abs(n.mean()/nbar-1)<1e-12,abs(nh/nbar-1)<1e-12,
                       abs(mh/expected-1)<1e-12,abs(sgeo/(e*nh*mh)-1)<1e-12])
        # Direct local boundary equations with Jx=1,<Jy>=0.
        B=1e-4;Ey=reff*B
        Jy=sigma*(Ey-RH*B)
        Ex=rho-RH*B*Jy
        mr=(sigma*RH**2).mean()-(sigma*RH).mean()**2/sigma.mean()
        expected_Ex=rho.mean()+B**2*mr
        checks.extend([abs(Jy.mean())/(1+abs(Jy).max())<1e-12,
                       abs(Ex.mean()/expected_Ex-1)<1e-12,mr>=-1e-12])
        rows.append(dict(a=a,local_mu_constant_cm2Vs=mu0,mean_n_cm3=float(n.mean()),
            local_n_min_cm3=float(n.min()),local_n_max_cm3=float(n.max()),
            measured_effective_nH_cm3=nh,measured_effective_muH_cm2Vs=mh,
            exact_mu0_over_I0=expected,conductivity_geo_S_cm=sgeo,
            exact_local_periodic_charge_source_not_implemented=True,synthetic_not_material_validation=True))
    c.e.csvwrite(OUT/'synthetic_cases.csv',rows)
    c.write(OUT/'audit.json',dict(passed=all(checks),checks=len(checks),synthetic_cases=len(rows),
        physical_samples=0,training_fits=0,all_local_mobility_equal205=True,
        conclusion='effective small Hall mobility does not uniquely determine every local mobility',
        KROGER_Poisson_and_real_predictive_validation_still_required=True,goal_complete=False))
    print('Synthetic spatial Hall counterexample',len(checks),'checks;failed',sum(not v for v in checks),flush=True)
    assert all(checks),'Preserve synthetic counterexample failure before versioned repair'


if __name__=='__main__':main()
