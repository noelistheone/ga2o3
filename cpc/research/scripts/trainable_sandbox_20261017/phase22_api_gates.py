"""Check deployment decisions with direct probability arithmetic, synthetic anchors."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.special import logsumexp
from sandbox.ensemble_predictor import FiniteEnsemblePredictor
from sandbox.relative_predictor_v2 import SavedModel
from sandbox.functional_predictor import DopedPropertyPredictor as Parse
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.kroger_db import load
import phase22_export as ex
p=ex.p;OUT=p.OUT/'api_gates'


def main():
    plans,_=p.plans_and_jobs();choices=[]
    for plan in plans:
        if plan['held'] is not None:continue
        for b in (0,1,2):
            choice=dict(p.DEFAULT) if b==0 else dict(tau=.5,sigma=.15,rho=.5 if plan['allow_spike'] else 0.)
            choices.append(dict(variant=plan['variant'],fit_key=plan['fit_key'],fit_path=plan['fit_path'],kind=plan['kind'],reg=plan['reg'],budget=b,selected_source=dict(config=choice)))
    bundle=OUT/'synthetic_bundle';manifest=ex.build(bundle,choices,grade='syntheticAPIcheck;source hyperparameters not selected production')
    db=load();profile=published_coherent_profile();api=FiniteEnsemblePredictor(bundle,db=db);checks=[]
    nodes=[dict(incorporated_cm3=c,growth_T_C=550.) for c in (1e18,1e19,1e20)];rows=[Parse.parse('Si',n) for n in nodes]
    u=6*np.linspace(-1,1,129)**3;edge=np.r_[-6,(u[1:]+u[:-1])/2,6];base=np.diff(edge)/12.;zero=64
    bank={}
    for method,spec in manifest['variants'].items():
        k=(spec['model_key'],spec['source_kind'])
        if k not in bank:
            rec=json.loads((bundle/'models'/f"{spec['model_key']}.json").read_text());m=SavedModel(rec,profile=profile,db=db)
            values=np.full((129,3,3),np.nan);valid=np.zeros((129,3),bool)
            for i,z in enumerate(u):
                for j,r in enumerate(rows):
                    try:v=m.point(r,source_kind=spec['source_kind'],shift=float(z),require_observation=False);values[i,j]=[v.values[0],*v.transport_values];valid[i,j]=v.diagnostics['observation_domain_valid']
                    except (ValueError,RuntimeError,np.linalg.LinAlgError):pass
            bank[k]=(values,valid)
        values,valid=bank[k];y=values[zero,:2,0]+np.array([.22,-.08]);Q=np.mean((values[:,:,0]-values[zero,:,0])**2,axis=1)
        for budget in (0,1,2):
            anchors=[dict(input_index=i,measured_nH_cm3=float(10**y[i])) for i in range(budget)]
            got=api.predict('Si',1e20,method=method,growth_T_C=550.,source_inputs=nodes,anchors=anchors)
            if budget:
                cfg=spec['source_selection'][str(budget)];good=np.all(valid,axis=1)&np.isfinite(Q)&(Q<=36)
                lp=np.full(129,-np.inf);lp[good]=np.log(base[good])-Q[good]/(2*cfg['tau']**2);lp-=logsumexp(lp)
                prior=(1-cfg['rho'])*np.exp(lp);prior[zero]+=cfg['rho'];ok=prior>0
                lp[:]=-np.inf;lp[ok]=np.log(prior[ok])-np.sum((values[ok,:budget,0]-y[:budget])**2,axis=1)/(2*cfg['sigma']**2)
                weights=np.exp(lp-logsumexp(lp));order=np.argsort(np.nan_to_num(values[:,2,0],nan=np.inf),kind='stable');cum=np.cumsum(weights[order])
                idx=int(order[np.searchsorted(cum,.5,side='left')])
            else:idx=zero
            truth=values[idx,2];error=float(np.max(abs(truth-[got['log10_nH_cm3'],got['log10_muH_cm2Vs'],got['log10_sigma_S_cm']])))
            checks.append(dict(test='independent_finite_probability_decision',method=method,budget=budget,error=error,passed=error<2e-7))
            if budget:
                perm=[2,0,1];a2=[dict(input_index=perm.index(a['input_index']),measured_nH_cm3=a['measured_nH_cm3']) for a in anchors]
                again=api.predict('Si',1e20,method=method,growth_T_C=550.,source_inputs=[nodes[i] for i in perm],anchors=a2)
                error=abs(again['log10_nH_cm3']-got['log10_nH_cm3']);checks.append(dict(test='source_order_and_cache',method=method,budget=budget,error=error,passed=error<1e-12 and again['calibration_cache_hit']))
    r=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,synthetic_anchors_only=True,
        scope='same-state deployment probability decisions; not material validation',
        code_hashes={str(f.relative_to(ROOT)):p.s.energy.sha(f) for f in (Path(__file__),ROOT/'src/sandbox/ensemble_predictor.py')})
    p.s.energy.write(OUT/'verification.json',r);print('ensemble API gates',r['passed'],len(checks),flush=True);assert r['passed'],[c for c in checks if not c['passed']]


if __name__=='__main__':main()
