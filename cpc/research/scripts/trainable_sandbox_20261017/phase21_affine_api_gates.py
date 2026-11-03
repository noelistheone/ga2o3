"""Check indexed empirical controls against an independent analytic formula."""
from pathlib import Path
import sys,json,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.affine_predictor import AffinePropertyPredictor
from sandbox.relative_predictor_v2 import SavedModel
import phase21_affine_controls as a
s=a.s;OUT=a.OUT


def main():
    bundle=OUT/'predictor_bundle';model=AffinePropertyPredictor(None,bundle);model.manifest=copy.deepcopy(model.manifest);checks=[]
    nodes=[dict(dopant='Si',incorporated_cm3=C,growth_T_C=550.) for C in (1e18,1e19,1e20)]
    for method,spec in model.manifest['variants'].items():
        for b in ('1','2'):spec['source_selection'][b]=dict(prior='quadratic',lam=0.)
        record=json.loads((bundle/'models'/f"{spec['model_key']}.json").read_text());base=SavedModel(record,profile=None,db=None)
        from sandbox.functional_predictor import DopedPropertyPredictor as Contract
        nr=[Contract.parse('Si',r) for r in nodes];n0=np.array([base.point(r).values[0] for r in nr]);x=np.array([0.,1.,2.])
        V=x[:,None] if spec['basis']=='slope' else np.c_[np.ones(3),x];G=V.T@V/3.;raw=np.array([.1]) if spec['basis']=='slope' else np.array([.1,.05])
        synthetic=n0+V@raw
        for budget in (0,1,2):
            indices=[1,2][:budget];anchors=[dict(input_index=i,measured_nH_cm3=float(10**synthetic[i])) for i in indices]
            p=model.predict('Si',1e18,growth_T_C=550.,method=method,source_inputs=nodes,anchors=anchors)
            if budget:
                # Independent minimum-function-norm constrained LS formula.
                A=V[indices];Gi=np.linalg.inv(G);r=synthetic[indices]-n0[indices]
                expected=Gi@A.T@np.linalg.pinv(A@Gi@A.T)@r
            else:expected=np.zeros(V.shape[1])
            error=abs(p['log10_nH_cm3']-float(n0[0]+V[0]@expected));checks.append(dict(method=method,budget=budget,error=error,passed=error<2e-6))
            if budget:
                q=model.predict('Si',1e19,observation_T_K=250.,growth_T_C=550.,method=method,source_inputs=nodes,anchors=anchors)
                error=float(np.max(abs(np.asarray(q['source_coefficients'])-p['source_coefficients'])))
                checks.append(dict(test='queryindependentcache',method=method,budget=budget,error=error,passed=error<1e-12 and q['calibration_cache_hit']))
    record=dict(passed=all(r['passed'] for r in checks),checks=len(checks),rows=checks,synthetic_labels_only=True,
        code_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in (Path(__file__),ROOT/'src/sandbox/affine_predictor.py',ROOT/'src/sandbox/affine_source.py')})
    s.energy.write(OUT/'api_gates.json',record);print('affine API',record['passed'],len(checks),flush=True);assert record['passed']


if __name__=='__main__':main()
