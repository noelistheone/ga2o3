"""Save the finite-difference scale diagnostic behind the retained failed gate."""
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import phase16_prediction as study
import phase2_energy as energy


def main():
    method='Thermal_original_RT_record_cap_g0';model=study.mapper(method)
    row=SimpleNamespace(element='Si',incorporated_cm3=1e18,growth_T_C=523.,Tobs_K=40.,
        H_known=False,H_cm3=0.,thermal_cap_K=796.15)
    w=np.array(study.core.specification(method)[2][0]);a=model.point(row,w);records=[]
    for j in (0,1):
        for h in (.16,.08,.02,.002,.0002,.00002,.000002):
            slopes=[]
            for step in (h,h/2):
                p=w.copy();n=w.copy();p[j]+=step;n[j]-=step
                ap=model.point(row,p);an=model.point(row,n)
                slopes.append((np.r_[ap.values,ap.transport_values]-np.r_[an.values,an.transport_values])/(2*step))
            exact=np.r_[a.jacobian[:,j],a.transport_jacobian[:,j]];rich=(4*slopes[1]-slopes[0])/3
            records.append(dict(parameter=j,h=h,analytic=exact,two_point=slopes[0],fourth_order=rich,
                two_point_scaled_error=float(max(abs(slopes[0]-exact)/(1+abs(slopes[0])))),
                fourth_order_scaled_error=float(max(abs(rich-exact)/(1+abs(rich))))))
    fresh=study.mapper(method).__class__(study.STRUCTURES[method],profile=model.profile,pO2_atm=1e-5)
    cold=fresh.point(row,w);perturb=w.copy();perturb[1]+=.5;fresh.point(row,perturb);warm=fresh.point(row,w)
    payload=dict(code_sha256=energy.sha(Path(__file__)),method=method,input=vars(row),parameters=w,
        base_values=a.values,base_transport=a.transport_values,root_residual=a.diagnostics['root_residual_log'],
        jacobian_condition=a.diagnostics['jacobian_condition'],free_n_cm3=a.diagnostics['free_n_cm3'],
        charge_squared_cm3=10**a.values[2],fresh_vs_warm_output_error=float(max(abs(np.r_[cold.values,cold.transport_values]-np.r_[warm.values,warm.transport_values]))),
        finite_difference_scale=records,
        interpretation='representative stronglycompensatedstate;smallh amplifiesfloatroundoff,Richardson removesleadingtruncation;not a material validation or universal highprecision errorbound',
        changed_physical_formula=False,changed_gradient_tolerance=False)
    energy.write(study.OUT/'numerical_precision_diagnostic.json',payload)
    print('Saved finite-difference scale diagnostic',payload['fresh_vs_warm_output_error'])


if __name__=='__main__':main()
