"""Derivative and claim-boundary checks for the bounded closing experiment."""
from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import targeted_closure_model_v1 as m
import phase30_core as c


def main():
    out=m.OUT/'gates_v1';assert not (out/'protocol_before.json').exists()
    c.write(out/'protocol_before.json',dict(created_utc=c.e.now(),
        code_hashes=c.file_hashes([Path(__file__),Path(m.__file__)]),
        meaning='complete wrapped parameter/input derivatives plus algebraic counterexamples,not material prediction accuracy'))
    table=c.cases();selected=[]
    for element in ('Si','Sn'):
        q=table[table.element.eq(element)].sort_values('incorporated_cm3');selected.extend([q.iloc[0].case_id,q.iloc[len(q)//2].case_id,q.iloc[-1].case_id])
    positive=table[table.H_known.eq(True)]
    if len(positive):selected.append(positive.iloc[0].case_id)
    rows=table[table.case_id.isin(selected)];results=[]
    w=np.array([.5,14.5,2.,-3.,.2,.2]);h=2e-4
    for row in rows.itertuples():
        p=m.model('DiluteHinge').point(row,w);J=np.r_[p.jacobian[[0]],p.transport_jacobian]
        for j in range(6):
            d=np.zeros(6);d[j]=h
            a=m.point_values(m.model('DiluteHinge').point(row,w+d));b=m.point_values(m.model('DiluteHinge').point(row,w-d))
            err=float(max(abs((a-b)/(2*h)-J[:,j])/np.maximum(1.,abs(J[:,j]))))
            results.append(dict(kind='parameter',case_id=row.case_id,coordinate=j,error=err,passed=err<2e-5))
        for name in ('log10_dose_cm3','growth_T_C'):
            k=p.input_names.index(name)
            if name=='log10_dose_cm3':ra=row._replace(incorporated_cm3=row.incorporated_cm3*10**h);rb=row._replace(incorporated_cm3=row.incorporated_cm3/10**h)
            else:ra=row._replace(growth_T_C=row.growth_T_C+h);rb=row._replace(growth_T_C=row.growth_T_C-h)
            a=m.point_values(m.model('DiluteHinge').point(ra,w));b=m.point_values(m.model('DiluteHinge').point(rb,w))
            analytic=np.r_[p.input_jacobian[[0],k],p.transport_input_jacobian[:,k]]
            err=float(max(abs((a-b)/(2*h)-analytic)/np.maximum(1.,abs(analytic))))
            results.append(dict(kind='input',case_id=row.case_id,coordinate=name,error=err,passed=err<2e-5))
        poisoned=row._replace(target=row.target+99.,mu_cm2Vs=1e55)
        err=float(max(abs(m.point_values(m.model('DiluteHinge').point(poisoned,w))-m.point_values(p))))
        results.append(dict(kind='query_label_isolation',case_id=row.case_id,error=err,passed=err==0.))
    # The data-loss increase is 2 r^T W d + d^T W d, not drift RMS squared.
    r=np.array([1.,-1.]);d=np.array([-.5,.5]);W=np.array([.5,.5])
    delta=float(W@((r+d)**2-r*r));rms2=float(W@(d*d));identity=2*float(W@(r*d))+rms2
    results.append(dict(kind='loss_slack_is_not_drift_bound',delta_loss=delta,drift_rms=float(np.sqrt(rms2)),
        error=abs(delta-identity),passed=abs(delta-identity)<1e-15 and delta<.0001 and np.sqrt(rms2)>.01))
    lo,dlo=m.hinge(10.);hi,dhi=m.hinge(25.)
    results.append(dict(kind='hinge_derivative_limits',low_derivative=dlo,high_derivative=dhi,
        passed=dlo<1e-12 and abs(dhi-.5)<1e-10))
    # Six-parameter low-dose logit retains a free intercept, process and species effects.
    psi_low=2.+(-3.)*lo+.2
    results.append(dict(kind='hinge_low_limit_not_fixed_reference',low_logit=psi_low,passed=abs(psi_low)>1.))
    c.verify_hashes(c.document(out/'protocol_before.json'))
    j=dict(created_utc=c.e.now(),passed=all(r['passed'] for r in results),checks=len(results),
        maximum_parameter_or_input_FD_error=max(r.get('error',0.) for r in results),records=results)
    c.write(out/'gates.json',j);print({k:j[k] for k in ['passed','checks','maximum_parameter_or_input_FD_error']})


if __name__=='__main__':main()
