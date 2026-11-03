"""Check actual saved fit gradients; no optimization or candidate modification."""
from pathlib import Path
import sys,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase29_data as data
import phase29_learning as l
import phase2_energy as e

OUT=data.OUT/'fitted_gradient_audit_v1'


def main():
    assert not (OUT/'protocol_before.json').exists(),'preserve original fitted-state audit before any versioned repair'
    failed=data.OUT/'fits/1f516a5bbd07e8f015326c12.json'
    first=json.loads(failed.read_text());units=first['train_units'];targets=[
        ('ShareConstant205',1.,0.),('ReferenceTau',1.,1.),
        ('JointConstant',1.,0.),('JointProcess',1.,1.)]
    fits=[];paths=[]
    for method,gamma,alpha in targets:
        matches=[]
        for path in (data.OUT/'fits').glob('*.json'):
            try:fit=json.loads(path.read_text())
            except json.JSONDecodeError:continue
            if fit['method']==method and fit['train_units']==units and fit['config']==dict(gamma=gamma,alpha=alpha):
                matches.append((path,fit))
        assert len(matches)==1,(method,gamma,alpha)
        path,fit=matches[0];fits.append(fit);paths.append(path)
    assert fits[0]['key']==first['key'] and not first['selected_success']
    code=[Path(__file__),Path(__file__).with_name('phase29_learning.py')]
    e.write(OUT/'protocol_before.json',dict(created_utc=e.now(),targets=targets,
        selection_scope='first nonstopped fit plus predeclared physical controls in identical training context;no outer labels',
        Richardson_steps=[.005,.0025,.00125],scaled_tolerance=3e-6,
        loss_replay_tolerance=3e-8,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths},
        no_fitting_refinement_or_prediction_change=True,goal_complete=False))
    cases=data.load_cases();table=cases[cases.related_cluster.isin(units)]
    rows=[];checks=[];calls=0;clock=time.perf_counter()
    for fit in fits:
        w=np.asarray(fit['parameters'],float);method=fit['method'];config=fit['config']
        r,J,q=l.loss_arrays(method,w,table,config);calls+=1
        gradient=2*J.T@r
        error=abs(float(r@r)-fit['selected_loss'])/(1+abs(fit['selected_loss']))
        checks.append(dict(name='saved_training_loss_replay',passed=error<=3e-8,error=error,fit_key=fit['key']))
        for j in range(len(w)):
            for h in (.005,.0025,.00125):
                derivatives=[];loss_derivatives=[]
                for step in (h,h/2):
                    plus=w.copy();minus=w.copy();plus[j]+=step;minus[j]-=step
                    rp,_,_=l.loss_arrays(method,plus,table,config)
                    rm,_,_=l.loss_arrays(method,minus,table,config);calls+=2
                    derivatives.append((rp-rm)/(2*step))
                    loss_derivatives.append((float(rp@rp)-float(rm@rm))/(2*step))
                numeric=(4*derivatives[1]-derivatives[0])/3
                numeric_loss=(4*loss_derivatives[1]-loss_derivatives[0])/3
                jerror=float(np.max(abs(numeric-J[:,j])/(1+abs(J[:,j]))))
                gerror=float(abs(numeric_loss-gradient[j])/(1+abs(gradient[j])))
                rows.append(dict(fit_key=fit['key'],method=method,selected_stopped=fit['selected_success'],
                    parameter_index=j,h=h,residual_jacobian_error=jerror,objective_gradient_error=gerror,
                    analytic_objective_gradient=float(gradient[j]),numeric_objective_gradient=float(numeric_loss)))
        # Use the finest predetermined step for the gate; retain the whole
        # convergence sequence, with no tolerance selection after the fact.
        own=[v for v in rows if v['fit_key']==fit['key'] and v['h']==.00125]
        for result in own:
            checks.append(dict(name='actual_saved_residual_Jacobian',passed=result['residual_jacobian_error']<=3e-6,
                error=result['residual_jacobian_error'],fit_key=fit['key'],parameter=result['parameter_index']))
            checks.append(dict(name='actual_saved_objective_gradient',passed=result['objective_gradient_error']<=3e-6,
                error=result['objective_gradient_error'],fit_key=fit['key'],parameter=result['parameter_index']))
        print('Fitted gradient29',method,'original stopped',fit['selected_success'],flush=True)
    e.csvwrite(OUT/'step_convergence.csv',rows)
    result=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,
        saved_fits=len(fits),diagnostic_objective_evaluations=calls,seconds=time.perf_counter()-clock,
        no_new_optimizer_starts=True,no_source_or_test_label_selection=True,
        gradient_consistency_does_not_prove_global_optimum_or_material_accuracy=True,goal_complete=False)
    e.write(OUT/'audit.json',result)
    print('Fitted gradient audit',result['passed'],len(checks),'seconds',result['seconds'],flush=True)
    if not result['passed']:raise RuntimeError('Preserve actual-state gradient failure before a versioned response')


if __name__=='__main__':main()
