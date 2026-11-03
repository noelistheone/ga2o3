"""Actual fitted-parameter derivative check, without re-optimization or scores."""
from pathlib import Path
import sys,importlib,time
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

OUT=c.BASE/'rate_fitted_gradient_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    chosen=[]
    for phase,folder in [(31,'phase31_rate_redesign'),(32,'phase32_rate_controls')]:
        registry=importlib.import_module(f'phase{phase}_learning')
        files=sorted((c.BASE/folder/'fits').glob('*.json'))
        fits=[(p,c.document(p)) for p in files]
        for method in registry.METHODS:
            values=[(p,f) for p,f in fits if f['method']==method and len(f['train_units'])==3
                    and f['config']==dict(gamma=1.,alpha=1.) and f['parameters'] is not None]
            assert values,(phase,method)
            p,f=values[0];chosen.append(dict(phase=phase,path=str(p.relative_to(c.ROOT)),
                sha256=c.e.sha(p),fit_key=f['key'],method=method,reason='first hash-ordered joint regularized fitted training context'))
        failed=[(p,f) for p,f in fits if f['parameters'] is not None and not f['selected_success']]
        if failed:
            p,f=failed[0]
            chosen.append(dict(phase=phase,path=str(p.relative_to(c.ROOT)),sha256=c.e.sha(p),
                fit_key=f['key'],method=f['method'],reason='preserved minimum-loss nonstopped actual fit'))
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_sha256=c.e.sha(__file__),
        selected_fits=chosen,selection_does_not_use_outer_scores=True,
        finite_difference_step=2e-5,tolerance=3e-6,no_optimization=True,no_prediction_changes=True,
        stopping_or_global_optimality_not_certified=True,goal_complete=False))
    checks=[];details=[];clock=time.perf_counter();physical_calls=0
    for item in chosen:
        l=importlib.import_module(f'phase{item["phase"]}_learning');f=c.document(c.ROOT/item['path'])
        assert c.e.sha(c.ROOT/item['path'])==item['sha256']
        w=np.array(f['parameters'],float);rows=l.subset(f['train_units'])
        r,J,q=l.loss_arrays(f['method'],w,rows,f['config']);grad=2*J.T@r
        physical_calls+=q.get('actual_correlated_model_calls',q.get('actual_state_rate_model_calls',0))
        for j in range(len(w)):
            step=np.eye(len(w))[j]*2e-5
            rp,_,qp=l.loss_arrays(f['method'],w+step,rows,f['config'])
            rm,_,qm=l.loss_arrays(f['method'],w-step,rows,f['config'])
            physical_calls+=sum(v.get('actual_correlated_model_calls',v.get('actual_state_rate_model_calls',0)) for v in [qp,qm])
            fd=(rp-rm)/(4e-5);fg=(rp@rp-rm@rm)/(4e-5)
            je=float(np.max(abs(fd-J[:,j])/(1+abs(fd)+abs(J[:,j]))))
            ge=float(abs(fg-grad[j])/(1+abs(fg)+abs(grad[j])))
            good=bool(np.isfinite([je,ge]).all() and je<3e-6 and ge<3e-6)
            checks.append(good);details.append(dict(phase=item['phase'],fit_key=f['key'],method=f['method'],
                selected_success=f['selected_success'],column=j,scaled_residual_jacobian_error=je,
                scaled_objective_gradient_error=ge,analytic_gradient=float(grad[j]),passed=good))
    c.write(OUT/'audit.json',dict(passed=all(checks),created_utc=c.e.now(),checks=len(checks),
        actual_saved_fits=len(chosen),actual_correlated_or_state_rate_point_calls=physical_calls,
        maximum_scaled_residual_error=max(r['scaled_residual_jacobian_error'] for r in details),
        maximum_scaled_objective_error=max(r['scaled_objective_gradient_error'] for r in details),details=details,
        preserved_nonstopped_not_requalified=True,no_new_training_or_material_accuracy=True,
        seconds=time.perf_counter()-clock,goal_complete=False))
    print('Actual fitted gradient checks',len(checks),'fits',len(chosen),'failed',sum(not q for q in checks),flush=True)
    assert all(checks),'Preserve failed fitted-gradient record;repair by version'


if __name__=='__main__':main()
