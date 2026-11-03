"""Separate condition-mean response from assumed scenario-variance constraints."""
from pathlib import Path
import sys,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase29_learning as l
import phase29_data as data
import phase2_energy as e

OUT=data.OUT/'information_audit'


def decompose(table,values,J,targets,weights):
    pieces=[];collapsed=[];variation=[];mean_loss=0.;var_loss=0.
    for sid in table.sample_id.unique():
        mask=table.sample_id.eq(sid).to_numpy(bool);total=float(weights[mask].sum())
        if total<=0:continue
        v=weights[mask]/total;x=values[mask];derivative=J[mask]
        average=float(v@x);averageJ=v@derivative
        target=targets[mask]
        if np.max(abs(target-target[0]))>1e-13:raise ValueError('One physical condition must retain one label across scenarios')
        centered=x-average
        mean_loss+=total*(average-target[0])**2;var_loss+=total*float(v@(centered*centered))
        pieces.append(np.sqrt(weights[mask])[:,None]*derivative)
        collapsed.append(np.sqrt(total)*averageJ)
        variation.append(np.sqrt(weights[mask])[:,None]*(derivative-averageJ))
    full=np.concatenate(pieces);collapsed=np.asarray(collapsed);variation=np.concatenate(variation)
    return dict(expanded=full,condition_mean=collapsed,variation=variation,
        mean_loss=float(mean_loss),scenario_variation_loss=float(var_loss))


def main():
    assert not (OUT/'protocol_before.json').exists(),'preserve prior information audit before versioned repair'
    assert json.loads((data.OUT/'training_audit.json').read_text())['passed']
    plan=json.loads((data.OUT/'prediction/plan.json').read_text())['plans']
    e.write(OUT/'protocol_before.json',dict(created_utc=e.now(),code_sha256=e.sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (
            data.OUT/'training_protocol.json',data.OUT/'prediction/plan.json',data.OUT/'data/scenarios49.csv')},
        decomposition_tolerance=2e-12,no_new_fitting=True,no_real_noise_Fisher_claim=True,goal_complete=False))
    cases=data.load_cases();records=[];checks=[];seen=set()
    for p in plan:
        if p['fit_key'] in seen or p['spec']['method'] not in l.NONLINEAR:continue
        seen.add(p['fit_key']);fit=json.loads((ROOT/p['fit_path']).read_text())
        if not fit['training_valid'] or not fit['selected_success']:
            records.append(dict(fit_key=fit['key'],qualified=False,reason='training status',method=fit['method']));continue
        table=cases[cases.related_cluster.isin(fit['train_units'])]
        q=l.forward(fit['method'],np.asarray(fit['parameters']),table)
        v=q['values'];J=q['jacobian'];wn=l.weights(table);wp=l.weights(table,paired_only=True)
        paired,rho=l.observed_pair(table);gamma=float(fit['config']['gamma'])
        parts=[decompose(table,v[:,0],J[:,0],table.target.to_numpy(float),wn)]
        expanded_loss=float(wn@((v[:,0]-table.target.to_numpy(float))**2))
        if gamma>0:
            parts.append(decompose(table.loc[paired],-v[paired,2],-J[paired,2],rho[paired],gamma*wp[paired]))
            expanded_loss+=float(gamma*wp[paired]@((-v[paired,2]-rho[paired])**2))
        A=np.concatenate([z['expanded'] for z in parts])
        B=np.concatenate([z['condition_mean'] for z in parts])
        V=np.concatenate([z['variation'] for z in parts])
        gram=A.T@A;mean_gram=B.T@B;variance_gram=V.T@V
        error=float(np.max(abs(gram-mean_gram-variance_gram)/(1+abs(gram))))
        mean_loss=sum(z['mean_loss'] for z in parts);var_loss=sum(z['scenario_variation_loss'] for z in parts)
        loss_error=abs(expanded_loss-mean_loss-var_loss)/(1+abs(expanded_loss))
        checks.append(dict(name='GaussNewton_Gram_decomposition',passed=error<=2e-12,error=error,fit_key=fit['key']))
        checks.append(dict(name='scenario_loss_bias_variance_decomposition',passed=loss_error<=2e-12,error=loss_error,fit_key=fit['key']))
        records.append(dict(fit_key=fit['key'],method=fit['method'],train_units=fit['train_units'],qualified=True,
            config=fit['config'],reported_conditions=table.sample_id.nunique(),scenario_rows=len(table),
            expanded_surrogate_residual_rank=int(np.linalg.matrix_rank(A)),
            condition_mean_response_rank=int(np.linalg.matrix_rank(B)),
            scenario_variation_rank=int(np.linalg.matrix_rank(V)),
            expanded_surrogate_singular_values=np.linalg.svd(A,compute_uv=False).tolist(),
            condition_mean_singular_values=np.linalg.svd(B,compute_uv=False).tolist(),
            condition_mean_squared_loss=mean_loss,scenario_variation_squared_loss=var_loss,
            scenario_variation_Gram_trace_fraction=float(np.trace(variance_gram)/np.trace(gram)) if np.trace(gram)>0 else None,
            expanded_rank_is_not_measured_information_rank=True,condition_mean_rank_is_not_an_identifiability_proof=True,
            actual_experimental_covariance_available=False))
    e.write(OUT/'fits.json',records)
    result=dict(passed=bool(checks and all(c['passed'] for c in checks)),created_utc=e.now(),
        checks=len(checks),rows=checks,selected_unique_nonlinear_fits=len(records),
        scope='actual chosen full/excluded shared fits;expanded surrogate versus condition means,not noise-calibrated Fisher information',
        historical_training_observation_jacobian_rank_requires_surrogate_interpretation=True,
        no_training_models_scores_or_queries_changed=True,goal_complete=False)
    e.write(OUT/'audit.json',result)
    print('Information29 audit',result['passed'],'fits',len(records),flush=True)
    if not result['passed']:raise RuntimeError('Preserve information decomposition failure')


if __name__=='__main__':main()
