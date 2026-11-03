"""Target the failed nominal-growth reference derivative without loosening gates."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase26_training as t
import phase2_energy as e


def main():
    data=t.TABLE;table=data.loc[[data[data.element.eq('Si')&data.H_known.eq(True)].index[0],
        data[data.element.eq('Sn')].index[0],data[data.sample_id.eq('10.1063/1.5047214::target-1')].index[0]]]
    rows=[]
    for method in ('GrowthReference','ShareGrowthConstant','ShareEffectiveConstant'):
        w=np.array([16.]) if method=='GrowthReference' else (np.array([16.,2.4]) if method=='ShareGrowthConstant' else np.array([.5,16.,2.4]))
        k=0 if method!='ShareEffectiveConstant' else 1
        m=t.model(method)
        for row in table.itertuples():
            mid=m.point(row,w,require_observation=False)
            for step in (.03,.01,.003,.001,.0003,.0001,.00003):
                d=np.zeros(len(w));d[k]=step
                plus=m.point(row,w+d,require_observation=False);minus=m.point(row,w-d,require_observation=False)
                fd=(plus.values[0]-minus.values[0])/(2*step);a=mid.jacobian[0,k]
                diag=mid.diagnostics
                rows.append(dict(method=method,sample_id=row.sample_id,step=step,log_nH=mid.values[0],
                    analytic=a,numeric=fd,scaled_error=abs(fd-a)/(1+abs(a)),
                    observation_valid=diag['observation_domain_valid'],root_residual=diag['root_residual_log'],
                    free_n=diag['free_n_cm3'],cold_slope=diag.get('cold_logcharge_EF_slope'),
                    root_condition=diag.get('jacobian_condition')))
    e.csvwrite(t.OUT/'gradient_diagnosis/background_steps.csv',rows)
    summary={}
    for method in ('GrowthReference','ShareGrowthConstant','ShareEffectiveConstant'):
        a=[r for r in rows if r['method']==method]
        summary[method]=dict(max_error_at_1e4=max(r['scaled_error'] for r in a if r['step']==.0001),
            best_step_max_error=min(max(r['scaled_error'] for r in a if r['step']==h) for h in (.03,.01,.003,.001,.0003,.0001,.00003)))
    e.write(t.OUT/'gradient_diagnosis/summary.json',dict(methods=summary,
        purpose='determine finite-difference precision versus analytic-chain defect;no gate or training objective changed',
        code_sha256=e.sha(Path(__file__)),new_fits=0,goal_complete=False))
    print(summary,flush=True)


if __name__=='__main__':main()
