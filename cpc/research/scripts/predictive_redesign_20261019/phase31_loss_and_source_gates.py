"""Full penalized loss derivatives and label-free saved/source adapter checks."""
from pathlib import Path
from types import SimpleNamespace
import os,sys,subprocess
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase31_learning as l
from sandbox.correlated_saved_model import CorrelatedSavedModel
from sandbox.kroger_db import NPZ

OUT=l.OUT/'loss_source_gates_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    first=c.document(l.OUT/'gates_v1/gates.json')
    assert first['passed']
    for name,h in first['code_and_input_hashes'].items():assert c.e.sha(c.ROOT/name)==h,name
    code=[Path(__file__),Path(l.__file__),c.ROOT/'src/sandbox/correlated_saved_model.py',
        Path(__file__).with_name('phase31_source_runtime.py')]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),
        code_hashes=c.file_hashes(code),numerical_loss_scope='complete nH/rho and both response regularizers',
        selected_material_accuracy_not_tested=True,goal_complete=False))
    cases=c.cases()
    ids=[cases[cases.mobility_eligible.eq(True)&cases.element.eq(el)].sample_id.iloc[0] for el in ('Si','Sn')]
    rows=cases[cases.sample_id.isin(ids)].copy()
    w=np.array([.5,16.,.5,.1,.15,-.1,-1.,.2])
    checks=[];details=[]
    for method in l.METHODS:
        for config in (dict(gamma=0.,alpha=0.),dict(gamma=1.,alpha=1.)):
            r,J,_=l.loss_arrays(method,w,rows,config);gradient=2*J.T@r
            for j in range(len(w)):
                h=2e-5;step=np.eye(len(w))[j]*h
                rp=l.loss_arrays(method,w+step,rows,config)[0]
                rm=l.loss_arrays(method,w-step,rows,config)[0]
                fd=(rp-rm)/(2*h)
                error=float(np.max(abs(fd-J[:,j])/(1+abs(fd)+abs(J[:,j]))))
                fg=float((rp@rp-rm@rm)/(2*h))
                ge=abs(fg-gradient[j])/(1+abs(fg)+abs(gradient[j]))
                ok=bool(error<3e-6 and ge<3e-6);checks.append(ok)
                details.append(dict(method=method,config=config,column=j,
                    scaled_residual_jacobian_error=error,scaled_objective_gradient_error=ge,passed=ok))
    # A real new process sees only the minimal artificial fit, explicit inputs
    # and allowed Hall anchor. It must not read any result table.
    source_rows=cases[cases.family.eq(cases[cases.element.eq('Si')].family.iloc[0])].iloc[:3]
    nodes=[c.design.inputs(r) for r in source_rows.itertuples()]
    model_fit=dict(key='numerical_gate_not_material_fit',method='ChargeRatioSpectrum',
        parameters=w.tolist(),selected_success=True,training_valid=True)
    mapper=CorrelatedSavedModel(model_fit)
    unlabelled=[SimpleNamespace(**r) for r in nodes]
    expected=[mapper.point(r,source_kind='share',shift=0.,require_observation=False) for r in unlabelled]
    task=dict(id='new_spectral_saved_source_numerical_gate',fit=model_fit,kind='internal',nodes=nodes,
        episodes=[dict(budget=0,anchor_condition_ids=[],anchor_indices=[],anchor_labels=[]),
            dict(budget=1,anchor_condition_ids=['artificial'],anchor_indices=[0],
                anchor_labels=[expected[0].log_nH])])
    packet=OUT/'packet.json';dest=OUT/'runtime.json'
    c.write(packet,dict(tasks=[task],lambdas=[None,1.],physical_reference_sha256=c.e.sha(NPZ)))
    with (OUT/'runtime.log').open('w') as log:
        p=subprocess.run([sys.executable,str(Path(__file__).with_name('phase31_source_runtime.py')),
            str(packet),str(dest)],cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0,'Preserve adapter output before versioned repair'
    result=c.document(dest);checks.append(not result['denied_reads'])
    zero=result['records'][0]['episodes'][0]['solutions'][0]['records']
    deltas=[]
    for actual,reference in zip(zero,expected):
        delta=float(np.max(abs(np.array([actual['predicted'],actual['predicted_log_mu'],actual['predicted_log_sigma']])
            -np.array([reference.log_nH,reference.log_muH,reference.log_sigma]))))
        checks.append(delta<1e-10);deltas.append(delta)
    c.write(OUT/'gates.json',dict(passed=all(checks),created_utc=c.e.now(),
        checks=len(checks),loss_derivative_checks=len(details),details=details,
        max_scaled_residual_error=max(v['scaled_residual_jacobian_error'] for v in details),
        max_scaled_objective_error=max(v['scaled_objective_gradient_error'] for v in details),
        isolated_source_max_log_difference=max(deltas),denied_result_reads=result['denied_reads'],
        artificial_saved_parameters_not_material_training=True,
        code_hashes=c.file_hashes(code),goal_complete=False))
    print('Full-loss/source gates',len(checks),'failed',sum(not v for v in checks),flush=True)
    assert all(checks),'Preserve original gate failure before a versioned repair'


if __name__=='__main__':main()
