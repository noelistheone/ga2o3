"""Independent moments, finite-difference state chains, loss and isolation."""
from pathlib import Path
from types import SimpleNamespace
import sys,subprocess,time
import numpy as np
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase32_learning as l
from sandbox.state_rate_transport import positive_state_rate_transport
from sandbox.state_rate_saved import StateRateSavedModel
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.coherent_transport import fermi_transport
from sandbox.ionized_transport import E_CHARGE,EPSILON0,K_B_J,HBAR,M_E
from sandbox.kroger_db import NPZ

OUT=l.OUT/'gates_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    code=[Path(__file__),Path(l.__file__),Path(__file__).with_name('phase32_source_runtime.py')]
    code+=list((c.ROOT/'src/sandbox').glob('*.py'))
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),
        code_hashes=c.file_hashes(code),training_not_yet_started=True,material_accuracy=False))
    p=published_coherent_profile();kw=dict(Nc_ref_cm3=p.Nc_ref,T_ref_K=p.T_ref,mass_ratio=p.electron_mass_ratio)
    mass=p.electron_mass_ratio*M_E;eps=10.2*EPSILON0
    nodes,weights=np.polynomial.legendre.leggauss(512)
    checks=[];details=[];clock=time.perf_counter()
    def check(kind,error,tolerance=3e-6):
        ok=bool(np.isfinite(error) and error<tolerance);checks.append(ok)
        details.append(dict(kind=kind,error=float(error),tolerance=tolerance,passed=ok))
    for logn,logq,T in [(14.,16.,250.),(18.,19.,300.),(20.,20.5,400.)]:
        for psi in [None,-10.,-1.,3.,30.,180.]:
            tr=positive_state_rate_transport(logn,logq,T,psi=psi,**kw)
            eta=tr.diagnostics['eta'];kT=K_B_J*T
            lo=np.sqrt(max(0.,eta-55.));hi=np.sqrt(max(65.,eta+55.))
            t=(nodes+1)*(hi-lo)/2+lo;x=t*t
            prob=weights*2*t**4*expit(eta-x)*expit(x-eta);prob/=prob.sum()
            qs2=E_CHARGE**2*tr.diagnostics['n_screen_cm3']*1e6/(eps*kT)
            b=8*mass*kT*x/(HBAR**2*qs2)
            angular=np.log1p(b)-b/(1+b)
            ion=(10.**logq*1e6*E_CHARGE**4/(16*np.sqrt(2)*np.pi*eps**2*np.sqrt(mass)*(kT*x)**1.5))*angular
            factor=1. if psi is None else 1+10.**psi
            extra=E_CHARGE/(mass*205.*1e-4)*factor
            rate=ion+extra;scale=float(np.min(rate));tau=scale/rate
            t1=float(prob@tau);t2=float(prob@(tau*tau));rh=t2/t1**2
            mudrift=E_CHARGE/mass*(t1/scale)*1e4
            expected=np.array([logn-np.log10(rh),np.log10(mudrift*rh),np.log10(E_CHARGE)+logn+np.log10(mudrift)])
            check('independent512moments',float(np.max(abs(tr.values-expected))),2e-7)
            check('positive_Hall_identity',abs(tr.values[2]-(np.log10(E_CHARGE)+tr.values[0]+tr.values[1])),1e-12)
            checks.append(rh>=1-1e-12)
            if psi is None:
                base=fermi_transport(logn,logq,T,extra_mobility_cm2Vs=205.,**kw)
                check('exact_baseline_values',float(np.max(abs(tr.values-base.values))),1e-12)
                check('exact_baseline_jac',float(np.max(abs(tr.jacobian[:,:3]-base.jacobian))),1e-12)
            else:
                args=np.array([logn,logq,np.log10(T),psi])
                def f(a):return positive_state_rate_transport(a[0],a[1],10**a[2],psi=a[3],**kw).values
                for j in range(4):
                    d=np.eye(4)[j]*2e-5;fd=(f(args+d)-f(args-d))/(4e-5)
                    check('transport_partial',float(np.max(abs(fd-tr.jacobian[:,j])/(1+abs(fd)+abs(tr.jacobian[:,j])))))
    cases=c.cases();ids=[cases[cases.mobility_eligible.eq(True)&cases.element.eq(el)].sample_id.iloc[0] for el in ('Si','Sn')]
    rows=cases[cases.sample_id.isin(ids)].copy()
    hcase=cases[cases.H_known.eq(True)].iloc[0]
    source_tasks=[];expected={}
    for method in l.METHODS:
        w=np.array([.5,16.,.5,.1,.15,-.1,-1.] + ([] if l.METHODS[method]=='global' else [.7]))
        for row in [rows.iloc[0],rows[rows.element.eq('Sn')].iloc[0],hcase]:
            row=SimpleNamespace(**row.to_dict());mapper=l.model(method)
            p0=mapper.point(row,w,source_kind='share',source_shift=.2,require_observation=False)
            values=lambda q:np.r_[q.values[0],q.transport_values]
            analytic=np.r_[p0.jacobian[[0]],p0.transport_jacobian]
            for j in range(len(w)):
                d=np.eye(len(w))[j]*2e-5
                fd=(values(mapper.point(row,w+d,source_kind='share',source_shift=.2,require_observation=False))-values(mapper.point(row,w-d,source_kind='share',source_shift=.2,require_observation=False)))/(4e-5)
                check('full_parameter_chain',float(np.max(abs(fd-analytic[:,j])/(1+abs(fd)+abs(analytic[:,j])))))
            fd=(values(mapper.point(row,w,source_kind='share',source_shift=.2+2e-5,require_observation=False))-values(mapper.point(row,w,source_kind='share',source_shift=.2-2e-5,require_observation=False)))/(4e-5)
            dj=np.r_[p0.source_jacobian[0],p0.transport_source_jacobian]
            check('source_chain',float(np.max(abs(fd-dj)/(1+abs(fd)+abs(dj)))))
            for j,name in enumerate(p0.input_names):
                attr={'log10_dose_cm3':'incorporated_cm3','log10_Tobs_K':'Tobs_K','log10_H_cm3':'H_cm3','growth_T_C':'growth_T_C'}[name]
                h=2e-5 if name.startswith('log10') else .002
                def changed(sign):
                    data=vars(row).copy();data[attr]=data[attr]*10**(sign*h) if name.startswith('log10') else data[attr]+sign*h
                    return values(mapper.point(SimpleNamespace(**data),w,source_kind='share',source_shift=.2,require_observation=False))
                fd=(changed(1)-changed(-1))/(2*h)
                dj=np.r_[p0.input_jacobian[0,j],p0.transport_input_jacobian[:,j]]
                check('input_chain',float(np.max(abs(fd-dj)/(1+abs(fd)+abs(dj)))))
        for config in (dict(gamma=0.,alpha=0.),dict(gamma=1.,alpha=1.)):
            r,J,_=l.loss_arrays(method,w,rows,config);grad=2*J.T@r
            for j in range(len(w)):
                d=np.eye(len(w))[j]*2e-5
                rp=l.loss_arrays(method,w+d,rows,config)[0];rm=l.loss_arrays(method,w-d,rows,config)[0]
                fd=(rp-rm)/(4e-5);fg=(rp@rp-rm@rm)/(4e-5)
                check('full_loss_J',float(np.max(abs(fd-J[:,j])/(1+abs(fd)+abs(J[:,j])))))
                check('full_objective_gradient',abs(fg-grad[j])/(1+abs(fg)+abs(grad[j])))
        f=dict(key='numerical_gate_'+method,method=method,parameters=w.tolist(),selected_success=True,training_valid=True)
        ns=[c.design.inputs(r) for r in rows.iloc[:2].itertuples()]
        saved=StateRateSavedModel(f);ev=[saved.point(SimpleNamespace(**r),require_observation=False) for r in ns]
        task=dict(id=method,fit=f,kind='internal',nodes=ns,episodes=[dict(budget=0,anchor_condition_ids=[],anchor_indices=[],anchor_labels=[]),dict(budget=1,anchor_condition_ids=['artificial'],anchor_indices=[0],anchor_labels=[ev[0].log_nH])])
        source_tasks.append(task);expected[method]=ev
    packet=OUT/'packet.json';target=OUT/'runtime.json'
    c.write(packet,dict(tasks=source_tasks,lambdas=[None,1.],physical_reference_sha256=c.e.sha(NPZ)))
    with (OUT/'runtime.log').open('w') as stream:
        status=subprocess.run([sys.executable,str(Path(__file__).with_name('phase32_source_runtime.py')),str(packet),str(target)],cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
    assert status==0,'Preserve runtime failure before versioned repair'
    result=c.document(target);checks.append(result['denied_reads']==[])
    for record in result['records']:
        for actual,ref in zip(record['episodes'][0]['solutions'][0]['records'],expected[record['id']]):
            a=np.array([actual['predicted'],actual['predicted_log_mu'],actual['predicted_log_sigma']]);b=np.array([ref.log_nH,ref.log_muH,ref.log_sigma])
            check('isolated_source_replay',float(np.max(abs(a-b))),1e-10)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    value=dict(passed=all(checks),created_utc=c.e.now(),checks=len(checks),details=details,
        maximum_scaled_or_absolute_error=max(r['error'] for r in details),
        denied_result_reads=result['denied_reads'],seconds=time.perf_counter()-clock,
        code_hashes=c.file_hashes(code),artificial_parameters_not_material_fit=True,goal_complete=False)
    c.write(OUT/'gates.json',value)
    print('State-rate gates',len(checks),'failed',sum(not r for r in checks),flush=True)
    assert value['passed'],'Preserve failed gate;repair by version'


if __name__=='__main__':main()
