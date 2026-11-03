"""Full shared-parameter/source/input Jacobians and spatial discretization audit."""
from pathlib import Path
from types import SimpleNamespace
import sys,time,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from spatial_inventory_model_v1 import SpatialInventoryModel
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel

OUT=c.BASE/'spatial_predictive_candidate/model_gates_v1'


def vals(point):return np.r_[point.values[0],point.transport_values]
def jac(point):return np.r_[point.jacobian,point.transport_jacobian]


def main():
    assert not (OUT/'protocol_before.json').exists()
    selected=c.document(c.BASE/'spatial_predictive_candidate/local_charge_gates_v2/protocol_before.json')['selected_cases'][:3]
    lookup={r.case_id:r for r in c.cases().itertuples()};fit,path=c.fit(c.units(),'ShareProcess205',dict(gamma=1.,alpha=1.))
    core=[Path(__file__),Path(__file__).with_name('spatial_inventory_model_v1.py'),
        *[Path(__file__).with_name(x) for x in ['spatial_transport_v1.py','spatial_local_charge_v1.py','spatial_parent_profile_v1.py','spatial_poisson_v1.py']]]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes(core),
        input_hashes=c.file_hashes([path,c.BASE/'spatial_predictive_candidate/transport_gates_v1/gates.json',
            c.OLD/'data/scenarios49.csv']),selected_cases=selected,stock_preset=fit['key'],
        modes=['common','structural_contrast'],FD_amplitude=.2,FD_period_nm=200.,FD_grid=32,
        grid_comparison=[32,64,128],grid_amplitudes=[0.,.2,2.,8.],grid_periods_nm=[20.,200.,1000.],
        grid_errors_are_diagnostics_not_automatically_passed_accuracy=True,goal_complete=False))
    start=time.perf_counter();checks=[];records=[];grids=[];profile=published_coherent_profile()
    mappers={(mode,N):SpatialInventoryModel(mode=mode,profile=profile,grid_size=N) for mode in ['common','structural_contrast'] for N in [32,64,128]}
    baseline=ChargeFeasibleModel(basis='process',profile=profile)
    for item in selected:
        row=SimpleNamespace(**c.design.inputs(lookup[item['case_id']]));w=np.r_[fit['parameters'],.2,np.log10(200.)]
        for mode in ['common','structural_contrast']:
            mapper=mappers[(mode,32)]
            try:
                p=mapper.point(row,w,source_kind='share',source_shift=.05,require_observation=False)
                analytic=jac(p)
                for k in range(8):
                    h=1e-4 if k!=1 else 3e-4;wp=w.copy();wm=w.copy();wp[k]+=h;wm[k]-=h
                    fd=(vals(mapper.point(row,wp,source_kind='share',source_shift=.05,require_observation=False))-
                        vals(mapper.point(row,wm,source_kind='share',source_shift=.05,require_observation=False)))/(2*h)
                    error=float(max(abs(fd-analytic[:,k])/(1+abs(fd)+abs(analytic[:,k]))))
                    records.append(dict(case_id=item['case_id'],mode=mode,kind='parameter',coordinate=k,error=error));checks.append(error<1e-4)
                for name,k in [('source',0),*[(v,i) for i,v in enumerate(p.input_names)]]:
                    h=2e-5 if name!='growth_T_C' else .01
                    def changed(delta):
                        r=SimpleNamespace(**vars(row));z=.05
                        if name=='source':z+=delta
                        elif name=='growth_T_C':r.growth_T_C+=delta
                        elif name=='log10_dose_cm3':r.incorporated_cm3*=10**delta
                        elif name=='log10_H_cm3':r.H_cm3*=10**delta
                        return mapper.point(r,w,source_kind='share',source_shift=z,require_observation=False)
                    fd=(vals(changed(h))-vals(changed(-h)))/(2*h)
                    aj=np.r_[p.source_jacobian,p.transport_source_jacobian] if name=='source' else np.r_[p.input_jacobian[0,k],p.transport_input_jacobian[:,k]]
                    error=float(max(abs(fd-aj)/(1+abs(fd)+abs(aj))))
                    checks.append(error<1e-4);records.append(dict(case_id=item['case_id'],mode=mode,kind=name,error=error))
                w0=w.copy();w0[6]=0.
                p0=mapper.point(row,w0,source_kind='share',source_shift=.05,require_observation=False)
                old=baseline.point(row,w0[:6],source_kind='share',source_shift=.05,require_observation=False)
                ev=float(max(abs(vals(p0)-vals(old))));ej=float(np.max(abs(jac(p0)[:,:6]-jac(old))))
                checks.extend([ev<2e-8,ej<1e-6]);records.append(dict(case_id=item['case_id'],mode=mode,kind='uniform_original',value_error=ev,jacobian_error=ej))
                forbidden=SimpleNamespace(**vars(row),target=-100.,mu_cm2Vs=1e-30,related_cluster='not an input',sample_id='not an input')
                q=mapper.point(forbidden,w,source_kind='share',source_shift=.05,require_observation=False)
                checks.append(max(abs(vals(p)-vals(q)))==0.)
            except Exception as exc:checks.append(False);records.append(dict(case_id=item['case_id'],mode=mode,kind='exception',error=repr(exc)))
            for a in [0.,.2,2.,8.]:
                for L in [20.,200.,1000.]:
                    entry=dict(case_id=item['case_id'],mode=mode,amplitude=a,period_nm=L)
                    try:
                        wx=w.copy();wx[6]=a;wx[7]=np.log10(L)
                        pp=[mappers[(mode,N)].point(row,wx,require_observation=False) for N in [32,64,128]]
                        values=[vals(q).tolist() for q in pp]
                        entry.update(success=True,values_N32_N64_N128=values,
                            error_N32_N128=float(max(abs(vals(pp[0])-vals(pp[2])))),
                            error_N64_N128=float(max(abs(vals(pp[1])-vals(pp[2])))),
                            valid_N32_N64_N128=[q.diagnostics['observation_domain_valid'] for q in pp],
                            maximum_local_hole_electron_ratio=pp[-1].diagnostics['maximum_local_hole_electron_ratio'])
                    except Exception as exc:entry.update(success=False,error=repr(exc))
                    grids.append(entry)
        print('shared spatial model gate',item['case_id'],flush=True)
    c.write(OUT/'derivative_and_uniform_records.json',records);c.write(OUT/'grid_records.json',grids)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    numerical=[r for r in grids if r['success']]
    result=dict(created_utc=c.e.now(),passed=all(checks),checks=len(checks),failures=sum(not bool(q) for q in checks),
        max_derivative_error=max((r['error'] for r in records if isinstance(r.get('error'),float)),default=None),
        exceptions=[r for r in records if r['kind']=='exception'],grid_cases=len(grids),successful_grid_cases=len(numerical),
        max_N32_N128_error=max((r['error_N32_N128'] for r in numerical),default=None),
        max_N64_N128_error=max((r['error_N64_N128'] for r in numerical),default=None),
        grid_convergence_not_an_unconditional_gate=True,seconds=time.perf_counter()-start,
        new_models_fitted=0,goal_complete=False)
    c.write(OUT/'gates.json',result);print('Full model gates',len(checks),'failed',result['failures'],'grid_success',len(numerical),flush=True)
    assert all(checks),'Preserve original failures and use versioned repairs'


if __name__=='__main__':main()
