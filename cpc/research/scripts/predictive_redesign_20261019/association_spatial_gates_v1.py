"""Full coupled inventory-fraction-profile-Poisson-Hall derivative gate."""
from pathlib import Path
from types import SimpleNamespace
import sys,time,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from si_multidonor_overlay_v1 import make_overlay
from association_spatial_model_v1 import AssociationSpatialModel
from sandbox.coherent_fermi import published_coherent_profile

OUT=c.BASE/'association_spatial_candidate/gates_v1'


def values(p):return np.r_[p.values[0],p.transport_values]
def jac(p):return np.r_[p.jacobian[[0]],p.transport_jacobian]


def main():
    assert not (OUT/'protocol_before.json').exists()
    selected=c.document(c.BASE/'spatial_predictive_candidate/local_charge_gates_v2/protocol_before.json')['selected_cases'][:3]
    fitted,path=c.fit(c.units(),'ShareProcess205',dict(gamma=1.,alpha=.1));db,meta=make_overlay()
    modes=[dict(coupling='triple_fraction',background='co_profiled'),dict(coupling='constant_amplitude',background='co_profiled'),dict(coupling='triple_fraction',background='uniform')]
    code=[Path(__file__),*[Path(__file__).with_name(q) for q in ['association_spatial_model_v1.py','neutral_association_model_v1.py',
        'neutral_association_inventory_v1.py','si_multidonor_overlay_v1.py','spatial_local_charge_v2.py','spatial_poisson_v2.py','spatial_parent_profile_v1.py','spatial_transport_v1.py']]]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes(code),
        input_hashes=c.file_hashes([path,c.BASE/'neutral_association_candidate/gates_v1/gates.json',
            c.BASE/'spatial_predictive_candidate/vector_background_gates_v1/gates.json',c.OLD/'data/scenarios49.csv']),
        selected_cases=selected,modes=modes,overlay=meta,period_nm=1000.,FD_grid=64,FD_step=2e-5,
        grid_comparison=[64,128,256],no_fit_or_predictive_gain_claim=True,goal_complete=False))
    start=time.perf_counter();checks=[];records=[];derivatives=[];grid=[];lookup={r.case_id:r for r in c.cases().itertuples()}
    for mode in modes:
        mapper=AssociationSpatialModel(profile=published_coherent_profile(),db=db,grid_size=64,**mode)
        fine=[AssociationSpatialModel(profile=published_coherent_profile(),db=db,grid_size=N,**mode) for N in [128,256]]
        for chosen in selected:
            row=SimpleNamespace(**c.design.inputs(lookup[chosen['case_id']]))
            w=np.r_[fitted['parameters'],4.,12.];entry=dict(case_id=chosen['case_id'],**mode)
            try:
                p=mapper.point(row,w,source_kind='share',source_shift=.02,require_observation=False);J=jac(p)
                checks.extend([abs(p.diagnostics['background_inventory']['parent_mass_relative_error'])<2e-10,
                    p.diagnostics['spatial_poisson']['global_charge_relative_error']<2e-8])
                for k in range(8):
                    h=2e-5;wp=w.copy();wm=w.copy();wp[k]+=h;wm[k]-=h
                    fd=(values(mapper.point(row,wp,source_kind='share',source_shift=.02,require_observation=False))-
                        values(mapper.point(row,wm,source_kind='share',source_shift=.02,require_observation=False)))/(2*h)
                    er=float(max(abs(fd-J[:,k])/(1+abs(fd)+abs(J[:,k]))));checks.append(er<8e-5)
                    derivatives.append(dict(case_id=chosen['case_id'],kind='parameter',coordinate=k,error=er,**mode))
                for name,k in [('source',0),*[(v,i) for i,v in enumerate(p.input_names)]]:
                    h=.01 if name=='growth_T_C' else 2e-5
                    def changed(delta):
                        r=SimpleNamespace(**vars(row));z=.02
                        if name=='source':z+=delta
                        elif name=='growth_T_C':r.growth_T_C+=delta
                        elif name=='log10_dose_cm3':r.incorporated_cm3*=10**delta
                        elif name=='log10_H_cm3':r.H_cm3*=10**delta
                        return values(mapper.point(r,w,source_kind='share',source_shift=z,require_observation=False))
                    fd=(changed(h)-changed(-h))/(2*h)
                    d=np.r_[p.source_jacobian,p.transport_source_jacobian] if name=='source' else np.r_[p.input_jacobian[0,k],p.transport_input_jacobian[:,k]]
                    er=float(max(abs(fd-d)/(1+abs(fd)+abs(d))));checks.append(er<8e-5)
                    derivatives.append(dict(case_id=chosen['case_id'],kind=name,error=er,**mode))
                w0=w.copy();w0[7]=0.;u=mapper.point(row,w0,source_kind='share',source_shift=.02,require_observation=False)
                prefix=mapper.prefix.point(row,w0[:7],source_kind='share',source_shift=.02,require_observation=False)
                uniform_error=float(max(abs(values(u)-values(prefix))));checks.append(uniform_error<2e-8)
                refined=[m.point(row,w,source_kind='share',source_shift=.02,require_observation=False) for m in fine]
                grid.append(dict(case_id=chosen['case_id'],**mode,error64_256=float(max(abs(values(p)-values(refined[-1])))),
                    error128_256=float(max(abs(values(refined[0])-values(refined[1]))))))
                entry.update(success=True,values=values(p).tolist(),valid=p.diagnostics['observation_domain_valid'],
                    uniform_error=uniform_error,diagnostics=p.diagnostics)
            except Exception as exc:checks.append(False);entry.update(success=False,error=repr(exc))
            records.append(entry)
        print('association-spatial gate',mode,flush=True)
    c.write(OUT/'records.json',records);c.write(OUT/'derivatives.json',derivatives);c.write(OUT/'grid.json',grid)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    result=dict(created_utc=c.e.now(),passed=all(checks),checks=len(checks),failures=sum(not bool(q) for q in checks),
        actual_states=len(records),successful_states=sum(r['success'] for r in records),derivative_coordinates=len(derivatives),
        max_FD_error=max((r['error'] for r in derivatives),default=None),
        max_grid128_256_error=max((r['error128_256'] for r in grid),default=None),
        grid_refinement_is_reported_diagnostic=True,seconds=time.perf_counter()-start,new_fits=0,goal_complete=False)
    c.write(OUT/'gates.json',result);print('Coupled gate',len(checks),'failed',result['failures'],flush=True)
    assert all(checks),'Preserve failed records and use versioned repair'


if __name__=='__main__':main()
