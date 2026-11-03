"""Actual-state atom budgets and full model/source/input derivative checks."""
from pathlib import Path
from types import SimpleNamespace
import sys,time,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from neutral_association_model_v1 import NeutralAssociationModel
from si_multidonor_overlay_v1 import make_overlay
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel

OUT=c.BASE/'neutral_association_candidate/gates_v1'


def values(p):return np.r_[p.values[0],p.transport_values]
def jac(p):return np.r_[p.jacobian[[0]],p.transport_jacobian]


def main():
    assert not (OUT/'protocol_before.json').exists()
    selected=c.document(c.BASE/'spatial_predictive_candidate/local_charge_gates_v2/protocol_before.json')['selected_cases'][:3]
    fit,path=c.fit(c.units(),'ShareProcess205',dict(gamma=1.,alpha=.1));db,meta=make_overlay()
    code=[Path(__file__),*[Path(__file__).with_name(q) for q in ['neutral_association_model_v1.py','neutral_association_inventory_v1.py','si_multidonor_overlay_v1.py']]]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes(code),
        input_hashes=c.file_hashes([path,c.BASE/'si_multidonor_overlay_v1/gates.json',c.OLD/'data/scenarios49.csv']),
        overlay=meta,bases=['constant','dose'],association_intercepts=[-8.,0.,8.],
        selected_cases=selected,softmin_power=20.,all_input_derivatives_including_Tobs=True,new_fits=0,goal_complete=False))
    lookup={r.case_id:r for r in c.cases().itertuples()};profile=published_coherent_profile()
    baseline=ChargeFeasibleModel(basis='process',profile=profile,db=db);checks=[];records=[];start=time.perf_counter()
    for basis in ['constant','dose']:
        mapper=NeutralAssociationModel(association_basis=basis,profile=profile,db=db)
        for item in selected:
            row=SimpleNamespace(**c.design.inputs(lookup[item['case_id']]));w=np.r_[fit['parameters'],0.,*([1.] if basis=='dose' else [])]
            for a in [-8.,0.,8.]:
                w[6]=a;entry=dict(case_id=item['case_id'],basis=basis,association_intercept=a)
                try:
                    p=mapper.point(row,w,source_kind='share',source_shift=.02,require_observation=False)
                    d=p.diagnostics
                    checks.extend([d['association_atom_conservation_relative_error']<2e-12,
                        d['association_atom_derivative_relative_error']<2e-10,
                        abs(values(p)[2]-np.log10(c.learning.E_CHARGE)-values(p)[0]-values(p)[1])<1e-12])
                    entry.update(success=True,values=values(p).tolist(),valid=d['observation_domain_valid'],
                        unassociated_values=p.unassociated_values.tolist(),
                        extent=d['neutral_association_extent_cm3'],capacity=d['neutral_association_smooth_capacity_cm3'],
                        atom_error=d['association_atom_conservation_relative_error'],atom_derivative_error=d['association_atom_derivative_relative_error'])
                    if a==0.:
                        J=jac(p)
                        for k in range(len(w)):
                            h=2e-5;wp=w.copy();wm=w.copy();wp[k]+=h;wm[k]-=h
                            fd=(values(mapper.point(row,wp,source_kind='share',source_shift=.02,require_observation=False))-
                                values(mapper.point(row,wm,source_kind='share',source_shift=.02,require_observation=False)))/(2*h)
                            error=float(max(abs(fd-J[:,k])/(1+abs(fd)+abs(J[:,k]))));checks.append(error<5e-5)
                            records.append(dict(case_id=item['case_id'],basis=basis,kind='parameter_FD',coordinate=k,error=error))
                        for name,k in [('source',0),*[(v,i) for i,v in enumerate(p.input_names)]]:
                            h=.01 if name=='growth_T_C' else 2e-5
                            def changed(delta):
                                r=SimpleNamespace(**vars(row));z=.02
                                if name=='source':z+=delta
                                elif name=='growth_T_C':r.growth_T_C+=delta
                                elif name=='log10_dose_cm3':r.incorporated_cm3*=10**delta
                                elif name=='log10_Tobs_K':r.Tobs_K*=10**delta
                                elif name=='log10_H_cm3':r.H_cm3*=10**delta
                                return values(mapper.point(r,w,source_kind='share',source_shift=z,require_observation=False))
                            fd=(changed(h)-changed(-h))/(2*h)
                            Jp=np.r_[p.source_jacobian[0],p.transport_source_jacobian] if name=='source' else np.r_[p.input_jacobian[0,k],p.transport_input_jacobian[:,k]]
                            error=float(max(abs(fd-Jp)/(1+abs(fd)+abs(Jp))));checks.append(error<5e-5)
                            records.append(dict(case_id=item['case_id'],basis=basis,kind=name,error=error))
                    w0=w.copy();w0[6]=-75.;p0=mapper.point(row,w0,source_kind='share',source_shift=.02,require_observation=False)
                    old=baseline.point(row,w0[:6],source_kind='share',source_shift=.02,require_observation=False)
                    checks.append(max(abs(values(p0)-values(old)))<1e-8)
                except Exception as exc:checks.append(False);entry.update(success=False,error=repr(exc))
                records.append(entry)
        print('Association gates basis',basis,flush=True)
    c.write(OUT/'records.json',records);c.verify_hashes(c.document(OUT/'protocol_before.json'))
    failures=[r for r in records if r.get('success') is False]
    c.write(OUT/'gates.json',dict(created_utc=c.e.now(),passed=all(checks),checks=len(checks),failures=sum(not bool(q) for q in checks),
        exceptions=failures,actual_atomic_cases=18,full_FD_coordinates=sum('error' in r and isinstance(r['error'],float) for r in records),
        max_FD_error=max((r['error'] for r in records if isinstance(r.get('error'),float)),default=None),
        seconds=time.perf_counter()-start,new_fits=0,no_material_or_predictive_gain_claim=True,goal_complete=False))
    print('Association gates',len(checks),'failed',sum(not bool(q) for q in checks),flush=True)
    assert all(checks),'Preserve failure and use versioned repair'


if __name__=='__main__':main()
