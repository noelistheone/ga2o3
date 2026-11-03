"""Actual trained-pilot full loss derivatives, grid refinement and domain audit."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,time,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase33_learning_v1 as l
from spatial_inventory_model_v1 import SpatialInventoryModel
from sandbox.coherent_fermi import published_coherent_profile

PILOT=l.OUT/'all_training_pilot_v1';OUT=l.OUT/'pilot_audit_v1'


def audit_one(path):
    fit=c.document(path);method=fit['method'];config=fit['config'];w=np.array(fit['parameters']);rows=l.subset(c.units())
    start=time.perf_counter();r,J,q=l.loss_arrays(method,w,rows,config)
    fd=[]
    for k in range(8):
        h=2e-5;wp=w.copy();wm=w.copy();wp[k]+=h;wm[k]-=h
        rp=l.loss_arrays(method,wp,rows,config)[0];rm=l.loss_arrays(method,wm,rows,config)[0]
        d=(rp-rm)/(2*h);error=float(max(abs(d-J[:,k])/(1+abs(d)+abs(J[:,k]))))
        fd.append(dict(coordinate=k,error=error,passed=error<2e-4))
    mapper=l.model(method);fine=SpatialInventoryModel(mode=l.METHODS[method],profile=published_coherent_profile(),grid_size=256)
    records=[]
    for row in rows.itertuples():
        p=mapper.point(row,w,require_observation=False);f=fine.point(row,w,require_observation=False)
        v=np.r_[p.values[0],p.transport_values];vf=np.r_[f.values[0],f.transport_values]
        d=p.diagnostics
        records.append(dict(case_id=row.case_id,sample_id=row.sample_id,family=row.family,
            values=v.tolist(),values_N256=vf.tolist(),grid_error_dex=float(max(abs(v-vf))),
            observation_valid=d['observation_domain_valid'],maximum_local_hole_electron_ratio=d['maximum_local_hole_electron_ratio'],
            max_locality_moment1=d['maximum_moment1_ell_grad_logn'],max_locality_moment2=d['maximum_moment2_ell_grad_logn'],
            conductivity_weighted_kell_bad=d['conductivity_weighted_kell_bad_moment1_fraction'],
            resistivity_weighted_kell_bad=d['resistivity_weighted_kell_bad_moment1_fraction'],
            maximum_cell_kell_bad=d['maximum_cell_kell_bad_moment1_fraction'],
            field_span_eV=d['spatial_poisson']['field_span_eV'],predicted_nH_error=float(v[0]-row.target),
            predicted_mu_error=float(v[1]-np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None,
            grid_refined_valid=f.diagnostics['observation_domain_valid'],
            diagnostics=d))
    result=dict(fit_key=fit['key'],method=method,config=config,full_loss_jacobian=fd,
        passed_full_derivatives=all(v['passed'] for v in fd),grid_max_error_dex=max(v['grid_error_dex'] for v in records),
        grid_error_above001=sum(v['grid_error_dex']>.001 for v in records),invalid_grid_refined=sum(not v['grid_refined_valid'] for v in records),
        states=len(records),max_locality_moment1=max(v['max_locality_moment1'] for v in records),
        max_conductivity_weighted_kell_bad=max(v['conductivity_weighted_kell_bad'] for v in records),
        max_resistivity_weighted_kell_bad=max(v['resistivity_weighted_kell_bad'] for v in records),
        seconds=time.perf_counter()-start,records=records,no_heldout_predictive_gain_claim=True,goal_complete=False)
    c.write(OUT/'fits'/(fit['key']+'.json'),result)
    print('pilot audit',method,config,'derivatives',result['passed_full_derivatives'],
        'grid',result['grid_max_error_dex'],'locality',result['max_locality_moment1'],flush=True)
    return {k:v for k,v in result.items() if k!='records'}


def main():
    assert not (OUT/'protocol_before.json').exists();assert (PILOT/'complete.json').exists()
    files=sorted((PILOT/'fits').glob('*.json'));assert len(files)==4
    c.verify_hashes(c.document(PILOT/'protocol_before.json'))
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes([Path(__file__)]),
        input_hashes=c.file_hashes([PILOT/'complete.json',PILOT/'protocol_before.json',*files]),
        full_loss_FD_step=2e-5,full_loss_error_tolerance=2e-4,grid_refinement=[128,256],
        locality_and_kell_diagnostics_not_universal_material_theorems=True,new_fits=0,goal_complete=False))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:records=list(pool.map(audit_one,files))
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'audit.json',dict(created_utc=c.e.now(),passed_all_full_loss_derivatives=all(r['passed_full_derivatives'] for r in records),
        fits=records,seconds=time.perf_counter()-start,new_fits=0,no_heldout_or_material_validity_claim=True,goal_complete=False))


if __name__=='__main__':main()
