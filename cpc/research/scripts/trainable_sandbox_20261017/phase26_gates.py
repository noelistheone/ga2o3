"""Actual state/chain gates before any absolute-share property training."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.pool_share_model import PoolShareModel
from sandbox.absolute_pool_share import allocate_absolute_share
from sandbox.relative_inventory import allocate_relative_odds
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.context_saved_model import INPUT_NAMES
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

OUT=e.BASE/'phase26_pool_share';PARENT=e.BASE/'phase25_shared_process'


def vector(p):return np.r_[p.values,p.transport_values]


def main():
    assert not (OUT/'numerical_gates.json').exists(),'Preserve any earlier gate result before a versioned repair'
    assert e.sha(e.BASE/'artifact_manifest_phase25.json')=='15003caddff0bdf6c2048a98929c2d185d05e2e56dc7d03f29ce13fc58a5d500'
    data=read(PARENT/'data/conditions33.csv')
    selected=[data[data.element.eq('Si')&data.H_known.eq(True)].iloc[0],
        data[data.element.eq('Si')&data.H_known.eq(False)&data.family.ne('10.1063/1.5047214')].iloc[0],
        data[data.element.eq('Sn')].iloc[0],data[data.sample_id.eq('10.1063/1.5047214::target-1')].iloc[0]]
    cases=[SimpleNamespace(**{k:e.clean(r[k]) for k in INPUT_NAMES}) for r in selected]
    code=[Path(__file__),ROOT/'src/sandbox/pool_share_model.py',ROOT/'src/sandbox/absolute_pool_share.py',
        ROOT/'src/sandbox/fixed_inventory_charge.py',ROOT/'src/sandbox/coherent_transport.py',ROOT/'src/sandbox/relative_inventory_model.py']
    inputs=[PARENT/'data/conditions33.csv',e.BASE/'artifact_manifest_phase25.json',ROOT/'docs/ML_SANDBOX_POOL_FRACTION_NEXT_20261018.md']
    e.write(OUT/'gate_protocol_before.json',dict(created_utc=e.now(),purpose='state/derivative consistency,not accuracy or training',
        numerical_share_logit_domain=[-80,80],cases=[vars(c) for c in cases],
        temperature_modes=['effective','growth'],basis=['constant','dose','process'],
        finite_difference_steps=dict(parameters=1e-4,loginputs=1e-4,growth_C=.02,source=1e-4),
        derivative_scaled_tolerance=3e-5,training_done=False,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs}))
    checks=[];omission=0.;states=[]
    def check(name,x,tol):checks.append(dict(name=name,value=float(x),tolerance=tol,passed=bool(x<=tol)))
    profile=published_coherent_profile()
    for mode in ('effective','growth'):
        for basis in ('constant','dose','process'):
            model=PoolShareModel(basis=basis,temperature_mode=mode,profile=profile)
            for i,row in enumerate(cases):
                nb=model.nbase;w=np.r_[[.5,16.] if nb==2 else [16.],np.array([2.4,.4,-.2,.1])[:len(model.parameter_names)-nb]]
                a=model.point(row,w,source_kind='share',source_shift=.1,require_observation=False);label=f'{mode}/{basis}/{i}'
                ana=np.r_[a.jacobian,a.transport_jacobian];num=[]
                for k in range(len(w)):
                    delta=np.zeros(len(w));delta[k]=1e-4
                    num.append((vector(model.point(row,w+delta,source_kind='share',source_shift=.1,require_observation=False))-
                        vector(model.point(row,w-delta,source_kind='share',source_shift=.1,require_observation=False)))/.0002)
                check(label+'/parameters',np.max(abs(np.array(num).T-ana)/(1+abs(ana))),3e-5)
                ai=np.r_[a.input_jacobian,a.transport_input_jacobian];ni=[]
                for name in a.input_names:
                    p=SimpleNamespace(**vars(row));q=SimpleNamespace(**vars(row))
                    if name=='growth_T_C':step=.02;p.growth_T_C+=step;q.growth_T_C-=step
                    else:
                        step=1e-4;k={'log10_dose_cm3':'incorporated_cm3','log10_Tobs_K':'Tobs_K','log10_H_cm3':'H_cm3'}[name]
                        setattr(p,k,getattr(p,k)*10**step);setattr(q,k,getattr(q,k)*10**(-step))
                    ni.append((vector(model.point(p,w,source_kind='share',source_shift=.1,require_observation=False))-
                        vector(model.point(q,w,source_kind='share',source_shift=.1,require_observation=False)))/(2*step))
                check(label+'/inputs',np.max(abs(np.array(ni).T-ai)/(1+abs(ai))),3e-5)
                for kind in ('share','offset'):
                    mid=model.point(row,w,source_kind=kind,source_shift=.1,require_observation=False)
                    p=model.point(row,w,source_kind=kind,source_shift=.1001,require_observation=False)
                    q=model.point(row,w,source_kind=kind,source_shift=.0999,require_observation=False)
                    aj=np.r_[mid.source_jacobian,mid.transport_source_jacobian]
                    check(label+'/'+kind+'/source',np.max(abs((vector(p)-vector(q))/.0002-aj)/(1+abs(aj))),3e-5)
                a=model.point(row,w,source_kind='share',source_shift=.1,require_observation=False)
                layer,solver=model.engine_for(row).layer_for(row);inv=solver.last_inventory;ref=solver._reference_cache[1]
                U=float(solver.dopant_count[solver.complex]@inv.totals[solver.complex]);V=float(inv.totals[solver.receivers].sum())
                check(label+'/actual_share',abs(V/(U+V)-a.diagnostics['isolated_atom_fraction_after']),2e-12)
                check(label+'/foreign',a.diagnostics['foreign_atom_relative_error'],3e-8)
                check(label+'/foreign_derivative',a.diagnostics['foreign_atom_derivative_error'],3e-7)
                check(label+'/native_unchanged',np.max(abs(inv.totals[solver.native]-ref.totals[solver.native])),0.)
                host=layer.db.dm[layer.first,:2].T@(inv.totals-ref.totals)
                check(label+'/oxygen',abs(host[1])/(U+V),1e-12)
                check(label+'/nonnegative',max(0.,-float(inv.totals.min())),0.)
                check(label+'/probability',a.diagnostics['parent_probability_error'],2e-12)
                check(label+'/charge_root',a.diagnostics['charge_relative_residual'],2e-8)
                check(label+'/derived_sigma',abs(a.transport_values[1]-np.log10(1.602176634e-19)-a.values[0]-a.transport_values[0]),2e-10)
                if mode=='growth':check(label+'/growthreference',abs(a.diagnostics['Tf_K']-row.growth_T_C-273.15),1e-10)
                # Equivalence is checked on inventories AND their derivatives,
                # including the reference-logit derivative omitted by a naive
                # change of coordinates.
                T=a.diagnostics['Tf_K'];pars=layer.parameters(T=T,Nd=1e16,weights=[0.],
                    doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
                    pO2=row.scenario_pO2_atm,Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
                chain=np.eye(layer.np);r=solver.reference(pars);dc=r.jacobian@chain
                U0=float(solver.dopant_count[solver.complex]@r.totals[solver.complex]);V0=float(r.totals[solver.receivers].sum())
                dU=solver.dopant_count[solver.complex]@dc[solver.complex];dV=dc[solver.receivers].sum(axis=0)
                dh=np.zeros(layer.np);dh[layer.idose]=.2;h=.2;psi=np.log(V0/U0)+np.log(10)*h
                rel=allocate_relative_odds(solver,pars,chain,h,dh)
                absolute=allocate_absolute_share(solver,pars,chain,psi,dV/V0-dU/U0+np.log(10)*dh)
                check(label+'/coordinate_inventory',np.max(abs(rel.totals-absolute.totals)/(1+rel.totals)),2e-10)
                check(label+'/coordinate_derivative',np.max(abs(rel.jacobian-absolute.jacobian)/(1+abs(rel.jacobian))),3e-7)
                if basis!='constant':
                    constant=PoolShareModel(basis='constant',temperature_mode=mode,profile=profile)
                    fixed=constant.point(row,np.r_[w[:nb],float(np.array([1.,(np.log10(row.incorporated_cm3)-18)/2]+(
                        [(row.growth_T_C-700)/300,float(row.element=='Sn')] if basis=='process' else []))@w[nb:])],
                        source_kind='share',source_shift=.1,require_observation=False)
                    omission=max(omission,abs(a.input_jacobian[0,0]-fixed.input_jacobian[0,0]))
                states.append(dict(case=label,nH_log10=float(a.values[0]),mu_log10=float(a.transport_values[0]),
                    sigma_log10=float(a.transport_values[1]),observation_valid=a.diagnostics['observation_domain_valid'],
                    share=a.diagnostics['isolated_atom_fraction_after'],reference_share=a.diagnostics['isolated_atom_fraction_before']))
    check('nonzero_learned_dose_chain_observed',0. if omission>1e-3 else 1.,0.)
    e.csvwrite(OUT/'gate_states.csv',states)
    result=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,
        maximum_scaled_derivative_error=max(c['value'] for c in checks if c['name'].endswith(('/parameters','/inputs','/source'))),
        maximum_input_chain_omission=omission,state_cases=len(states),new_shared_training=0,new_material_measurements=0,
        grade='numerical and atom-contract consistency only;not propertyaccuracy or microscopictruth',goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code})
    e.write(OUT/'numerical_gates.json',result)
    print('phase26 absolute-share gates',result['passed'],len(checks),result['maximum_scaled_derivative_error'],flush=True)
    if not result['passed']:raise RuntimeError('Retain failed gates/code before repair')


if __name__=='__main__':main()
