"""Posthoc chemical/charge-constrained inventory moment envelopes, no fitting."""
from pathlib import Path
from types import SimpleNamespace
import sys, time
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from sandbox.joint_saved_model import SavedJointModel
from sandbox.fixed_inventory_charge import InventoryState

OUT=c.BASE/'inventory_moment_reachability_v1'
THEORY=c.ROOT/'docs/ML_SANDBOX_INVENTORY_MOMENT_REACHABILITY_20261020.md'
MODES=('foreign_open_host','foreign_closed_host','all_closed_host')
BOXES=('inherited_caps','relative100')
RANK_RTOL=1e-11


def geometry(A,s):
    scale=np.maximum(np.linalg.norm(A,axis=1),1.)
    W=A/scale[:,None]
    _,sv,vt=np.linalg.svd(W,full_matrices=False)
    rank=int(np.sum(sv>RANK_RTOL*sv[0])) if len(sv) and sv[0]>0 else 0
    residual=s-vt[:rank].T@(vt[:rank]@s)
    return dict(constraint_rows=len(A),variable_parents=len(s),numerical_rank=rank,
        numerical_nullity=len(s)-rank,singular_values=sv.tolist(),
        q2_outside_rowspace_relative=float(np.linalg.norm(residual)/max(np.linalg.norm(s),1e-300)))


def main():
    assert not (OUT/'protocol_before.json').exists(),'Preserve first probe and use versioned repairs'
    plans=c.document(c.OLD/'prediction/plan.json')['plans']
    selected=[p for p in plans if p['outer_held'] is not None and p['variant'] in
              ('Reference205_internal','JointProcess_nHonly_internal')]
    inputs=[c.OLD/'prediction/plan.json',c.OLD/'data/scenarios49.csv']+[c.ROOT/p['fit_path'] for p in selected]
    code=[Path(__file__),Path(c.__file__),THEORY]+list((c.ROOT/'src/sandbox').glob('*.py'))
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes(code),
        input_hashes=c.file_hashes(set(inputs)),modes=list(MODES),boxes=list(BOXES),
        planned_states=50,planned_linear_programs=600,rank_relative_tolerance=RANK_RTOL,
        highs_primal_and_dual_tolerance=1e-9,relative_box_factor=100.,
        held_out_labels_not_used_to_select_inventory_witnesses=True,new_fits=0,
        necessary_inventory_ranges_not_thermodynamic_or_kinetic_reachability=True,
        exact_free_n_preservation_not_exact_Hall_n_preservation=True,goal_complete=False))
    cases=c.cases();paired=cases[cases.mobility_eligible.eq(True)]
    rows=[];witnesses=[];state_checks=[];geometry_rows=[];start=time.monotonic()
    for plan in selected:
        mapper=SavedJointModel(plan['minimal_fit'])
        for row in paired[paired.related_cluster.eq(plan['outer_held'])].itertuples():
            point=mapper.point(SimpleNamespace(**c.design.inputs(row)),require_observation=False)
            backend=(mapper.engine.engine_for(row) if hasattr(mapper.engine,'engine_for')
                     else mapper.engine.engines[row.scenario_pO2_atm])
            layer,solver=backend.layer_for(row)
            inventory=solver.last_inventory;cold=solver.last_prediction;c0=inventory.totals.copy()
            dm=layer.db.dm[layer.first];atom=solver.parent_atoms.T
            a=np.bincount(layer.parent,weights=cold.probabilities*layer.q,minlength=layer.db.nD)
            s=np.bincount(layer.parent,weights=cold.probabilities*layer.q**2,minlength=layer.db.nD)
            foreign=np.any(atom>0,axis=0);pref=layer.db.prefactor[layer.first]
            cap=np.where(foreign,.1*(1-1e-6),1.)*pref
            state_checks.extend([bool(np.all(c0>=0)),bool(np.all(c0<=cap*(1+1e-12)))])
            scale=float(row.incorporated_cm3)
            Nd=float(cold.diagnostics['Nd_cm3']);original=float(Nd+s@c0)
            pars=layer.parameters(T=point.diagnostics['Tf_K'],Nd=Nd,weights=[0.],
                doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
                pO2=row.scenario_pO2_atm,Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
            chain=np.zeros((layer.np,inventory.jacobian.shape[1]))
            for mode in MODES:
                active=foreign if mode!='all_closed_host' else np.ones(len(c0),bool)
                B=atom if mode=='foreign_open_host' else np.vstack([dm[:,:2].T,atom])
                A=np.vstack([B[:,active],a[active]])
                info=geometry(A,s[active])
                geometry_rows.append(dict(variant=plan['variant'],case_id=row.case_id,mode=mode,**info))
                v0=c0[active]/scale;b=A@v0
                rs=np.maximum(1.,np.maximum(np.max(abs(A),axis=1),abs(b)))
                Ae=A/rs[:,None];be=b/rs
                fixed=float(Nd+s[~active]@c0[~active])
                for box in BOXES:
                    lower=np.zeros_like(v0);upper=cap[active]/scale
                    if box=='relative100':
                        lower=v0/100.;upper=np.minimum(upper,100.*v0)
                    for direction,sign in [('min',1.),('max',-1.)]:
                        opt=linprog(sign*s[active],A_eq=Ae,b_eq=be,
                            bounds=list(zip(lower,upper)),method='highs',
                            options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                        entry=dict(variant=plan['variant'],case_id=row.case_id,sample_id=row.sample_id,
                            element=row.element,mode=mode,box=box,direction=direction,
                            lp_status=int(opt.status),lp_success=bool(opt.success),lp_message=opt.message,
                            original_Nq2_cm3=original,original_free_n_cm3=cold.diagnostics['free_n_cm3'],
                            original_EF_eV=cold.ef_eV,numerical_rank=info['numerical_rank'],
                            numerical_nullity=info['numerical_nullity'],
                            q2_outside_rowspace_relative=info['q2_outside_rowspace_relative'])
                        if opt.success:
                            stock=c0.copy();stock[active]=opt.x*scale
                            q2=float(fixed+scale*(s[active]@opt.x))
                            denom=1+np.sum(abs(A)*np.maximum(abs(v0),abs(opt.x)),axis=1)
                            eqerr=float(max(abs(A@opt.x-b)/denom))
                            foreign_error=float(max(abs(atom@stock-atom@c0)/(1+abs(atom@c0))))
                            host_change=dm[:,:2].T@(stock-c0)
                            entry.update(witness_Nq2_at_fixed_EF_cm3=q2,
                                q2_ratio_to_original=q2/original,equality_relative_error=eqerr,
                                foreign_atom_relative_error=foreign_error,
                                host_Ga_change_cm3=float(host_change[0]),host_O_change_cm3=float(host_change[1]),
                                parent_fraction_max=float(max(stock/pref)),
                                native_parent_fraction_max=float(max(stock[solver.native]/pref[solver.native])),
                                bound_saturated_parents=int(sum(np.isclose(opt.x,upper,rtol=1e-8,atol=1e-12))))
                            diagnostics=dict(inventory.diagnostics,
                                probe_scope='LP necessary chemistry/charge range;not a learned or thermodynamic state')
                            inv=InventoryState(stock,np.zeros_like(inventory.jacobian),diagnostics)
                            try:
                                trial=solver.solve(pars,inv,chain)
                                entry.update(cold_replay_success=True,
                                    cold_EF_difference_eV=float(trial.ef_eV-cold.ef_eV),
                                    cold_log_free_n_difference=float(trial.values[0]-cold.values[0]),
                                    cold_Nq2_cm3=trial.diagnostics['charge_squared_scatter_budget_cm3'],
                                    cold_q2_relative_difference=float(abs(trial.diagnostics['charge_squared_scatter_budget_cm3']-q2)/(1+q2)),
                                    replay_close_to_fixed_EF=bool(abs(trial.ef_eV-cold.ef_eV)<1e-4 and
                                        abs(trial.values[0]-cold.values[0])<1e-3))
                            except Exception as error:
                                entry.update(cold_replay_success=False,cold_replay_error=f'{type(error).__name__}: {error}')
                            witnesses.append(dict(variant=plan['variant'],case_id=row.case_id,mode=mode,
                                box=box,direction=direction,old_c_cm3=c0.tolist(),new_c_cm3=stock.tolist(),
                                original_parent_ids=layer.original_parent_ids[layer.first].tolist(),
                                average_q=a.tolist(),average_q2=s.tolist(),active=active.tolist(),
                                formula_delta_atoms=dm.tolist(),upper_cap_cm3=cap.tolist()))
                        rows.append(entry)
            print('inventory range',plan['variant'],row.case_id,'LPs',len(rows),flush=True)
    assert len(rows)==600 and len(geometry_rows)==150 and len(state_checks)==100
    c.e.csvwrite(OUT/'moment_ranges.csv',rows)
    c.write(OUT/'constraint_geometry.json',geometry_rows)
    c.write(OUT/'inventory_witnesses.json',witnesses)
    summary=[]
    for variant in sorted({r['variant'] for r in rows}):
        for mode in MODES:
            for box in BOXES:
                q=[r for r in rows if r['variant']==variant and r['mode']==mode and r['box']==box]
                valid=[r for r in q if r['lp_success']]
                replay=[r for r in valid if r.get('cold_replay_success',False)]
                summary.append(dict(variant=variant,mode=mode,box=box,linear_programs=len(q),
                    lp_successes=len(valid),cold_replay_successes=len(replay),
                    fixed_EF_close=sum(r.get('replay_close_to_fixed_EF',False) for r in replay),
                    q2_ratio_min=min((r['q2_ratio_to_original'] for r in valid),default=None),
                    q2_ratio_max=max((r['q2_ratio_to_original'] for r in valid),default=None),
                    max_EF_difference_eV=max((abs(r['cold_EF_difference_eV']) for r in replay),default=None),
                    max_parent_fraction=max((r['parent_fraction_max'] for r in valid),default=None)))
    c.e.csvwrite(OUT/'summary.csv',summary)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'audit.json',dict(created_utc=c.e.now(),passed=all(state_checks),
        original_state_domain_checks=len(state_checks),actual_saved_states=50,linear_programs=len(rows),
        lp_successes=sum(r['lp_success'] for r in rows),
        cold_replay_successes=sum(r.get('cold_replay_success',False) for r in rows),
        failed_or_nonclose_witnesses_preserved=True,seconds=time.monotonic()-start,
        no_new_fit=True,no_predictive_gain_claim=True,no_thermodynamic_reachability_claim=True,
        finite_precision_rank_not_analytic_rank=True,goal_complete=False))
    assert all(state_checks),'Original state outside inherited domain;preserve result'
    for row in summary:print(row,flush=True)


if __name__=='__main__':main()
