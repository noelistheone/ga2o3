"""Audit actual selected states without removing points or modifying scores."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from sandbox.joint_saved_model import SavedJointModel
from sandbox.context_saved_model import INPUT_NAMES
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.transport_scope_diagnostic import reconstruct_transport
from sandbox.transport import mu_caughey_thomas
from sandbox.ionized_transport import E_CHARGE
from sandbox.kroger_db import NPZ,N_SITE
import phase29_data as data
import phase2_energy as e

OUT=data.OUT/'physical_scope'


def main():
    assert not (OUT/'protocol_before.json').exists(),'preserve prior scope audit before versioned repair'
    assert json.loads((data.OUT/'transport_scope_gates_v1/gates.json').read_text())['passed']
    assert json.loads((data.OUT/'prediction/audit.json').read_text())['passed']
    paths=[data.OUT/'prediction/predictions.csv',data.OUT/'prediction/diagnostics.json',
        data.OUT/'prediction/plan.json',data.OUT/'transport_scope_gates_v1/gates.json',NPZ]
    code=[Path(__file__),ROOT/'src/sandbox/transport_scope_diagnostic.py',
        ROOT/'docs/ML_SANDBOX_TRANSPORT_SCOPE_PHASE29_20261019.md']
    e.write(OUT/'protocol_before.json',dict(created_utc=e.now(),readout_reconstruction_tolerance=1e-6,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths},
        actual_scores_unchanged=True,no_posthoc_domain_filter=True,goal_complete=False))
    profile=published_coherent_profile();cache={};rows=[];errors=[]
    def calculation(tr):
        key=tuple(float(tr[k]) for k in ('free_n_cm3','Nq2_cm3','temperature_K','extra_mobility_cm2Vs','mass_ratio','epsilon_r'))
        if key not in cache:
            cache[key]=reconstruct_transport(*key[:4],mass_ratio=key[4],epsilon_r=key[5],
                Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,order=512)
        return cache[key]
    def append_state(base,dg,expected,growth_C):
        tr=dg.get('coherent_transport')
        if tr is None:
            rows.append(dict(base,numeric_state=False,not_a_latent_transport_state=True));return
        try:
            scope=calculation(tr);v=np.asarray(scope['values'],float).copy()
            delta=float(dg.get('observation_offset_dex',0.))
            v[[0,2]]+=delta
            head=dg.get('transport_head','native');offset=float(dg.get('transport_head_offset_dex',0.))
            if head=='CT_offset':
                v[1]=np.log10(mu_caughey_thomas(10**v[0],300.))+offset
                v[2]=np.log10(E_CHARGE)+v[0]+v[1]
            elif head=='native_offset':v[1:]+=offset
            error=float(np.max(abs(v-np.asarray(expected,float))))
            Tf=dg.get('Tf_K')
            rows.append(dict(base,numeric_state=True,reconstructed_readout_error_dex=error,
                implemented_observation_valid=dg.get('observation_domain_valid'),
                extra_mobility_cm2Vs=tr['extra_mobility_cm2Vs'],free_n_cm3=tr['free_n_cm3'],
                Nq2_cm3=tr['Nq2_cm3'],Tf_K=Tf,
                effective_Tf_above_declared_growth_scenario=bool(Tf is not None and Tf>growth_C+273.15),
                transport_head=head,observation_offset_dex=delta,
                **{k:scope[k] for k in ('fraction_sigma_kl_lt1','fraction_Hall_kl_lt1',
                    'sigma_weighted_kl','threshold_x','threshold_scope','hall_factor')}))
        except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
            errors.append(dict(base,error=repr(exc)));rows.append(dict(base,numeric_state=False,error=repr(exc)))
    frame=read(data.OUT/'prediction/predictions.csv')
    ds={v['row_id']:v['diagnostics'] for v in json.loads((data.OUT/'prediction/diagnostics.json').read_text())}
    for r in frame.itertuples():
        base=dict(scope='outer_prediction',row_id=r.row_id,variant=r.variant,budget=int(r.budget),
            case_id=r.case_id,sample_id=r.sample_id,family=r.family,role=r.role,
            actual_prediction_numeric=bool(r.numeric),actual_prediction_valid=bool(r.valid))
        dg=ds[r.row_id]
        if not r.numeric or not dg:
            rows.append(dict(base,numeric_state=False,not_a_latent_transport_state=True));continue
        append_state(base,dg,[r.predicted,r.predicted_log_mu,r.predicted_log_sigma],r.growth_T_C)
    # Full-data chosen fits are an implementation/scope readback, not a
    # validation sample. Deduplicate scalar heads sharing the same raw fit.
    plans=json.loads((data.OUT/'prediction/plan.json').read_text())['plans']
    full={p['fit_key']:p for p in plans if p['outer_held'] is None}
    raw=np.load(NPZ);pid=raw['cs_ID'].astype(int).ravel();site=raw['cs_num_each_site'].astype(float)
    parent_sites={int(i):site[pid==i].max(axis=0) for i in np.unique(pid)}
    site_invariant=all(np.max(abs(site[pid==i]-site[pid==i][0]))==0 for i in np.unique(pid))
    cases=data.load_cases();inventory_rows=[]
    for key,p in full.items():
        fit=dict(p['minimal_fit'],transport_head='native',transport_head_offset_dex=0.)
        if fit['method'] in ('ActivationProcessPair','DirectMultiRidge','UnitDose','MedianActivation','DoseLine'):continue
        try:mapper=SavedJointModel(fit);failure=None
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:mapper=None;failure=repr(exc)
        for r in cases.itertuples():
            base=dict(scope='full_training_inputs',fit_key=key,method=fit['method'],case_id=r.case_id,
                sample_id=r.sample_id,family=r.family)
            if mapper is None:
                rows.append(dict(base,numeric_state=False,unqualified_fit=True,error=failure));continue
            row=SimpleNamespace(**{k:getattr(r,k) for k in INPUT_NAMES})
            try:
                point=mapper.point(row,require_observation=False)
                append_state(base,point.diagnostics,[point.log_nH,point.log_muH,point.log_sigma],r.growth_T_C)
                engine=(mapper.engine.engine_for(row) if hasattr(mapper.engine,'engine_for')
                    else mapper.engine.engines[row.scenario_pO2_atm])
                layer,solver=engine.layer_for(row);inventory=solver.last_inventory
                sites=np.array([parent_sites[int(i)] for i in layer.original_parent_ids[layer.first]])
                claim=sites.T@inventory.totals/N_SITE
                inventory_rows.append(dict(base,foreign_atom_relative_error=point.diagnostics.get('foreign_atom_relative_error'),
                    max_parent_fraction=point.diagnostics.get('max_parent_fraction'),
                    shared_site_claim_upper_envelope=float(claim.max()),site_claim_above_one=bool(claim.max()>1.),
                    charge_dependent_sites_use_upper_envelope_not_exact_identity=True,
                    actual_input_volume_uniformity_established=False))
            except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
                errors.append(dict(base,error=repr(exc)));inventory_rows.append(dict(base,error=repr(exc)))
    e.csvwrite(OUT/'states.csv',rows);e.csvwrite(OUT/'inventory_readback.csv',inventory_rows)
    numeric=[r for r in rows if r['numeric_state']]
    maximum=max((r['reconstructed_readout_error_dex'] for r in numeric),default=np.inf)
    result=dict(passed=bool(numeric and not errors and maximum<=1e-6),created_utc=e.now(),
        recorded_rows=len(rows),numeric_transport_states=len(numeric),unique_transport_moment_reconstructions=len(cache),
        independent_readout_maximum_error_dex=maximum,errors=errors,
        outer_prediction_rows=len(frame),full_input_inventory_rows=len(inventory_rows),
        parent_site_counts_charge_invariant=bool(site_invariant),
        majority_current_weight_kl_lt1=sum(r['fraction_sigma_kl_lt1']>.5 for r in numeric),
        unique_numerical_transport_states_majority_kl_lt1=sum(r['fraction_sigma_kl_lt1']>.5 for r in cache.values()),
        diagnostic_state_counts_are_not_independent_material_samples=True,
        effective_Tf_above_declared_growth_scenario=sum(r['effective_Tf_above_declared_growth_scenario'] for r in numeric),
        states_without_latent_transport=sum(not r['numeric_state'] for r in rows),
        actual_material_Born_validity_established=False,
        known_C_H_and_process_scope_unresolved=True,posthoc_removed_prediction_rows=0,
        model_and_scores_unchanged=True,not_independent_material_validation=True,goal_complete=False)
    e.write(OUT/'audit.json',result)
    print('Physical scope29',result['passed'],'states',len(numeric),'max',maximum,flush=True)
    if not result['passed']:raise RuntimeError('Preserve scope/reconstruction failure;do not alter scores or tolerance')


if __name__=='__main__':main()
