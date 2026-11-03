"""Read actual new full-model states without filtering or changing any score."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.feasible_property_predictor_v2 import FeasiblePropertyPredictorV2 as ContextPropertyPredictor
from sandbox.coherent_fermi import published_coherent_profile,reference
from sandbox.coherent_transport import quadrature
from sandbox.kroger_db import NPZ,N_SITE
from sandbox.ionized_transport import E_CHARGE,EPSILON0,K_B_J,HBAR,M_E,coulomb_log
from sandbox.context_saved_model import INPUT_NAMES
import phase27_registry as t
import phase2_energy as e


def transport_scope(n,NI):
    profile=published_coherent_profile();T=300.;m=profile.electron_mass_ratio*M_E;eps=10.2*EPSILON0;kT=K_B_J*T
    eta=reference().eta_from_logF(np.log(n/profile.Nc_ref));_,g,_=reference().log(eta)
    nodes,w=quadrature(512);lo=np.sqrt(max(0.,eta-50.));hi=np.sqrt(max(60.,eta+50.))
    tt=(nodes+1.)*(hi-lo)/2.+lo;x=tt*tt;w=w*(hi-lo)/2.
    p=w*2*tt**4*expit(eta-x)*expit(x-eta)/expit(eta);p/=p.sum()
    qs2=E_CHARGE**2*n*g*1e6/(eps*kT)
    b=8*m*kT*x/(HBAR**2*qs2)
    pref=NI*1e6*E_CHARGE**4/(16*np.sqrt(2)*np.pi*eps**2*np.sqrt(m)*kT**1.5)
    tau=1/(pref*x**(-1.5)*coulomb_log(b)+E_CHARGE/(m*205e-4))
    kl=2*x*kT*tau/HBAR;sw=p*tau;sw/=sw.sum();hw=p*tau**2;hw/=hw.sum()
    mud=E_CHARGE/m*(p@tau)*1e4;rh=(p@(tau*tau))/(p@tau)**2
    return dict(reconstructed_log_mu=float(np.log10(mud*rh)),fraction_sigma_kl_lt1=float(sw[kl<1].sum()),
        fraction_Hall_kl_lt1=float(hw[kl<1].sum()),weighted_kl=float(sw@kl))


def main():
    out=t.OUT/'physical_scope';bundle=t.OUT/'delivery/predictor_bundle';model=ContextPropertyPredictor(bundle)
    raw=np.load(NPZ);pid=raw['cs_ID'].astype(int).ravel();sites=raw['cs_num_each_site'].astype(float)
    parent_sites={};consistent=True
    for i in np.unique(pid):
        a=sites[pid==i];consistent=consistent and bool(np.max(abs(a-a[0]))==0)
        parent_sites[int(i)]=a.max(axis=0)
    rows=[];maximum=0.;cache={};qualified_expected=0
    for name,spec in model.manifest['presets'].items():
        if spec.get('model_type')!='charge_feasible' or spec['context'] not in ('DOI::full','cluster::full') or spec['source_kind']!='offset':continue
        if not spec['method'].startswith('Feasible'):continue
        qualified_expected+=len(t.TABLE)*bool(spec['serving_qualified'])
        saved=None;qualification_error=None
        try:saved,_=model.model(name)
        except (ValueError,RuntimeError) as exc:qualification_error=repr(exc)
        for r in t.TABLE.itertuples():
            row=SimpleNamespace(**{k:getattr(r,k) for k in INPUT_NAMES});record=dict(preset=name,sample_id=r.sample_id,family=r.family,element=r.element,
                C_cm3=float(r.incorporated_cm3),nominal_growth_C=float(r.growth_T_C),method=spec['method'],actual_evaluation_method=spec['actual_evaluation_method'],
                serving_qualified=bool(spec['serving_qualified']),reference_fallback=bool(spec['explicit_reference_fallback']))
            if saved is None:
                record.update(numeric=False,not_evaluated_due_to_model_qualification=True,error=qualification_error);rows.append(record);continue
            try:
                p=saved.engine.point(row,saved.fit['parameters'],require_observation=False)
                engine=saved.engine.engine_for(row) if hasattr(saved.engine,'engine_for') else saved.engine.engines[row.scenario_pO2_atm]
                layer,solver=engine.layer_for(row);inventory=solver.last_inventory;cold=solver.last_prediction
                cp=np.array([parent_sites[int(i)] for i in layer.original_parent_ids[layer.first]])
                fractions=cp.T@inventory.totals/N_SITE
                n=p.diagnostics['free_n_cm3'];NI=p.diagnostics['coherent_transport']['Nq2_cm3'];key=(float(n),float(NI))
                if key not in cache:cache[key]=transport_scope(*key)
                sc=cache[key];difference=abs(sc['reconstructed_log_mu']-p.transport_values[0]);maximum=max(maximum,difference)
                car=saved.engine.profile.carriers(cold.ef_eV,300.)[0]
                record.update(numeric=True,isolated_atom_fraction_after=inventory.diagnostics.get('isolated_atom_fraction_after'),
                    excess_share_logit=inventory.diagnostics.get('excess_share_logit'),critical_free_fraction=inventory.diagnostics.get('critical_free_fraction'),implemented_observation_valid=p.diagnostics['observation_domain_valid'],
                    Tf_K=p.diagnostics['Tf_K'],Tf_above_growth=bool(p.diagnostics['Tf_K']>r.growth_T_C+273.15),
                    max_individual_parent_fraction=p.diagnostics['max_parent_fraction'],shared_site_claim_max=float(fractions.max()),
                    shared_site_claims=fractions.tolist(),site_claim_exceeds_one=bool(fractions.max()>1),
                    foreign_atom_error=p.diagnostics['foreign_atom_relative_error'],all_holes_to_n=float(car[1:].sum()/car[0]),
                    implemented_n_gt_p=bool(car[0]>car[1]),isolated_localized_fraction=p.diagnostics['localized_fraction'],
                    mu_reconstruction_error=difference,**sc)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:record.update(numeric=False,error=repr(exc))
            rows.append(record)
    assert len(rows)==198
    e.csvwrite(out/'states.csv',rows)
    numeric_states=sum(r['numeric'] for r in rows)
    value=dict(passed=bool(maximum<1e-6 and qualified_expected>0 and numeric_states==qualified_expected),states=len(rows),numeric_states=numeric_states,
        expected_qualified_model_points=qualified_expected,
        maximum_independent_mu_reconstruction_error=maximum,site_counts_charge_invariant=consistent,
        unqualified_model_points_not_evaluated=sum(r.get('not_evaluated_due_to_model_qualification',False) for r in rows),
        numeric_implemented_invalid_states=sum(r.get('numeric',False) and not r.get('implemented_observation_valid',False) for r in rows),
        majority_current_weight_kl_lt1=sum(r.get('fraction_sigma_kl_lt1',0)>.5 for r in rows),
        shared_site_claim_exceeds_one=sum(r.get('site_claim_exceeds_one',False) for r in rows),
        effective_Tf_above_growth=sum(r.get('Tf_above_growth',False) for r in rows),
        actual_material_Born_validity_established=False,all_original_scores_unchanged=True,removed_rows=0,
        scope='threenewphysicalfamilies x2fullcontexts x33knowninputs;unqualifiedmodels refused,not heldoutaccuracy;charge-dependent sitecounts use max envelope;kl/siteclaims not Born/exclusion certificates',
        new_fits=0,new_material_measurements=0,goal_complete=False,code_sha256=e.sha(Path(__file__)))
    e.write(out/'audit.json',value)
    print('Phase27 physical scope readback',value,flush=True)
    if not value['passed']:raise RuntimeError('Retain failed transport reconstruction diagnostic')


if __name__=='__main__':main()
