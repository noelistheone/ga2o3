"""Attribute the restricted-temperature deficit before choosing a stock learner."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sandbox.thermal_coherent_model import ThermalCoherentHallModel
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.coherent_transport import fermi_transport
from sandbox.thermo import KB_EV
import phase16_refinement as prior
import phase2_energy as energy

OUT=energy.BASE/'phase17_inventory_learning'


def fixed_readout(layer,p,totals):
    T=10**p[layer.iobs];Nd=10**p[1];H,_,_,profile=layer._reference(T,p);s,_=layer._entropy(T)
    A=H-3*KB_EV*T*s*layer.db.sum_dm;beta=1/(KB_EV*T);ell,_=layer.fraction(p)
    def population(ef):
        logw=-(A+layer.q*ef)*beta+layer.logg
        maxima=np.full(layer.db.nD,-np.inf);np.maximum.at(maxima,layer.parent,logw)
        v=np.exp(logw-maxima[layer.parent]);pi=v/np.bincount(layer.parent,weights=v,minlength=layer.db.nD)[layer.parent]
        for plus,neutral in layer.pairs:
            d=logw[neutral]-logw[plus];pi[neutral]=ell*expit(d);pi[plus]=(1-ell)+ell*expit(-d)
        return totals[layer.parent]*pi
    def Q(ef):
        N=population(ef);c=profile.carriers(ef,T)[0]
        P=np.maximum(layer.q,0)@N+c[1:].sum()+Nd;M=np.maximum(-layer.q,0)@N+c[0]
        return (P-M)/max(P,M,1.)
    ef=brentq(Q,-5,12,xtol=2e-14,rtol=1e-15);N=population(ef);car=profile.carriers(ef,T)[0]
    q2=float((layer.q**2)@N+Nd)
    readout=fermi_transport(np.log10(car[0]),np.log10(q2),T,Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,
        mass_ratio=profile.electron_mass_ratio,extra_mobility_cm2Vs=205.)
    return readout,N,ef


def main():
    study=prior.study;bank=prior.composite();base=study.OUT
    protocol=dict(code_sha256=energy.sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (energy.BASE/'artifact_manifest_phase16.json',base/'prediction_protocol.json',
            base/'thermal_audit/specimens.csv',energy.BASE/'dataset_v1.csv')},
        purpose='audit native versus foreign-containing compensation before implementing proposed native survival learner',
        models=['Thermal_original_RT_record_cap_g0','Thermal_binding_joint_record_cap_g0'],
        counterfactual_native_factors=[1.,.1,0.],scope='all29existingRT samples,sourceexcludedfinalparameters;no fit/no newmeasurement;not globalreachabilityproof',
        goal_complete=False)
    pp=OUT/'inventory_audit/protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==energy.clean(protocol)
    else:energy.write(pp,protocol)
    parent_rows=[];state_rows=[];probes=[];errors=[]
    for method in protocol['models']:
        model=ThermalCoherentHallModel(study.STRUCTURES[method],profile=published_coherent_profile(),pO2_atm=1e-5)
        for row in study.core.PRED[study.core.PRED.kind.eq('RT')].itertuples():
            fit=next(f for f in bank.values() if f['method']==method and len(f['train_families'])==4 and row.family not in f['train_families'])
            point=model.point(row,fit['parameters']);w=fit['parameters'];elements=(row.element,'H') if row.H_known else (row.element,)
            layer=model.layers[elements];state=layer.last_state;N=state['cold_charge'].copy();totals=state['totals'].copy()
            native=np.all(layer.db.dm[layer.first,2:]==0,axis=1)
            positive=np.bincount(layer.parent,weights=np.maximum(layer.q,0)*N,minlength=layer.db.nD)
            negative=np.bincount(layer.parent,weights=np.maximum(-layer.q,0)*N,minlength=layer.db.nD)
            q2=np.bincount(layer.parent,weights=layer.q**2*N,minlength=layer.db.nD)
            for i in range(layer.db.nD):
                if totals[i]<1.:continue
                parent_rows.append(dict(method=method,sample_id=row.sample_id,family=row.family,parent_id=int(layer.original_parent_ids[layer.first[i]]),
                    native=bool(native[i]),total_cm3=float(totals[i]),positive_charge_cm3=float(positive[i]),negative_charge_cm3=float(negative[i]),
                    q2_cm3=float(q2[i]),stoichiometry=json.dumps(dict(zip(layer.db.elements,layer.db.dm[layer.first[i]].astype(int).tolist())))))
            state_rows.append(dict(method=method,sample_id=row.sample_id,family=row.family,observed_log_nH=row.target,
                predicted_log_nH=float(point.values[0]),negative_native=float(negative[native].sum()),negative_foreign=float(negative[~native].sum()),
                fraction_negative_from_native=float(negative[native].sum()/negative.sum()),positive_native=float(positive[native].sum()),
                positive_foreign=float(positive[~native].sum()),q2_native=float(q2[native].sum()),q2_foreign=float(q2[~native].sum())))
            p=layer.parameters(T=point.diagnostics['Tf_K'],Nd=point.diagnostics['Nd_cm3'],weights=[0.],pO2=1e-5,
                doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),Tobs=row.Tobs_K,
                binding_eV=10**w[model.iE] if model.iE is not None else .0366,screen_s=w[model.iscreen] if model.iscreen is not None else 0.)
            for factor in protocol['counterfactual_native_factors']:
                c=totals.copy();c[native]*=factor;r,_,ef=fixed_readout(layer,p,c)
                if factor==1.:errors.append(float(max(abs(r.values-np.r_[point.values[0],point.transport_values]))))
                probes.append(dict(method=method,sample_id=row.sample_id,family=row.family,native_factor=factor,
                    observed_log_nH=row.target,predicted_log_nH=float(r.values[0]),predicted_log_mu=float(r.values[1]),ef_eV=ef))
    assert len(state_rows)==58 and len(probes)==174 and max(errors)<1e-6
    dest=OUT/'inventory_audit';pd.DataFrame(parent_rows).to_csv(dest/'parent_contributions.csv',index=False)
    pd.DataFrame(state_rows).to_csv(dest/'state_contributions.csv',index=False);pd.DataFrame(probes).to_csv(dest/'native_counterfactuals.csv',index=False)
    states=pd.DataFrame(state_rows);probe=pd.DataFrame(probes)
    summary=probe.assign(error=abs(probe.predicted_log_nH-probe.observed_log_nH)).groupby(['method','native_factor','family']).error.mean().groupby(['method','native_factor']).mean().reset_index()
    summary.to_csv(dest/'counterfactual_summary.csv',index=False)
    energy.write(dest/'contract_audit.json',dict(passed=True,RT_states=58,existing_specimens=29,DOIs=5,counterfactual_states=174,
        maximum_reference_replay_error=max(errors),native_negative_fraction_min=float(states.fraction_negative_from_native.min()),
        native_negative_fraction_max=float(states.fraction_negative_from_native.max()),new_fits=0,removed_samples=0,goal_complete=False))
    print(states.groupby('method').fraction_negative_from_native.agg(['min','max','mean']).to_string());print(summary.to_string(index=False))


if __name__=='__main__':main()
