"""Frozen-state audit of native/complex source coordinates before training.

The cap transform below is an exact effective-temperature coordinate transform,
not a supplied or inferred film temperature history. Earlier code is untouched.
"""
from pathlib import Path
from types import SimpleNamespace
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.special import expit
from csv_io_v2 import read_result_csv as read
from sandbox.inventory_coherent_model import InventoryCoherentModel,FIXED_TEMPERATURE_FRACTION
from sandbox.coherent_fermi import CoherentMaterialProfile
from sandbox.property_predictor_v5 import DopedPropertyPredictor
from sandbox.kroger_db import load
import phase16_prediction as inputs
import phase2_energy as energy

BASE=energy.BASE;OUT=BASE/'phase20_source_inventory';REFERENCE=BASE/'phase19_delivery/predictor_bundle_v5/models/HallExtra_original_RT.json'


def main():
    ref=json.loads(REFERENCE.read_text());w=np.array(ref['parameters']);profile=CoherentMaterialProfile.from_dict(ref['profile'])
    Tf=10**(np.log10(600.)+(np.log10(1950.)-np.log10(600.))*expit(w[0]))
    effective_cap=600.*(Tf/600.)**(1/FIXED_TEMPERATURE_FRACTION)
    protocol=dict(created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),
        previous_goal_turn='progress:fullsource delivery and frozen Leedy/oxygen prediction evidence completed',
        stage='diagnosticbefore selecting one predictive source-inference experiment',
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (REFERENCE,BASE/'dataset_v1.csv',BASE/'phase19_delivery/leedy2018_v2/samples.csv',BASE/'artifact_manifest_phase19.json')},
        base='full5trained HallExtra_original_RT;allweights frozen',Tf_K=Tf,
        mathematical_cap_K=effective_cap,cap_interpretation='algebraic inversecoordinate only;not measuredgrowth or filmTmax',
        source_scan=[0.,.01,.03,.1,.3,1.,3.,6.],coordinates='z>=0,inventorysurvival10^-z,nativeorcomplex separately',
        samples='old29RT plus newLeedy5;no pooling fortraining;allseenretrospective',
        endpoint='nH,own-state mu/sigma,doseelasticity andchargecontributions;no physicalcauseidentification')
    p=OUT/'probe_protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        for key in ('code_sha256','input_hashes'):assert old[key]==protocol[key]
    else:energy.write(p,protocol)
    old=inputs.PRED[inputs.PRED.kind.eq('RT')].copy();old['cohort']='old29'
    new=read(BASE/'phase19_delivery/leedy2018_v2/samples.csv');new['cohort']='Leedy2018'
    new['Tobs_K']=300.;new['target']=np.log10(new.observed_nH_cm3);new['family']='10.1063/1.5047214'
    data=pd.concat([old,new],ignore_index=True);db=load();models={};rows=[];contributions=[];parity=[]
    collection=DopedPropertyPredictor(db,BASE/'phase19_delivery/predictor_bundle_v5')
    for row in data.itertuples():
        values=row._asdict();values['thermal_cap_K']=effective_cap;r=SimpleNamespace(**values)
        base=collection.predict(r.element,float(r.incorporated_cm3),method='HallExtra_original_RT',
            H_known=bool(r.H_known),H_cm3=float(r.H_cm3),growth_T_C=float(r.growth_T_C))
        for kind in ('native','complex'):
            if kind not in models:
                definition=dict(fraction='zero',training='RT',readout='ionized_extra',inventory_kind=kind,relation='constant')
                models[kind]=InventoryCoherentModel(definition,profile=profile,pO2_atm=1e-5,db=db)
            model=models[kind]
            for z in protocol['source_scan']:
                try:
                    pred=model.point(r,[w[1],0.],source_kind=kind,source_shift=z)
                    rows.append(dict(cohort=r.cohort,sample_id=r.sample_id,family=r.family,element=r.element,
                        incorporated_cm3=r.incorporated_cm3,coordinate=kind,z=z,survival=10**(-z),
                        observed_log_nH=r.target,predicted_log_nH=float(pred.values[0]),
                        log_mu=float(pred.transport_values[0]),log_sigma=float(pred.transport_values[1]),
                        dlog_nH_dlogC=float(pred.input_jacobian[0,0]),dnH_dz=float(pred.source_jacobian[0]),
                        numeric=True,root_residual=pred.diagnostics['root_residual_log'],
                        foreign_atom_error=pred.diagnostics['foreign_atom_relative_error']))
                    if z!=0:continue
                    parity.append(abs(pred.values[0]-base['log10_nH_cm3']))
                    elements=(r.element,'H') if r.H_known else (r.element,);solver=model.solvers[elements]
                    state=solver.last_prediction;l=solver.layer;N=state.populations
                    masks=dict(native=solver.native,complex=solver.complex,isolated=solver.receivers,
                        other=~(solver.native|solver.complex|solver.receivers))
                    total_negative=float(np.maximum(-l.q,0.)@N)
                    for label,mask in masks.items():
                        charge_mask=mask[l.parent];neg=float(np.maximum(-l.q,0.)[charge_mask]@N[charge_mask]);pos=float(np.maximum(l.q,0.)[charge_mask]@N[charge_mask])
                        contributions.append(dict(cohort=r.cohort,sample_id=r.sample_id,element=r.element,
                            group=label,negative_charge_cm3=neg,positive_charge_cm3=pos,
                            fraction_negative=neg/total_negative if total_negative>0 else None,
                            parent_total_cm3=float(solver.last_inventory.totals[mask].sum()),
                            observed_log_nH=r.target,base_log_nH=float(pred.values[0])))
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    rows.append(dict(cohort=r.cohort,sample_id=r.sample_id,coordinate=kind,z=z,numeric=False,error=repr(exc)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'frozen_scans.csv',index=False)
    pd.DataFrame(contributions).to_csv(OUT/'charge_contributions.csv',index=False)
    audit=dict(passed=max(parity)<2e-6 and result.numeric.all(),zero_replays=len(parity),maximum_zero_error=max(parity),
        scans=len(rows),numeric_scans=int(result.numeric.sum()),old_specimens=29,new_source_specimens=5,
        weights_trained=False,cap_is_mathematical_transform=True,source_coordinate_validation_previously_unrun=True)
    energy.write(OUT/'probe_audit.json',audit);print(json.dumps(audit,ensure_ascii=False),flush=True)
    q=pd.DataFrame(contributions);q=q[q.cohort.eq('Leedy2018')]
    print(q[['sample_id','group','fraction_negative']].to_string(index=False),flush=True)
    v=result[result.cohort.eq('Leedy2018')]
    print(v.groupby(['coordinate','z']).agg(n=('sample_id','size'),mean_pred=('predicted_log_nH','mean'),mean_slope=('dlog_nH_dlogC','mean')).to_string(),flush=True)
    assert audit['passed']


if __name__=='__main__':main()
