"""Gate the new conservative map/IFT against unchanged physical references."""
from pathlib import Path
from types import SimpleNamespace
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from csv_io_v2 import read_result_csv as read
from sandbox.relative_inventory_model import RelativeInventoryModel
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.property_predictor_v5 import DopedPropertyPredictor
from sandbox.kroger_db import load
import phase16_prediction as inputs
import phase2_energy as energy

OUT=energy.BASE/'phase20_source_inventory'


def main():
    checks=[];db=load();profile=published_coherent_profile();models={k:RelativeInventoryModel(dict(field=k),profile=profile,pO2_atm=1e-5,db=db) for k in ('inventory','output')}
    rt=inputs.PRED[inputs.PRED.kind.eq('RT')];ref=json.loads((energy.BASE/'phase19_delivery/predictor_bundle_v5/models/HallExtra_original_RT.json').read_text())
    old=DopedPropertyPredictor(db,energy.BASE/'phase19_delivery/predictor_bundle_v5');w0=np.r_[ref['parameters'],0.]
    for row in rt.itertuples():
        reference=old.predict(row.element,row.incorporated_cm3,method='HallExtra_original_RT',H_known=bool(row.H_known),H_cm3=float(row.H_cm3))
        for kind,model in models.items():
            p=model.point(row,w0);error=max(abs(p.values[0]-reference['log10_nH_cm3']),abs(p.transport_values[0]-reference['log10_muH_cm2Vs']))
            checks.append(dict(test='zero_reference',sample_id=row.sample_id,kind=kind,error=float(error),passed=bool(error<2e-6)))
    representative=[rt[rt.element.eq('Sn')].iloc[0],rt[rt.element.eq('Si')&~rt.H_known].iloc[-1],rt[rt.H_known].iloc[0]]
    for row0 in representative:
        for T in (300.,150.):
            row=SimpleNamespace(**row0.to_dict());row.Tobs_K=T
            for kind,model in models.items():
                for h in (0.,.1):
                    w=np.array([1.3,16.,h]);p=model.point(row,w);step=2e-4
                    for k in range(3):
                        a=w.copy();b=w.copy();a[k]+=step;b[k]-=step
                        # A numerical continuation is used only if a finite
                        # difference crosses the Hall detection boundary.
                        pa=model.point(row,a,require_observation=False);pb=model.point(row,b,require_observation=False)
                        fd=(np.r_[pa.values,pa.transport_values]-np.r_[pb.values,pb.transport_values])/(2*step)
                        J=np.r_[p.jacobian[:,k],p.transport_jacobian[:,k]]
                        e=float(np.max(np.abs(fd-J)/(1+np.abs(fd))))
                        checks.append(dict(test='IFT',kind=kind,T=T,sample_id=row.sample_id,column=k,error=e,passed=e<3e-5))
                    for col,key in ((0,'incorporated_cm3'),(1,'Tobs_K')):
                        a=SimpleNamespace(**vars(row));b=SimpleNamespace(**vars(row))
                        setattr(a,key,getattr(a,key)*10**step);setattr(b,key,getattr(b,key)*10**(-step))
                        pa=model.point(a,w,require_observation=False);pb=model.point(b,w,require_observation=False)
                        fd=(np.r_[pa.values,pa.transport_values]-np.r_[pb.values,pb.transport_values])/(2*step)
                        J=np.r_[p.input_jacobian[:,col],p.transport_input_jacobian[:,col]]
                        e=float(np.max(np.abs(fd-J)/(1+np.abs(fd))))
                        checks.append(dict(test='inputIFT',kind=kind,T=T,sample_id=row.sample_id,column=key,error=e,passed=e<3e-5))
                    source='field' if kind=='inventory' else 'offset'
                    pa=model.point(row,w,source_kind=source,source_shift=step,require_observation=False)
                    pb=model.point(row,w,source_kind=source,source_shift=-step,require_observation=False)
                    fd=(np.r_[pa.values,pa.transport_values]-np.r_[pb.values,pb.transport_values])/(2*step)
                    J=np.r_[p.source_jacobian,p.transport_source_jacobian]
                    # p has source_kind=none, whose derivative is identically
                    # zero; evaluate the active coordinate at z=0 explicitly.
                    active=model.point(row,w,source_kind=source,source_shift=0.)
                    J=np.r_[active.source_jacobian,active.transport_source_jacobian]
                    e=float(np.max(np.abs(fd-J)/(1+np.abs(fd))))
                    checks.append(dict(test='sourceIFT',kind=kind,T=T,sample_id=row.sample_id,error=e,passed=e<3e-5))
                    dg=p.diagnostics
                    checks.append(dict(test='foreignatoms',kind=kind,T=T,error=float(dg['foreign_atom_relative_error']),passed=dg['foreign_atom_relative_error']<3e-8))
                    checks.append(dict(test='chargeprobability',kind=kind,T=T,error=float(dg['parent_probability_error']),passed=dg['parent_probability_error']<2e-12))
                    if kind=='inventory':
                        elements=(row.element,'H') if row.H_known else (row.element,);solver=model.solvers[elements]
                        # Check invariant native inventory and the exact signed
                        # atom-share odds identity from separately saved states.
                        m=model.point(row,w);before=m.diagnostics['isolated_atom_fraction_before'];after=m.diagnostics['isolated_atom_fraction_after']
                        residual=np.log10(after/(1-after))-np.log10(before/(1-before))-h
                        checks.append(dict(test='relative_odds_identity',T=T,error=abs(float(residual)),passed=abs(residual)<1e-9))
    paths=[ROOT/'src/sandbox/relative_inventory.py',ROOT/'src/sandbox/relative_inventory_model.py',Path(__file__)]
    payload=dict(passed=bool(all(c['passed'] for c in checks)),checks=len(checks),maximum_error=max(c['error'] for c in checks),
        rows=checks,code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},
        scope='conservation/probability/reference/derivative gates;not new material accuracy')
    energy.write(OUT/'numerical_gates.json',payload);print(json.dumps(energy.clean(payload|{'rows':None}),ensure_ascii=False),flush=True)
    assert payload['passed']


if __name__=='__main__':main()
