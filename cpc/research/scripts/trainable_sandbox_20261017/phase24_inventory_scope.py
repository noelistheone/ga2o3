"""Audit dose-dependent native/complex inventory and learned allocation coverage at frozen parameters."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.special import expit
from sandbox.functional_predictor import DopedPropertyPredictor
from sandbox.kroger_db import load
from sandbox.relative_inventory_model import L0,L1
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';DEST=OUT/'inventory_scope'


def main():
    bundle=BASE/'phase21_curve_response/delivery/predictor_bundle_v8'
    protocol=dict(created_utc=e.now(),code_sha256=e.sha(Path(__file__)),property_manifest_sha256=e.sha(bundle/'manifest.json'),
        prior_probe='exploratory3Si inputs showedsame-Ga structuralpool negligible;do not assume a missingstructuralpool cause',
        purpose='test whether inheritedGa-vacancy availability grows withdose andwhich existinglearnedcoordinates canchangeit',
        elements=['Si','Sn'],doses=[1e17,1e18,1e19,3e19,1e20,3e20],source_shifts=[0.,6.,-6.],H='unreportedzero scenario',
        observations='all300K/noHalltargets,modeledchargeinventory notmeasureddefectdensities',
        scope='globaloriginal full5 frozenpointstates;not a new predictivefit or real pathway',goal_complete=False)
    e.write(DEST/'protocol.json',protocol);predictor=DopedPropertyPredictor(load(),bundle);states=[];parents=[]
    for element in protocol['elements']:
        for C in protocol['doses']:
            q=predictor.predict(element,C,method='OriginalHall',growth_T_C=525.,source_inputs=[])
            spec=predictor.manifest['variants']['OriginalHall'];saved=predictor.models[(spec['model_key'],1e-5)];model=saved.model;w=np.array(saved.record['parameters'])
            row=predictor.parse(element,dict(incorporated_cm3=C,growth_T_C=525.,observation_T_K=300.));layer,solver=model.layer_for(row)
            Tf=10**(L0+(L1-L0)*expit(w[0]));p=layer.parameters(T=Tf,Nd=10**w[1],weights=[0.],doses=(C,),pO2=1e-5,Tobs=300.,binding_eV=.0366,screen_s=0.)
            ref=solver.reference(p);dm=layer.db.dm[layer.first];k=solver.dopant_count;ix=layer.db.col(element);other=[i for i in range(2,len(layer.db.elements)) if i!=ix]
            pure=(dm[:,1]==0)&np.all(dm[:,other]==0,axis=1);v=np.where(pure,np.maximum(-dm[:,0]-k,0),0.)
            Nv=float(v@ref.totals);dNv=float(v@ref.jacobian[:,layer.idose]);elastic=dNv/(Nv*np.log(10.))
            for shift in protocol['source_shifts']:
                a=saved.point(row,source_kind='field',shift=shift,require_observation=False);r=solver.last_prediction;stock=solver.last_inventory.totals;N=r.populations
                neg=np.bincount(layer.parent,weights=N*np.maximum(-layer.q,0),minlength=layer.db.nD)
                q2=np.bincount(layer.parent,weights=N*layer.q**2,minlength=layer.db.nD);den=float(neg.sum());native=solver.native
                same=pure&(k>0)&(dm[:,0]==-k);positive_budget=pure&(v>0)
                states.append(dict(element=element,incorporated_cm3=C,source_shift=shift,log10_nH=a.values[0],log10_muH=a.transport_values[0],log10_sigma=a.transport_values[1],
                    material_readout_valid=a.diagnostics['observation_domain_valid'],continued_state_only=not a.diagnostics['observation_domain_valid'],
                    cold_EF_eV=r.ef_eV,modeled_negative_charge_cm3=den,native_negative_fraction=float(neg[native].sum()/den),
                    vacancy_complex_negative_fraction=float(neg[solver.complex].sum()/den),sameGa_foreign_negative_fraction=float(neg[same].sum()/den),
                    positive_vacancy_budget_negative_fraction=float(neg[positive_budget].sum()/den),
                    modeled_reference_vacancy_equivalent_cm3=Nv,reference_vacancy_dose_elasticity=elastic,
                    actual_shifted_vacancy_equivalent_cm3=float(v@stock),native_stock_fixed_under_current_source_shift=True,
                    atom_budget_error=a.diagnostics['foreign_atom_relative_error'],charge_budget_error=a.diagnostics['charge_relative_residual'],
                    scope='conditional modelinventory;not independently measured Ga vacancy density or observedmaterialproperty'))
                for j,parent in enumerate(layer.original_parent_ids[layer.first]):
                    parents.append(dict(element=element,incorporated_cm3=C,source_shift=shift,parent_id=int(parent),parent_stock_cm3=float(stock[j]),
                        Ga_atom_change=int(dm[j,0]),O_atom_change=int(dm[j,1]),foreign_count=float(k[j]),positive_vacancy_equivalent=float(v[j]),
                        native=bool(native[j]),current_learned_complex_mask=bool(solver.complex[j]),sameGa_foreign=bool(same[j]),
                        negative_charge_cm3=float(neg[j]),negative_charge_fraction=float(neg[j]/den),charge_squared_cm3=float(q2[j])))
    e.csvwrite(DEST/'states.csv',states);e.csvwrite(DEST/'parents.csv',parents)
    # These are bookkeeping checks, not a criterion for the next hypothesis winning.
    good=all(r['atom_budget_error']<3e-8 and r['charge_budget_error']<2e-8 for r in states)
    e.write(DEST/'audit.json',dict(passed=good,states=len(states),parent_rows=len(parents),observation_valid=sum(r['material_readout_valid'] for r in states),
        invalid_continuations_retained=sum(r['continued_state_only'] for r in states),no_Hall_targets_used=True,new_fitting=0,
        structuralpool_dominance_claim=False,goal_complete=False))
    print('frozeninventory scope',len(states),'states;allcontinuationsretained;',sum(r['continued_state_only'] for r in states),'outside materialreadout',flush=True)
    for r in states:
        if r['source_shift']==0:print(r['element'],r['incorporated_cm3'],'native',r['native_negative_fraction'],'complex',r['vacancy_complex_negative_fraction'],'sameGa',r['sameGa_foreign_negative_fraction'],'Nv',r['modeled_reference_vacancy_equivalent_cm3'],'elastic',r['reference_vacancy_dose_elasticity'],flush=True)
    assert good


if __name__=='__main__':main()
