"""Check catalogue stoichiometry against a visually verified primary table."""
from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from sandbox.kroger_db import load,NPZ

OUT=c.BASE/'multidonor_catalogue_audit_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    pdf=c.BASE/'literature_v3/lany2018_nlr.pdf'
    page=c.BASE/'literature_v3/lany2018_tableII_page4.png'
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_sha256=c.e.sha(__file__),
        input_hashes=c.file_hashes([NPZ,pdf,page]),primary_DOI='10.1063/1.5019938',
        primary_table='TableII,PDFpage4,visuallyread;bindingnegative relative to isolated species',
        catalogue_not_modified=True,no_new_formation_energy_labels_ingested=True,goal_complete=False))
    db=load();rows=[];summary=[]
    binding={'Si':[-.94,-1.60,-1.83],'Sn':[-1.17,-1.79,-2.05]}
    for element in ['Si','Sn']:
        ix=db.col(element);selected=db.dm[:,ix]>0
        summary.append(dict(element=element,maximum_atoms_per_current_parent=int(db.dm[selected,ix].max()),
            unique_Ga_O_dopant_stoichiometries=np.unique(db.dm[selected][:,[0,1,ix]],axis=0).tolist()))
        for k in [1,2,3]:
            dm=np.zeros(len(db.elements),int);dm[0]=-1-k;dm[ix]=k
            matching=np.all(db.dm==dm,axis=1)
            rows.append(dict(element=element,dopants_per_complex=k,delta_Ga=-1-k,delta_O=0,
                expected_n_type_charge=k-3,current_matching_charge_rows=int(sum(matching)),
                current_matching_original_parents=int(len(np.unique(db.cs_ID[matching]))),
                current_available_charges=sorted(set(db.charge[matching].tolist())),
                primary_binding_eV=binding[element][k-1],
                primary_binding_is_not_capture_or_migration_barrier=True,
                primary_reference='isolated VGa3- and lowest isolated DGa+,DFT with GW corrections',
                primary_formation_reference='CBM at0;elementalstandardstates0K',
                no_direct_cross_method_energy_replacement=True))
    c.e.csvwrite(OUT/'complex_stoichiometry.csv',rows)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'audit.json',dict(created_utc=c.e.now(),passed=True,summary=summary,
        Si_multidonor_stoichiometries_absent=all(r['current_matching_charge_rows']==0 for r in rows if r['element']=='Si' and r['dopants_per_complex']>1),
        scope='current loaded NPZ,not all physically possible defects',
        prior_neutral_Si_probe_only_current_catalogue=True,
        adding_states_requires_formation_reference_method_multiplicity_and_charge_scope_audit=True,
        current_training_and_prediction_artifacts_unchanged=True,new_fits=0,goal_complete=False))
    for r in summary:print(r,flush=True)


if __name__=='__main__':main()
