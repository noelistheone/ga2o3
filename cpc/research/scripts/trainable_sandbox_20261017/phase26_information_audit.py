"""Count supplied energy entries and disclose separate input-information budgets."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.kroger_db import load,NPZ
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.relative_inventory_model import RelativeInventoryModel
from csv_io_v2 import read_result_csv as read
import phase26_report as report
import phase2_energy as e

OUT=e.BASE/'phase26_pool_share/information_audit'


def main():
    db=load();table=read(e.BASE/'phase25_shared_process/data/conditions33.csv');cohorts=[];checks=[]
    def check(name,x):checks.append(dict(name=name,error=int(x),passed=bool(x==0)))
    for (element,Hknown),group in table.groupby(['element','H_known']):
        elements=(element,'H') if Hknown else (element,)
        other=[j for j in range(2,len(db.elements)) if db.elements[j] not in elements]
        keep=np.all(db.dm[:,other]==0,axis=1);native=np.all(db.dm[:,2:]==0,axis=1)
        row=SimpleNamespace(element=element,H_known=bool(Hknown))
        model=RelativeInventoryModel(dict(field='inventory'),profile=published_coherent_profile(),pO2_atm=1e-5)
        layer,_=model.layer_for(row)
        check(element+'/'+str(Hknown)+'/raw_plus_derived_states',abs(layer.db.nCS-(int(keep.sum())+2)))
        check(element+'/'+str(Hknown)+'/same_parent_set',int(set(layer.original_parent_ids)!=set(db.cs_ID[keep])))
        check(element+'/'+str(Hknown)+'/current_fully_ionized_branch',int(layer.fixed_fraction!=0.))
        cohorts.append(dict(element=element,H_known=bool(Hknown),input_conditions=len(group),
            supplied_original_energy_entries=int(keep.sum()),native_energy_entries=int((keep&native).sum()),
            foreign_energy_entries=int((keep&~native).sum()),original_parent_count=len(set(db.cs_ID[keep])),
            derived_neutral_rows=2,internal_state_rows=layer.db.nCS,
            derived_neutral_rows_are_new_DFT_labels=False,localized_fraction=layer.fixed_fraction))
    physical=set(v for v in report.NEW if v.startswith('Share') or v=='GrowthReference')
    physical.update(v for v in report.OLD if v=='FixedPhysical' or v.startswith('Inventory') or v.startswith('Output') or v=='PhysicalAffine')
    check('all36_named',abs(len(set(report.VARIANTS))-36))
    costs=[]
    for v in report.VARIANTS:
        physics=v in physical
        for c in cohorts:
            costs.append(dict(variant=v,element=c['element'],H_known=c['H_known'],input_conditions=c['input_conditions'],
                uses_defect_energies=physics,supplied_original_energy_entries=c['supplied_original_energy_entries'] if physics else 0,
                supplied_original_parent_count=c['original_parent_count'] if physics else 0,
                derived_neutral_rows=c['derived_neutral_rows'] if physics else 0,
                new_DFT_jobs=0,new_material_measurements=0,new_mu_training_labels=0,
                Hall_supervision='same33condition task,fold-excluded training;perDOI0/1/2 adaptation',
                cost_meaning='provided charge-entry count;not independent labels,minimal required labels,DFT jobs or compute time'))
    e.csvwrite(OUT/'active_catalogue_counts.csv',cohorts);e.csvwrite(OUT/'method_information_budgets.csv',costs)
    rare=['Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
    available=[x for x in rare if x in db.elements]
    protocol=e.BASE/'phase2_energy/acquisition/protocol.json';acquisition=json.loads(protocol.read_text())
    check('earlier_acquisition_not_new_material_labels',int(not acquisition['no_material_measurement_labels_used']))
    value=dict(passed=all(r['passed'] for r in checks),checks=len(checks),rows=checks,
        raw_catalogue_charge_rows=db.nCS,raw_catalogue_parents=db.nD,raw_catalogue_elements=db.elements,
        trained_prediction_elements=['Si','Sn'],rareearth_elements_in_current_numeric_catalogue=available,
        rareearth_training_or_validated_prediction_established=False,
        active_input_types=len(cohorts),named_methods=36,physical_energy_methods=len(physical),
        direct_or_empirical_no_defect_energy_methods=36-len(physical),budget_rows=len(costs),
        original_energy_entry_counts_not_minimum_required_or_independent_labels=True,
        shared_Hall_and_anchor_budgets_not_same_prior_information=True,
        extra_transport_priors={'physical':'Nc-consistent scalar mass,epsilon_r10.2,fixedextra205cm2Vs,Born/RTA',
            'empirical':'frozen CT mobility formula at predictednH;not a newly supervised mu model'},
        external_proxy_chemical_budget='same12layer/twostack/oneDOI calibration supplied to all36 methods;not12 independent Hall chemistry truths',
        earlier_goal_acquisition_grade=acquisition['validation'],earlier_goal_acquisition_target=acquisition['primary_metric'],
        no_new_DFT_MD=True,no_new_measurements=True,no_fits_or_property_prediction_scores=True,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (Path(__file__),ROOT/'src/sandbox/kroger_db.py',ROOT/'src/sandbox/temperature_defects.py',
            ROOT/'src/sandbox/relative_inventory_model.py',ROOT/'scripts/trainable_sandbox_20261017/phase26_report.py')},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (NPZ,e.BASE/'phase25_shared_process/data/conditions33.csv',protocol,e.BASE/'phase2_energy/label_algebra_audit.json')})
    e.write(OUT/'audit.json',value)
    print('Actual information-budget audit',value['passed'],len(costs),'rows;energy methods',len(physical),'of36',flush=True)
    print('Active original-energy row counts',[(r['element'],r['H_known'],r['supplied_original_energy_entries']) for r in cohorts],flush=True)
    if not value['passed']:raise RuntimeError('Preserve failed information-budget correspondence before repair')


if __name__=='__main__':main()
