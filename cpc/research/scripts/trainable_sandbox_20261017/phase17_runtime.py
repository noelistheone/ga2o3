"""Export and independently replay conserved inventory and direct controls."""
from pathlib import Path
import sys,json,argparse,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(tasks_path,output_path):
    forbidden=('dataset_v1.csv','episodes_v1.csv','paired_specimens.csv','temperature_observations.csv',
        '/fits/','/refined_fits/','/inherited/','/methods/','/validation/','/inventory_audit/','/thermal_audit/')
    blocked=[];opened=[]
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        name=str(args[0]);opened.append(name)
        if any(s in name for s in forbidden):blocked.append(name);raise RuntimeError('Property/fit/selection table read forbidden')
    sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    from scipy.special import expit
    from types import SimpleNamespace
    from sandbox.inventory_coherent_model import InventoryCoherentModel
    from sandbox.coherent_fermi import CoherentMaterialProfile,TABLE
    from sandbox.kroger_db import NPZ
    from sandbox import transport
    tasks=json.loads(Path(tasks_path).read_text());rows=[];adaptations=[]
    def features(d):
        C=np.array([r['incorporated_cm3'] for r in d]);T=np.array([r['Tobs_K'] for r in d]);c=np.log10(C)-18.
        hk=np.array([r['H_known'] for r in d],float);ck=np.array([r['carbon_known'] for r in d],float)
        H=np.array([r['H_cm3'] for r in d]);carbon=np.array([r['carbon_cm3'] for r in d])
        return np.c_[c,c*c,(np.array([r['growth_T_C'] for r in d])+273.15-1100)/300,
            (np.array([r['thermal_cap_K'] for r in d])-1100)/300,np.array([r['element']=='Sn' for r in d],float),
            np.log(T/300),300/T-1,c*(300/T-1),hk,(np.log10(np.maximum(H,1))-17)*hk,ck,(np.log10(np.maximum(carbon,1))-17)*ck]
    def empirical(b,inputs):
        d=b['definition'];w=b['parameters'];C=np.array([r['incorporated_cm3'] for r in inputs]);H=np.array([r['H_cm3'] for r in inputs]);x=np.log10(C)-18.
        if d['empirical']=='ridge':
            meta=b['metadata'];return ((features(inputs)-meta['center'])/meta['scale'])@np.array(meta['coefficients'])+meta['intercept']
        if d['empirical']=='activation_sigmoid':return np.log10(C+H+10**w[0])-np.logaddexp(0.,-w[1]-w[2]*x)/np.log(10.)
        relation=d['relation'];phi=np.zeros(len(C)) if relation=='constant' else np.logaddexp(0.,x if relation=='dose_plus' else -x)
        s=w[1]+(w[2]*phi if len(w)>2 else 0.)
        return np.log10(10**w[0]+(C+H)*(-np.expm1(-s)))
    def ct(logn,T):
        ceiling=transport.CT_MU_MAX*float(transport.mu_pop_hall(T)/transport.mu_pop_hall(300.))
        inverse=expit(-transport.CT_ALPHA*np.log(10.)*(logn-np.log10(transport.CT_N_REF)))
        return float(np.log10(transport.CT_MU_MIN+(ceiling-transport.CT_MU_MIN)*inverse))
    for task in tasks:
        b=json.loads(Path(task['bundle']).read_text());assert sha(NPZ)==b['energetics_sha256'] and sha(TABLE)==b['fd_reference_sha256']
        physical='empirical' not in b['definition']
        model=InventoryCoherentModel(b['definition'],profile=CoherentMaterialProfile.from_dict(b['profile']),pO2_atm=b['pO2_atm']) if physical else None
        lam=np.inf if b['lambda'] is None else b['lambda'];delta=0.
        if task['anchors'] and np.isfinite(lam):
            inputs=[a['input'] for a in task['anchors']]
            raw=[model.point(SimpleNamespace(**r),b['parameters']).values[0] for r in inputs] if physical else empirical(b,inputs)
            delta=float(sum(a['observed_log_nH']-p for a,p in zip(task['anchors'],raw))/(len(raw)+lam))
        adaptations.append(dict(task_id=task['task_id'],delta=delta))
        empirical_values=None if physical else empirical(b,[p['input'] for p in task['points']])
        for j,p in enumerate(task['points']):
            try:
                inp=p['input']
                if physical:
                    a=model.point(SimpleNamespace(**inp),b['parameters'],source_kind='offset',source_shift=-delta);g=a.diagnostics
                    values=a.values.tolist();transport_values=a.transport_values.tolist()
                    dg={k:g[k] for k in ('same_physical_state_for_output','free_n_cm3','Tf_K','hall_factor','native_survival','complex_survival',
                        'complex_reallocation_fraction','foreign_atom_relative_error','foreign_atom_derivative_error','host_Ga_O_inventory_change_cm3')}
                    dg['physical_log_nH']=float(a.physical_values[0]);dg['physical_log_muH']=g['physical_transport_values'][0]
                    dg['free_screen_cm3']=g['coherent_transport']['n_screen_cm3']
                    layer,solver=model.layer_for(SimpleNamespace(**inp));pop=solver.last_prediction;stock=solver.last_inventory.totals
                    mq=np.bincount(layer.parent,weights=pop.probabilities*layer.q,minlength=layer.db.nD)
                    chi=np.bincount(layer.parent,weights=pop.probabilities*(layer.q-mq[layer.parent])**2,minlength=layer.db.nD)
                    ell=g['localized_fraction']
                    for plus,neutral in layer.pairs:
                        f=pop.probabilities[neutral]/ell if ell>0 else 0.;chi[layer.parent[plus]]=ell*f*(1-f)
                    dg['bound_equilibrium_response_over_free_screen']=float((stock@chi)/dg['free_screen_cm3'])
                    dg['max_native_parent_fraction']=float(max(stock[solver.native]/layer.db.prefactor[layer.first[solver.native]]))
                else:
                    n=float(empirical_values[j]+delta);mu=ct(n,float(inp['Tobs_K']));values=[n,None,None]
                    transport_values=[mu,float(np.log10(transport.E)+n+mu)]
                    dg=dict(same_physical_state_for_output=False,empirical_CT_uses_calibrated_predicted_nH=True,
                        outside_CT_declared_temperature_calibration=not 300<=inp['Tobs_K']<=500)
                if not np.isfinite([values[0],*transport_values]).all():raise ValueError('Nonfinite property prediction')
                rows.append(dict(task_id=task['task_id'],obs_id=p['obs_id'],values=values,transport_values=transport_values,diagnostics=dg,error=None))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                rows.append(dict(task_id=task['task_id'],obs_id=p['obs_id'],values=None,transport_values=None,error=repr(exc)))
    Path(output_path).write_text(json.dumps(dict(rows=rows,adaptations=adaptations,property_tables_read=False,
        property_table_open_guard_active=True,blocked_paths=blocked,opened_path_count=len(opened)),ensure_ascii=False,indent=2)+'\n')


def parent():
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
    import numpy as np
    from sandbox.coherent_fermi import published_coherent_profile,TABLE
    from sandbox.kroger_db import NPZ
    from sandbox import transport
    import phase17_audit as audit
    import phase2_energy as energy
    final=audit.final;study=final.study;OUT=study.OUT/'research_runtime';bank=final.composite()
    assert json.loads((study.OUT/'validation/contract_audit.json').read_text())['passed']
    protocol=dict(code_sha256=sha(Path(__file__)),
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/sandbox/inventory_coherent_model.py',ROOT/'src/sandbox/fixed_inventory_charge.py')},
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in (study.OUT/'validation/contract_audit.json',study.OUT/'prediction_protocol.json',TABLE)},
        scope='all31models/groups,465bundles,all230ownTn/mu points;onlypermittedRTnHanchors;no querylabels',
        CT_control='after nH sourcecalibration,recomputeCT(nHpred,T) fordirectcontrols;physicalmodels retainstate-mu andempiricalnHoffset;bothdisclosed',
        property_scope='nHactualdoseprediction,muRT5/2,muT54/3/1;independentsigma0',goal_complete=False)
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==energy.clean(protocol)
    else:energy.write(p,protocol)
    def inputs(r):
        return dict(element=r.element,incorporated_cm3=float(r.incorporated_cm3),growth_T_C=float(r.growth_T_C),Tobs_K=float(r.Tobs_K),
            thermal_cap_K=float(r.thermal_cap_K),H_known=bool(r.H_known),H_cm3=float(r.H_cm3),
            carbon_known=bool(r.carbon_known),carbon_cm3=float(r.carbon_cm3))
    tasks=[];index=[];expected={};offsets={}
    for method in audit.ALL_METHODS:
        choices=json.loads((study.OUT/'methods'/method/'selections.json').read_text())
        for c in choices:
            held,budget=c['held'],c['budget'];f=bank[c['fit_key']];assert held not in f['train_families']
            tid=f'{method}::{held}::{budget}';bp=OUT/'bundles'/(study.core.prior.old.digest(tid)+'.json')
            energy.write(bp,dict(method=method,selected_member=f['method'],held=held,budget=budget,parameters=f['parameters'],definition=f['definition'],
                metadata=f['metadata'],**{'lambda':c['selected']['lam']},profile=published_coherent_profile().to_dict(),pO2_atm=1e-5,
                fit_key=f['key'],fit_origin=f['composite_origin'],energetics_sha256=sha(NPZ),fd_reference_sha256=sha(TABLE),
                new_mu_training_labels=False,temperature_interpretation='conditional recorded bounds,not actualthermalhistory'))
            d=study.TARGETS[study.TARGETS.family.eq(held)];roles=study.core.ROLES[study.core.ROLES.family.eq(held)&study.core.ROLES.budget.eq(budget)].set_index('sample_id').role
            anchors=[dict(input=inputs(r),observed_log_nH=float(r.target)) for r in d[d.endpoint.eq('nH')&d.kind.eq('RT')].itertuples() if roles[r.sample_id]=='anchor']
            assert len(anchors)==budget
            tasks.append(dict(task_id=tid,bundle=str(bp),anchors=anchors,points=[dict(obs_id=r.obs_id,input=inputs(r)) for r in d.itertuples()]))
            index.append(dict(task_id=tid,method=method,selected_member=f['method'],held=held,budget=budget,bundle=str(bp.relative_to(ROOT))))
            _,_,delta=audit.primary(f,held,budget,audit.number(c['selected']['lam']));offsets[tid]=delta
            lookup={r['obs_id']:r for r in f['predictions']+f['property_predictions']}
            for r in d.itertuples():
                p=lookup[r.obs_id];physical='empirical' not in f['definition'];n=p['predicted_log_nH']+delta
                if physical:mu,sigma=p['transport_values'];sigma+=delta
                else:mu=float(np.log10(transport.mu_caughey_thomas(10**n,float(r.Tobs_K))));sigma=float(np.log10(transport.E)+n+mu)
                expected[tid,r.obs_id]=dict(n=n,charges=p['charge_outputs'],transport=[mu,sigma])
    energy.write(OUT/'tasks.json',tasks);energy.write(OUT/'index.json',index)
    output=OUT/'independent_outputs.json'
    subprocess.run([sys.executable,str(Path(__file__)),'--worker',str(OUT/'tasks.json'),str(output)],cwd=ROOT,check=True)
    actual=json.loads(output.read_text());ne=[];ce=[];te=[];de=[];failures=[]
    for r in actual['rows']:
        if r['error'] is not None:failures.append(r);continue
        p=expected[r['task_id'],r['obs_id']];ne.append(abs(r['values'][0]-p['n']));te.extend(abs(np.array(r['transport_values'])-p['transport']))
        if p['charges'] is not None:ce.extend(abs(np.array(r['values'][1:])-p['charges']))
        assert abs(r['transport_values'][1]-(np.log10(transport.E)+r['values'][0]+r['transport_values'][0]))<1e-10
    for r in actual['adaptations']:de.append(abs(r['delta']-offsets[r['task_id']]))
    assert len(tasks)==465 and len(actual['rows'])==21390 and len(expected)==21390
    result=dict(passed=not failures and max(ne)<2e-6 and max(ce)<2e-6 and max(te)<2e-6 and max(de)<2e-6,
        saved_bundles=len(tasks),independent_rows=len(actual['rows']),maximum_carrier_error=max(ne),maximum_charge_error=max(ce),
        maximum_transport_error=max(te),maximum_source_offset_error=max(de),failures=failures,
        property_table_open_guard_active=True,property_tables_read=False,query_labels_in_tasks=False,
        empirical_CT_uses_calibrated_predicted_nH=True,new_mu_training_labels=False,new_sigma_labels=0,goal_complete=False)
    energy.write(OUT/'verification.json',result);assert result['passed'],result
    print('Conserved inventory replay',len(tasks),'bundles',len(actual['rows']),'rows','max',max(ne),max(te))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',nargs=2);a=ap.parse_args()
    if a.worker:worker(*a.worker)
    else:parent()


if __name__=='__main__':main()
