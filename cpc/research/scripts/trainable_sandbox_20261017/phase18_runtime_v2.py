"""Export and independently replay fixed and re-inferred source predictions."""
from pathlib import Path
import sys,json,argparse,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(tasks_path,output_path):
    forbidden=('dataset_v1.csv','episodes_v1.csv','paired_specimens.csv','temperature_observations.csv','/fits/','/refined_fits/',
        '/inherited/','/methods/','/validation/','/thermal_audit/','/inventory_audit/','/source_adaptation_v2/')
    blocked=[]
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)) and any(s in str(args[0]) for s in forbidden):
            blocked.append(str(args[0]));raise RuntimeError('Experimental/fit/selection/source table read forbidden')
    sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    from sandbox.logodds_predictor import LogOddsResearchPredictor
    rows=[];adaptations=[]
    for task in json.loads(Path(tasks_path).read_text()):
        b=json.loads(Path(task['bundle']).read_text());predictor=LogOddsResearchPredictor(b)
        lam=np.inf if b['lambda'] is None else float(b['lambda']);cal=predictor.calibrate(task['anchors'],b['source_kind'],lam)
        fixed=dict(kind=b['source_kind'],shift=b['saved_source_shift']);adaptations.append(dict(task_id=task['task_id'],kind=b['source_kind'],
            saved_shift=b['saved_source_shift'],refit_shift=cal['shift'],refit_search=cal['search']))
        for point in task['points']:
            try:
                f=predictor.predict(point['input'],fixed)
                r=f if fixed['shift']==cal['shift'] else predictor.predict(point['input'],cal)
                rows.append(dict(task_id=task['task_id'],obs_id=point['obs_id'],**f,refit_values=r['values'],refit_transport_values=r['transport_values'],error=None))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:rows.append(dict(task_id=task['task_id'],obs_id=point['obs_id'],error=repr(exc)))
    def encode(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    Path(output_path).write_text(json.dumps(dict(rows=rows,adaptations=adaptations,blocked_paths=blocked),allow_nan=False,default=encode))


def parent():
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
    import numpy as np
    from sandbox.coherent_fermi import published_coherent_profile,TABLE
    from sandbox.kroger_db import NPZ
    import phase18_audit as audit
    import phase2_energy as energy
    source=audit.source;study=audit.study;OUT=study.OUT/'research_runtime_v2';bank=audit.final.composite()
    assert json.loads((study.OUT/'validation/contract_audit.json').read_text())['passed']
    assert json.loads((source.OUT/'validation/contract_audit.json').read_text())['passed']
    protocol=dict(code_sha256=sha(Path(__file__)),code_hashes={str(p.relative_to(ROOT)):sha(p) for p in
        (ROOT/'src/sandbox/logodds_predictor.py',ROOT/'src/sandbox/logodds_inventory_model.py',ROOT/'src/sandbox/bounded_source_learning.py')},
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in (study.OUT/'prediction_protocol.json',study.OUT/'validation/contract_audit.json',
            source.OUT/'source_protocol.json',source.OUT/'validation/contract_audit.json',TABLE)},
        methods=14,expected_bundles=210,expected_rows=9660,querylabels=False,
        fixed_shift_tolerance_dex=2e-6,independent_source_refit_tolerance_dex=2e-4,
        source_precision='fixed-state2e-6 inherited;finite scalarsearch xatol1e-5 refit2e-4;coordinate drift reported separately,not identifiedtruth',
        anchors='only0/1/2RTspecimenanchors;legacypercurveTanchortask is separate',goal_complete=False)
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==energy.clean(protocol)
    else:energy.write(pp,protocol)
    def inputs(r):return dict(element=r.element,incorporated_cm3=float(r.incorporated_cm3),growth_T_C=float(r.growth_T_C),Tobs_K=float(r.Tobs_K),
        thermal_cap_K=float(r.thermal_cap_K),H_known=bool(r.H_known),H_cm3=float(r.H_cm3),carbon_known=bool(r.carbon_known),carbon_cm3=float(r.carbon_cm3))
    tasks=[];index=[];expected={}
    plans=[(m,study.OUT/'methods'/m,False) for m in study.STRUCTURES]+[(m,source.OUT/'methods'/m,True) for m in source.VARIANTS]
    for method,folder,internal in plans:
        property_rows=json.loads((folder/'property_predictions.json').read_text()) if internal else None
        for c in json.loads((folder/'selections.json').read_text()):
            held,budget=c['held'],c['budget'];f=bank[c['fit_key']];assert held not in f['train_families']
            if internal:shift=c['source_z'];kind='logodds'
            else:_,_,shift=audit.primary(f,held,budget,audit.number(c['selected']['lam']));kind='offset'
            tid=f'{method}::{held}::{budget}';bp=OUT/'bundles'/(study.core.prior.old.digest(tid)+'.json')
            energy.write(bp,dict(method=method,base_method=f['method'],held=held,budget=budget,definition=f['definition'],parameters=f['parameters'],
                **{'lambda':c['selected']['lam']},source_kind=kind,saved_source_shift=shift,profile=published_coherent_profile().to_dict(),pO2_atm=1e-5,
                fit_key=f['key'],fit_origin=f['composite_origin'],energetics_sha256=sha(NPZ),fd_reference_sha256=sha(TABLE),
                scope='conditionalSi/Snresearchpredictor,not measured microscopicinventory or universaltransportvalidation'))
            d=study.TARGETS[study.TARGETS.family.eq(held)];roles=study.core.ROLES[study.core.ROLES.family.eq(held)&study.core.ROLES.budget.eq(budget)].set_index('sample_id').role
            anchors=[dict(input=inputs(r),observed_log_nH=float(r.target)) for r in d[d.endpoint.eq('nH')&d.kind.eq('RT')].itertuples() if roles[r.sample_id]=='anchor'];assert len(anchors)==budget
            tasks.append(dict(task_id=tid,bundle=str(bp),anchors=anchors,points=[dict(obs_id=r.obs_id,input=inputs(r)) for r in d.itertuples()]))
            index.append(dict(task_id=tid,method=method,base_method=f['method'],held=held,budget=budget,source_kind=kind,bundle=str(bp.relative_to(ROOT))))
            look={r['obs_id']:r for r in property_rows if r['family']==held and r['budget']==budget} if internal else {r['obs_id']:r for r in f['predictions']+f['property_predictions']}
            for r in d.itertuples():
                p=look[r.obs_id];n=p['predicted_log_nH']+(0 if internal else shift);tv=list(p['transport_values'])
                if not internal:
                    if 'empirical' in f['definition']:
                        tv[0]=float(np.log10(study.transport.mu_caughey_thomas(10**n,float(r.Tobs_K))));tv[1]=float(np.log10(study.transport.E)+n+tv[0])
                    else:tv[1]+=shift
                expected[tid,r.obs_id]=dict(n=n,charges=p['charge_outputs'],transport=tv)
    energy.write(OUT/'tasks.json',tasks);energy.write(OUT/'index.json',index);output=OUT/'independent_outputs.json'
    subprocess.run([sys.executable,str(Path(__file__)),'--worker',str(OUT/'tasks.json'),str(output)],cwd=ROOT,check=True)
    actual=json.loads(output.read_text());ne=[];ce=[];te=[];rn=[];rt=[];failures=[]
    for r in actual['rows']:
        if r['error'] is not None:failures.append(r);continue
        e=expected[r['task_id'],r['obs_id']];ne.append(abs(r['values'][0]-e['n']));te.extend(abs(np.array(r['transport_values'])-e['transport']))
        if e['charges'] is not None:ce.extend(abs(np.array(r['values'][1:])-e['charges']))
        rn.append(abs(r['refit_values'][0]-e['n']));rt.extend(abs(np.array(r['refit_transport_values'])-e['transport']))
        for n,t in [(r['values'][0],r['transport_values']),(r['refit_values'][0],r['refit_transport_values'])]:assert abs(t[1]-np.log10(study.transport.E)-n-t[0])<1e-10
    shifts={k:max([abs(r['saved_shift']-r['refit_shift']) for r in actual['adaptations'] if r['kind']==k]) for k in ('offset','logodds')}
    assert len(tasks)==210 and len(actual['rows'])==9660 and len(expected)==9660
    result=dict(passed=not failures and max(ne)<2e-6 and max(te)<2e-6 and max(ce)<2e-6 and max(rn)<2e-4 and max(rt)<2e-4,
        saved_bundles=210,independent_rows=9660,maximum_fixed_carrier_error=max(ne),maximum_fixed_charge_error=max(ce),maximum_fixed_transport_error=max(te),
        maximum_source_refit_carrier_error=max(rn),maximum_source_refit_transport_error=max(rt),maximum_offset_coordinate_error=shifts['offset'],
        maximum_logodds_coordinate_error=shifts['logodds'],failures=failures,property_tables_read=False,query_labels_in_tasks=False,
        sameRTspecimenanchors_for_allownToutputs=True,legacy_percurve_Tanchor_predictions_not_claimed_replayed_here=True,
        new_mu_supervision=False,new_sigma_labels=0,goal_complete=False)
    energy.write(OUT/'verification.json',result);print(json.dumps(energy.clean(result),indent=2));assert result['passed'],result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',nargs=2);args=ap.parse_args()
    if args.worker:worker(*args.worker)
    else:parent()


if __name__=='__main__':main()
