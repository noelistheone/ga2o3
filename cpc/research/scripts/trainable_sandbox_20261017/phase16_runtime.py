"""Independent thermal/joint model export, own-temperature property replay."""
from pathlib import Path
import sys,json,argparse,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(tasks_path,output_path):
    forbidden=('dataset_v1.csv','episodes_v1.csv','paired_specimens.csv','temperature_observations.csv',
        '/fits/','/refined_fits/','/inherited/','/methods/','/validation/','/thermal_audit/',
        '/mu_controls/','head_parameters.json')
    blocked=[];opened=[]
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        name=str(args[0]);opened.append(name)
        if any(s in name for s in forbidden):blocked.append(name);raise RuntimeError('Property/fit/selection/audit table read forbidden')
    sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    from types import SimpleNamespace
    from sandbox.thermal_coherent_model import ThermalCoherentHallModel
    from sandbox.coherent_fermi import CoherentMaterialProfile,TABLE
    from sandbox.kroger_db import NPZ
    tasks=json.loads(Path(tasks_path).read_text());rows=[];adaptations=[]
    for task in tasks:
        b=json.loads(Path(task['bundle']).read_text())
        assert sha(NPZ)==b['energetics_sha256'] and sha(TABLE)==b['fd_reference_sha256']
        model=ThermalCoherentHallModel(b['definition'],profile=CoherentMaterialProfile.from_dict(b['material_profile']),pO2_atm=b['pO2_atm'])
        lam=np.inf if b['lambda'] is None else b['lambda'];w=b['parameters'];z=0.
        if task['anchors'] and np.isfinite(lam):
            residual=[model.point(SimpleNamespace(**a['input']),w).values[0]-a['observed_log_nH'] for a in task['anchors']]
            z=float(sum(residual)/(len(residual)+lam))
        adaptations.append(dict(task_id=task['task_id'],source_shift=z))
        for point in task['points']:
            try:
                a=model.point(SimpleNamespace(**point['input']),w,source_kind='offset',source_shift=z)
                g=a.diagnostics
                record=dict(task_id=task['task_id'],obs_id=point['obs_id'],values=a.values.tolist(),transport_values=a.transport_values.tolist(),
                    same_physical_state_for_output=g['same_physical_state_for_output'],Tf_K=g['Tf_K'],
                    free_n_cm3=g['free_n_cm3'],hall_factor=g['hall_factor'],
                    thermal_cap_K=float(point['input']['thermal_cap_K']),error=None)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                record=dict(task_id=task['task_id'],obs_id=point['obs_id'],values=None,transport_values=None,error=repr(exc))
            rows.append(record)
    Path(output_path).write_text(json.dumps(dict(rows=rows,adaptations=adaptations,property_tables_read=False,
        property_table_open_guard_active=True,blocked_paths=blocked,opened_path_count=len(opened)),ensure_ascii=False,indent=2)+'\n')


def parent():
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
    import numpy as np
    import pandas as pd
    from csv_io_v2 import read_result_csv as read
    from sandbox.coherent_fermi import published_coherent_profile,TABLE
    from sandbox.kroger_db import NPZ
    import phase16_audit as audit
    import phase16_refinement as final
    import phase2_energy as energy
    study=final.study;OUT=study.OUT/'research_runtime';bank=final.composite()
    assert json.loads((study.OUT/'validation/contract_audit.json').read_text())['passed']
    protocol=dict(code_sha256=sha(Path(__file__)),
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/sandbox/thermal_coherent_model.py',
            ROOT/'src/sandbox/coherent_hall_model.py',ROOT/'src/sandbox/coherent_fermi.py',ROOT/'src/sandbox/coherent_transport.py')},
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in (study.OUT/'validation/contract_audit.json',
            study.OUT/'prediction_protocol.json',TABLE)},
        scope='12structures/180sourceexcludedbundles;all230nH/muownTpoints;onlyauthorizednRTanchorsintheworker',
        fixed205_empirical_prior_disclosed=True,independent_sigma_labels=0,goal_complete=False)
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==energy.clean(protocol)
    else:energy.write(pp,protocol)
    def inputs(r):
        return dict(element=r.element,incorporated_cm3=float(r.incorporated_cm3),growth_T_C=float(r.growth_T_C),
            Tobs_K=float(r.Tobs_K),H_known=bool(r.H_known),H_cm3=float(r.H_cm3),thermal_cap_K=float(r.thermal_cap_K))
    tasks=[];expected={};index=[];shifts={}
    for method in study.STRUCTURES:
        choices=json.loads((study.OUT/'methods'/method/'selections.json').read_text())
        for c in choices:
            held,budget=c['held'],c['budget'];f=bank[c['fit_key']];assert held not in f['train_families']
            tid=f'{method}::{held}::{budget}';bundle=OUT/'bundles'/(study.core.prior.old.digest(tid)+'.json')
            energy.write(bundle,dict(method=method,held=held,budget=budget,definition=f['definition'],parameters=f['parameters'],
                **{'lambda':c['selected']['lam']},material_profile=published_coherent_profile().to_dict(),pO2_atm=1e-5,
                fit_key=f['key'],fit_path=f['composite_origin']['path'],fit_sha256=f['composite_origin']['sha256'],
                energetics_sha256=sha(NPZ),fd_reference_sha256=sha(TABLE),
                new_mu_supervision=study.STRUCTURES[method]['gamma']>0,mu_training_families=f['metadata']['mu_training_families'],
                thermal_scope='conditional cap or effective coordinate;not actualfilmTmax',fixed205_empirical_prior=True))
            d=study.TARGETS[study.TARGETS.family.eq(held)]
            role=study.core.ROLES[study.core.ROLES.family.eq(held)&study.core.ROLES.budget.eq(budget)].set_index('sample_id').role
            anchors=[dict(input=inputs(r),observed_log_nH=float(r.target)) for r in d[d.endpoint.eq('nH')&d.kind.eq('RT')].itertuples() if role[r.sample_id]=='anchor']
            assert len(anchors)==budget
            tasks.append(dict(task_id=tid,bundle=str(bundle),anchors=anchors,points=[dict(obs_id=r.obs_id,input=inputs(r)) for r in d.itertuples()]))
            index.append(dict(task_id=tid,method=method,held=held,budget=budget,bundle=str(bundle.relative_to(ROOT))))
            _,_,delta=audit.primary(f,held,budget,audit.number(c['selected']['lam']));shifts[tid]=-delta
            lookup={p['obs_id']:p for p in f['predictions']+f['property_predictions']}
            for r in d.itertuples():
                p=lookup[r.obs_id];assert p['predicted'] is not None
                expected[tid,r.obs_id]=dict(values=[p['predicted_log_nH']+delta,*p['charge_outputs']],
                    transport=[p['transport_values'][0],p['transport_values'][1]+delta])
    energy.write(OUT/'tasks.json',tasks);energy.write(OUT/'index.json',index)
    output=OUT/'independent_outputs.json'
    subprocess.run([sys.executable,str(Path(__file__)),'--worker',str(OUT/'tasks.json'),str(output)],cwd=ROOT,check=True)
    actual=json.loads(output.read_text());e=[];et=[];es=[];failures=[]
    for r in actual['rows']:
        if r['error'] is not None:failures.append(r);continue
        wanted=expected[r['task_id'],r['obs_id']]
        e.extend(abs(np.array(r['values'])-wanted['values']));et.extend(abs(np.array(r['transport_values'])-wanted['transport']))
        assert abs(r['transport_values'][1]-(np.log10(1.602176634e-19)+r['values'][0]+r['transport_values'][0]))<1e-10
    for a in actual['adaptations']:es.append(abs(a['source_shift']-shifts[a['task_id']]))
    assert len(tasks)==180 and len(actual['rows'])==8280 and len(expected)==8280
    result=dict(passed=not failures and max(e)<1e-6 and max(et)<1e-6 and max(es)<1e-6,
        saved_bundles=len(tasks),independent_rows=len(actual['rows']),maximum_state_error=max(e),maximum_transport_error=max(et),
        maximum_source_shift_error=max(es),failures=failures,property_table_open_guard_active=True,
        property_tables_read=False,query_labels_in_tasks=False,new_sigma_labels=0,
        source_anchor_kind='nH_RT specimenanchorsonly;no mu labels',goal_complete=False)
    energy.write(OUT/'verification.json',result);assert result['passed'],result
    print('Thermal joint runtime',len(tasks),'bundles',len(actual['rows']),'rows','max',max(e),max(et))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',nargs=2);a=ap.parse_args()
    if a.worker:worker(*a.worker)
    else:parent()


if __name__=='__main__':main()
